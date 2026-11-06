"""Shared fixtures for offline tests — no real Gemini, Pinecone, or GitHub calls."""

import itertools

import pytest

from app.ingestion.store import IngestedRepository, RepositoryStore, RetrievedFile
from app.retrieval.search_service import SearchMatch


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def sample_store() -> RepositoryStore:
    """A repository store pre-populated with a small, realistic-looking bug scenario."""
    store = RepositoryStore()
    store.save(
        IngestedRepository(
            owner="acme",
            repo="webapp",
            default_branch="main",
            files=[
                RetrievedFile(
                    path="app/auth/session.py",
                    content=(
                        "def is_session_valid(user):\n"
                        "    # Only checks that a session token exists.\n"
                        "    return user.session_token is not None\n"
                    ),
                    size=120,
                    sha="sha-session",
                ),
                RetrievedFile(
                    path="app/auth/deactivate.py",
                    content=(
                        "def deactivate_user(user):\n"
                        "    user.is_active = False\n"
                        "    db.save(user)\n"
                        "    # Existing sessions are not revoked here.\n"
                    ),
                    size=100,
                    sha="sha-deactivate",
                ),
                RetrievedFile(
                    path="README.md",
                    content="# Acme Webapp\n\nSample internal app for testing DevPilot.\n",
                    size=50,
                    sha="sha-readme",
                ),
            ],
        )
    )
    return store


class FakeSearchService:
    """Substring-matching stand-in for RepositorySearchService — no embeddings/Pinecone involved."""

    def __init__(self, store: RepositoryStore):
        self._store = store
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, top_k: int) -> list[SearchMatch]:
        self.calls.append((query, top_k))
        repository = self._store.get_current()
        terms = [t for t in query.lower().split() if t]
        results = []
        for file in repository.files:
            content_lower = file.content.lower()
            if any(term in content_lower for term in terms):
                lines = file.content.splitlines()
                results.append(
                    SearchMatch(
                        path=file.path,
                        chunk_index=0,
                        start_line=1,
                        end_line=len(lines),
                        score=0.8,
                        text=file.content,
                    )
                )
        return results[:top_k]


@pytest.fixture
def fake_search_service(sample_store: RepositoryStore) -> FakeSearchService:
    return FakeSearchService(sample_store)


class _StructuredCall:
    """Stand-in for `chat_model.with_structured_output(Schema)` — returns scripted values in order."""

    def __init__(self, iterator):
        self._iterator = iterator

    async def ainvoke(self, messages):
        return next(self._iterator)


class ScriptedChatModel:
    """
    Fully deterministic fake chat model for offline graph tests.

    - `structured_responses`: {SchemaClass: single_instance_or_list_of_instances}.
      A single instance is returned on every call; a list is consumed one per call
      (so later rounds can return different values, e.g. EvidenceAssessment going
      from insufficient -> sufficient across investigation rounds).
    - `tool_responses`: list of AIMessage objects returned in order by the
      tools-bound model's `.ainvoke` (used by investigate_agent/agent_node).
    """

    def __init__(self, structured_responses: dict, tool_responses: list):
        self._structured_iterators = {
            schema: iter(value) if isinstance(value, list) else itertools.repeat(value)
            for schema, value in structured_responses.items()
        }
        self._tool_responses = iter(tool_responses)
        self.tool_call_count = 0

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema):
        return _StructuredCall(self._structured_iterators[schema])

    async def ainvoke(self, messages):
        self.tool_call_count += 1
        return next(self._tool_responses)


# --- Milestone 7: fake Mongo-ish infrastructure ---
#
# No real MongoDB and no mongomock dependency: these are minimal in-memory
# stand-ins for exactly the pymongo Collection methods / langgraph.store
# BaseStore methods our own code actually calls, so tests exercise our
# logic (ConversationStore, LongTermMemoryService, route ownership checks)
# without depending on MongoDB's wire protocol at all.


class _FakeCursor:
    def __init__(self, docs: list[dict]):
        self._docs = docs

    def sort(self, field, direction):
        self._docs = sorted(self._docs, key=lambda d: d.get(field), reverse=(direction == -1))
        return self

    def skip(self, n):
        self._docs = self._docs[n:]
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def __iter__(self):
        return iter(self._docs)


class FakeCollection:
    """
    Minimal in-memory stand-in for a pymongo Collection — just enough for
    ConversationStore/ProposalStore/ProposalAuditStore's actual method calls.
    """

    def __init__(self):
        self.docs: dict[str, dict] = {}
        self._auto_id = 0

    def insert_one(self, document: dict):
        if "_id" not in document:
            self._auto_id += 1
            document = {**document, "_id": self._auto_id}
        self.docs[document["_id"]] = dict(document)

    def find_one(self, query: dict):
        doc = self.docs.get(query.get("_id"))
        return dict(doc) if doc else None

    def find(self, query: dict):
        matched = [dict(d) for d in self.docs.values() if all(d.get(k) == v for k, v in query.items())]
        return _FakeCursor(matched)

    def count_documents(self, query: dict) -> int:
        return sum(1 for d in self.docs.values() if all(d.get(k) == v for k, v in query.items()))

    def update_one(self, query: dict, update: dict):
        doc = self.docs.get(query.get("_id"))
        if doc:
            doc.update(update.get("$set", {}))

    def replace_one(self, query: dict, replacement: dict, upsert: bool = False):
        existed = query.get("_id") in self.docs
        if existed or upsert:
            self.docs[query["_id"]] = dict(replacement)

        class _Result:
            matched_count = 1 if existed else 0
            upserted_id = None if existed else query.get("_id")

        return _Result()

    def delete_one(self, query: dict):
        existed = query.get("_id") in self.docs
        self.docs.pop(query.get("_id"), None)

        class _Result:
            deleted_count = 1 if existed else 0

        return _Result()


@pytest.fixture
def fake_conversations_collection() -> FakeCollection:
    return FakeCollection()


class _FakeStoreItem:
    def __init__(self, key, value, updated_at):
        self.key = key
        self.value = value
        self.updated_at = updated_at


class FakeLongTermStore:
    """
    Minimal in-memory stand-in for langgraph.store.base.BaseStore (aget/aput/
    asearch/adelete only). Validates namespace segments the same way the real
    MongoDBStore does (no "/" in any segment) — a live Atlas test caught a
    real bug (repository_id="owner/repo" used as a single namespace segment)
    that this fake originally let slide silently; it now replicates that
    check so the offline suite would catch the same class of bug.
    """

    def __init__(self):
        self.data: dict[tuple, dict[str, dict]] = {}

    def _validate_namespace(self, namespace: tuple) -> None:
        for part in namespace:
            if "/" in part:
                raise ValueError(f"Namespace parts must not contain the separator '/'. Got namespace: {namespace}")

    async def aget(self, namespace: tuple, key: str):
        self._validate_namespace(namespace)
        bucket = self.data.get(namespace, {})
        if key not in bucket:
            return None
        return _FakeStoreItem(key, bucket[key], updated_at=None)

    async def aput(self, namespace: tuple, key: str, value: dict, index=None, **kwargs):
        self._validate_namespace(namespace)
        self.data.setdefault(namespace, {})[key] = dict(value)

    async def asearch(self, namespace_prefix: tuple, *, query=None, filter=None, limit=10, offset=0, **kwargs):
        self._validate_namespace(namespace_prefix)
        bucket = self.data.get(namespace_prefix, {})
        items = [_FakeStoreItem(k, v, updated_at=None) for k, v in bucket.items()]
        return items[offset : offset + limit]

    async def adelete(self, namespace: tuple, key: str):
        self._validate_namespace(namespace)
        self.data.get(namespace, {}).pop(key, None)


@pytest.fixture
def fake_long_term_store() -> FakeLongTermStore:
    return FakeLongTermStore()
