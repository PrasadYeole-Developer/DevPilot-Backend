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
