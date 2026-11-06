"""
Long-term, cross-conversation memory (Milestone 7).

Backed by `MongoDBStore` (the official `langgraph-store-mongodb` package),
used in plain filter/namespace mode rather than its semantic `query` mode.

Why filter mode, not semantic search: MongoDBStore's query-based search
requires an Atlas Vector Search index (auto-created on first use) plus an
embedding call on every write/read. That's real operational complexity
(index build time, a second embedding pipeline alongside the one already
used for repository code search) for a feature this milestone doesn't
actually need — "retrieve relevant memories" is satisfied by retrieving
everything in a repository's namespace, not by semantic ranking, because the
memory volume per (session, repository) is small and deliberately curated
(see `remember()` — only deterministic, high-value facts get stored at all).
Filter/namespace mode is officially supported (`index_config` is optional on
MongoDBStore) and needs no index and no embedding model, which also makes it
trivially mockable for offline tests.

Namespace shape: (session_id, owner, repo) — repository_id ("owner/repo") is
split into two segments because MongoDBStore forbids "/" inside a single
namespace segment (confirmed via a live Atlas integration test, where a
real repository_id tripped this validation; the offline fakes didn't catch
it since they don't replicate that check). This directly satisfies "do not
mix memories belonging to different users or repositories" — a search
against namespace_prefix=(session_id, owner, repo) can structurally never
return another session's or repository's memories; there is no filter to
forget.
"""

import hashlib
import re
from typing import Any, Literal

from langgraph.store.base import BaseStore

MemoryKind = Literal["confirmed_finding", "preference", "note"]

DEFAULT_RECALL_LIMIT = 20

# Defensive guard against ever persisting an actual secret VALUE. Every
# current caller only ever passes LLM-derived descriptions of code behavior
# (never raw environment/config values) — this is a safety net, not the
# primary control.
#
# Deliberately matches secret-*shaped* values (key=value assignments,
# embedded connection-string credentials, known key-prefix formats, bearer
# tokens), not bare words like "token"/"password"/"secret" alone — a finding
# like "sessions are validated by checking token presence" is a completely
# legitimate, benign thing to remember about auth code, and must not be
# blocked just for naming the concept.
_SECRET_PATTERN = re.compile(
    r"("
    r"(api[_-]?key|secret|password|token)\s*[:=]\s*\S+"  # key=value / key: value assignment
    r"|bearer\s+[A-Za-z0-9_\-.]{10,}"  # "Bearer <actual token>"
    r"|mongodb(\+srv)?://[^/\s]*:[^/\s]*@"  # connection string with embedded user:pass
    r"|AIza[0-9A-Za-z_\-]{10,}"  # Google API key prefix
    r"|sk-[A-Za-z0-9]{10,}"  # common vendor secret-key prefix
    r")",
    re.IGNORECASE,
)


class MemoryStoreError(Exception):
    """Raised when a long-term-memory operation fails, or would store unsafe content."""


def _namespace(session_id: str, repository_id: str) -> tuple[str, str, str]:
    owner, _, repo = repository_id.partition("/")
    return (session_id, owner, repo)


class LongTermMemoryService:
    def __init__(self, store: BaseStore):
        self._store = store

    async def remember(
        self,
        session_id: str,
        repository_id: str,
        content: str,
        kind: MemoryKind,
        source: str,
    ) -> str:
        """
        Upsert a memory, deduplicated by an exact-content key: storing the same
        finding again just refreshes it (MongoDBStore tracks created_at/updated_at
        itself) rather than creating a duplicate row. This is an intentionally
        simple exact-match dedup — near-duplicate phrasing of the same fact is
        not merged; see README for that tradeoff.
        """
        if _SECRET_PATTERN.search(content):
            raise MemoryStoreError(
                "Refusing to store a memory that looks like it contains a secret/credential."
            )

        namespace = _namespace(session_id, repository_id)
        key = hashlib.sha256(f"{kind}:{content}".encode("utf-8")).hexdigest()[:24]
        value = {"content": content, "kind": kind, "source": source}
        await self._store.aput(namespace, key, value, index=False)
        return key

    async def recall(
        self, session_id: str, repository_id: str, limit: int = DEFAULT_RECALL_LIMIT
    ) -> list[dict[str, Any]]:
        items = await self._store.asearch(_namespace(session_id, repository_id), limit=limit)
        return [
            {
                "key": item.key,
                "content": item.value.get("content"),
                "kind": item.value.get("kind"),
                "source": item.value.get("source"),
                "updated_at": item.updated_at,
            }
            for item in items
        ]

    async def forget(self, session_id: str, repository_id: str, key: str) -> None:
        await self._store.adelete(_namespace(session_id, repository_id), key)
