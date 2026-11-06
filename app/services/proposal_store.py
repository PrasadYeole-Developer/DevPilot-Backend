"""
Proposal persistence (Milestone 8) — the `proposals` and `proposal_audit`
collections.

Mirrors app.services.conversation_store's pattern: a synchronous pymongo
Collection wrapped with asyncio.to_thread, decoupled from the LangGraph
checkpoint internals (which persist the *graph's* pending/paused state,
not the API-facing proposal document).

Audit records are append-only (ProposalAuditStore.record only ever inserts)
— approving/rejecting a proposal updates the `proposals` document's status
but never overwrites or removes a prior audit entry, satisfying "audit
records must not be silently overwritten."
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from app.schemas.proposal import ProposalResponse


class ProposalStoreError(Exception):
    """Raised when a proposal persistence operation fails."""


class ProposalStore:
    def __init__(self, collection: Collection):
        self._collection = collection

    async def save(self, proposal: ProposalResponse, session_id: str) -> None:
        """
        Upsert — called at every status transition (DRAFT -> ... ->
        APPLIED/REJECTED/FAILED). `session_id` is stored alongside the
        proposal (denormalized from its owning conversation) purely so
        listing/ownership checks here work the same simple way
        app.services.conversation_store's do, without a join.
        """
        document = proposal.model_dump(mode="json")
        # mode="json" turns created_at/updated_at into ISO strings (needed so
        # nested Pydantic models like ProposedFileChange serialize to plain
        # dicts pymongo can store) -- restore the real datetime objects
        # afterward so they round-trip as native BSON dates, same as
        # app.services.conversation_store's timestamps, not strings a
        # Pydantic model's datetime field would later have to be force-fed.
        document["created_at"] = proposal.created_at
        document["updated_at"] = proposal.updated_at
        document["_id"] = proposal.proposal_id
        document["session_id"] = session_id
        try:
            await asyncio.to_thread(
                self._collection.replace_one, {"_id": proposal.proposal_id}, document, True
            )
        except PyMongoError as exc:
            raise ProposalStoreError(f"Failed to save proposal: {exc}") from exc

    async def get(self, proposal_id: str) -> Optional[dict[str, Any]]:
        try:
            return await asyncio.to_thread(self._collection.find_one, {"_id": proposal_id})
        except PyMongoError as exc:
            raise ProposalStoreError(f"Failed to fetch proposal: {exc}") from exc

    async def list_for_session(
        self, session_id: str, repository_id: Optional[str] = None, limit: int = 20, offset: int = 0
    ) -> tuple[list[dict[str, Any]], int]:
        query: dict[str, Any] = {"session_id": session_id}
        if repository_id is not None:
            query["repository_id"] = repository_id

        def _run() -> tuple[list[dict[str, Any]], int]:
            total = self._collection.count_documents(query)
            cursor = self._collection.find(query).sort("updated_at", -1).skip(offset).limit(limit)
            return list(cursor), total

        try:
            return await asyncio.to_thread(_run)
        except PyMongoError as exc:
            raise ProposalStoreError(f"Failed to list proposals: {exc}") from exc


class ProposalAuditStore:
    def __init__(self, collection: Collection):
        self._collection = collection

    async def record(self, proposal_id: str, event: str, session_id: str, detail: Optional[dict] = None) -> None:
        document = {
            "proposal_id": proposal_id,
            "event": event,
            "session_id": session_id,
            "detail": detail or {},
            "timestamp": datetime.now(timezone.utc),
        }
        try:
            await asyncio.to_thread(self._collection.insert_one, document)
        except PyMongoError as exc:
            raise ProposalStoreError(f"Failed to record audit event: {exc}") from exc

    async def history(self, proposal_id: str) -> list[dict[str, Any]]:
        def _run():
            return list(self._collection.find({"proposal_id": proposal_id}).sort("timestamp", 1))

        try:
            return await asyncio.to_thread(_run)
        except PyMongoError as exc:
            raise ProposalStoreError(f"Failed to fetch audit history: {exc}") from exc
