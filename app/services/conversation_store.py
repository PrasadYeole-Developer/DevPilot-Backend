"""
Conversation metadata CRUD (Milestone 7) — the `conversations` collection.

This is deliberately separate from the LangGraph checkpoint data (messages,
tool calls, etc., which live in the checkpointer's own collections). This
collection only tracks conversation *identity*: who owns it, which repository
it's scoped to, which workflow (chat vs. investigation) it belongs to, and
timestamps/title/status — exactly what's needed for ownership checks and the
/api/v1/conversations listing endpoints, without exposing raw checkpoint
internals.

Uses a synchronous pymongo Collection under the hood (the shared client from
app.core.mongodb), with each call offloaded via asyncio.to_thread — the same
"sync driver, async-wrapped" pattern the official checkpointer/store packages
use, so these simple CRUD calls never block the event loop.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pymongo.collection import Collection
from pymongo.errors import PyMongoError

ConversationWorkflow = Literal["chat", "investigation"]


class ConversationStoreError(Exception):
    """Raised when a conversation-metadata operation fails."""


class ConversationStore:
    def __init__(self, collection: Collection):
        self._collection = collection

    async def create(
        self,
        thread_id: str,
        session_id: str,
        repository_id: str,
        workflow: ConversationWorkflow,
        title: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        status: str = "active",
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        document = {
            "_id": thread_id,
            "session_id": session_id,
            "repository_id": repository_id,
            "workflow": workflow,
            "title": title,
            "metadata": metadata or {},
            "status": status,
            "created_at": now,
            "updated_at": now,
        }
        try:
            await asyncio.to_thread(self._collection.insert_one, document)
        except PyMongoError as exc:
            raise ConversationStoreError(f"Failed to create conversation: {exc}") from exc
        return document

    async def get(self, thread_id: str) -> Optional[dict[str, Any]]:
        try:
            return await asyncio.to_thread(self._collection.find_one, {"_id": thread_id})
        except PyMongoError as exc:
            raise ConversationStoreError(f"Failed to fetch conversation: {exc}") from exc

    async def list_for_session(
        self,
        session_id: str,
        repository_id: Optional[str] = None,
        workflow: Optional[ConversationWorkflow] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        query: dict[str, Any] = {"session_id": session_id}
        if repository_id is not None:
            query["repository_id"] = repository_id
        if workflow is not None:
            query["workflow"] = workflow

        def _run() -> tuple[list[dict[str, Any]], int]:
            total = self._collection.count_documents(query)
            cursor = (
                self._collection.find(query)
                .sort("updated_at", -1)
                .skip(offset)
                .limit(limit)
            )
            return list(cursor), total

        try:
            return await asyncio.to_thread(_run)
        except PyMongoError as exc:
            raise ConversationStoreError(f"Failed to list conversations: {exc}") from exc

    async def touch(self, thread_id: str, status: Optional[str] = None) -> None:
        update: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
        if status is not None:
            update["status"] = status
        try:
            await asyncio.to_thread(
                self._collection.update_one, {"_id": thread_id}, {"$set": update}
            )
        except PyMongoError as exc:
            raise ConversationStoreError(f"Failed to update conversation: {exc}") from exc

    async def delete(self, thread_id: str) -> bool:
        try:
            result = await asyncio.to_thread(self._collection.delete_one, {"_id": thread_id})
        except PyMongoError as exc:
            raise ConversationStoreError(f"Failed to delete conversation: {exc}") from exc
        return result.deleted_count > 0
