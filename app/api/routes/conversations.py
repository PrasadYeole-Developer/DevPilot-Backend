"""
Conversation management endpoints (Milestone 7).

These endpoints never expose raw LangGraph checkpoint documents — messages
are read via the checkpointer's own `aget_tuple` and mapped into the plain
`ConversationMessage` schema before being returned.

Ownership: every call here is scoped to the caller's session_id (see
app.core.session — an explicit, documented pre-auth placeholder). A
conversation belonging to a different session_id is treated as not found
(404), not forbidden (403) — this avoids confirming to a guesser that a
thread_id exists at all.

Deletion policy (see DELETE below): removes the conversation's checkpoint and
metadata. Long-term memories derived from that conversation are deliberately
NOT deleted — they represent durable repository knowledge that outlives any
one conversation; each memory keeps its originating thread_id in `source`
for audit purposes even after the conversation itself is gone.
"""

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from app.core.config import Settings, get_settings
from app.core.mongodb import MongoDBConnectionError
from app.core.persistence import get_checkpointer, get_conversation_store
from app.core.session import get_session_id
from app.ingestion.store import repository_store
from app.schemas.conversation import (
    ConversationCreateRequest,
    ConversationDetail,
    ConversationListResponse,
    ConversationMessage,
    ConversationSummary,
    ConversationWorkflow,
)
from app.services.conversation_store import ConversationStore, ConversationStoreError

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


def _require_conversation_store(settings: Settings) -> ConversationStore:
    if not settings.mongodb_uri:
        raise HTTPException(
            status_code=503,
            detail="MongoDB is not configured. Set MONGODB_URI in .env to use conversation persistence.",
        )
    try:
        return get_conversation_store(settings)
    except MongoDBConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _check_ownership(document: Optional[dict], session_id: str) -> dict:
    if document is None or document["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return document


def _to_summary(document: dict) -> ConversationSummary:
    return ConversationSummary(
        thread_id=document["_id"],
        repository_id=document["repository_id"],
        workflow=document["workflow"],
        title=document.get("title"),
        status=document.get("status", "active"),
        created_at=document["created_at"],
        updated_at=document["updated_at"],
    )


def _serialize_messages(messages: list) -> list[ConversationMessage]:
    role_by_type = {"human": "human", "ai": "ai", "tool": "tool", "system": "system"}
    serialized = []
    for message in messages:
        role = role_by_type.get(getattr(message, "type", None), "ai")
        text = message.text if hasattr(message, "text") else str(message.content)
        serialized.append(
            ConversationMessage(
                role=role,
                content=text,
                tool_name=getattr(message, "name", None) if role == "tool" else None,
            )
        )
    return serialized


@router.post("", response_model=ConversationSummary, status_code=201)
async def create_conversation(
    payload: ConversationCreateRequest,
    settings: Settings = Depends(get_settings),
    session_id: str = Depends(get_session_id),
) -> ConversationSummary:
    conversation_store = _require_conversation_store(settings)

    repository_id = payload.repository_id
    if repository_id is None:
        repository = repository_store.get_current()
        if repository is None:
            raise HTTPException(
                status_code=409,
                detail="No repository has been ingested yet, and no repository_id was provided.",
            )
        repository_id = f"{repository.owner}/{repository.repo}"

    thread_id = uuid.uuid4().hex
    try:
        document = await conversation_store.create(
            thread_id=thread_id,
            session_id=session_id,
            repository_id=repository_id,
            workflow=payload.workflow,
            title=payload.title,
            metadata=payload.metadata,
        )
    except ConversationStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return _to_summary(document)


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    repository_id: Optional[str] = None,
    workflow: Optional[ConversationWorkflow] = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    settings: Settings = Depends(get_settings),
    session_id: str = Depends(get_session_id),
) -> ConversationListResponse:
    conversation_store = _require_conversation_store(settings)
    bounded_limit = min(limit, settings.conversation_list_max_limit)

    try:
        documents, total = await conversation_store.list_for_session(
            session_id=session_id,
            repository_id=repository_id,
            workflow=workflow,
            limit=bounded_limit,
            offset=offset,
        )
    except ConversationStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ConversationListResponse(
        conversations=[_to_summary(doc) for doc in documents],
        total=total,
        limit=bounded_limit,
        offset=offset,
    )


@router.get("/{thread_id}", response_model=ConversationDetail)
async def get_conversation(
    thread_id: str,
    settings: Settings = Depends(get_settings),
    session_id: str = Depends(get_session_id),
) -> ConversationDetail:
    conversation_store = _require_conversation_store(settings)

    try:
        document = await conversation_store.get(thread_id)
    except ConversationStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    document = _check_ownership(document, session_id)

    try:
        checkpointer = get_checkpointer(settings)
        checkpoint_tuple = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    except MongoDBConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    messages: list = []
    if checkpoint_tuple is not None:
        channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])

    summary = _to_summary(document)
    return ConversationDetail(**summary.model_dump(), messages=_serialize_messages(messages))


@router.delete("/{thread_id}", status_code=204)
async def delete_conversation(
    thread_id: str,
    settings: Settings = Depends(get_settings),
    session_id: str = Depends(get_session_id),
) -> Response:
    conversation_store = _require_conversation_store(settings)

    try:
        document = await conversation_store.get(thread_id)
    except ConversationStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    _check_ownership(document, session_id)

    try:
        checkpointer = get_checkpointer(settings)
        await checkpointer.adelete_thread(thread_id)
        await conversation_store.delete(thread_id)
    except (MongoDBConnectionError, ConversationStoreError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(status_code=204)
