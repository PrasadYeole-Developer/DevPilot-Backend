"""
Human-in-the-loop code-change proposal endpoints (Milestone 8).

Unlike /chat and /investigate, this workflow has no ephemeral mode: pausing
for human approval (langgraph.types.interrupt) requires a checkpointer to
persist the paused state, so MongoDB is mandatory here — every route below
returns a clean 503 up front if MONGODB_URI isn't configured, before ever
touching the graph.

Ownership/isolation follow the exact pattern established in Milestone 7's
chat/investigate routes: a proposal belongs to the session that created it
(404, not 403, on mismatch — never confirm existence to a guesser) and to
the repository that was ingested when it was created (409 if the currently
ingested repository has since changed).

Security-critical invariant enforced here, not just in the graph: a decision
can only be made once. A proposal not in PENDING_APPROVAL status is rejected
outright — the graph is never even asked to resume — which is what prevents
replaying or duplicating an approval/rejection decision.
"""

import uuid
from typing import Optional

from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError, GoogleAPIError, GoogleRateLimitError
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from fastapi import APIRouter, Depends, HTTPException

from app.agents.model import ChatModelError, get_chat_model
from app.agents.proposal_graph import build_proposal_graph
from app.agents.proposal_state import ProposalState
from app.agents.tools import build_tools
from app.core.config import Settings, get_settings
from app.core.mongodb import MongoDBConnectionError
from app.core.persistence import get_checkpointer, get_conversation_store, get_proposal_audit_store, get_proposal_store
from app.core.session import get_session_id
from app.ingestion.store import repository_store
from app.retrieval.search_service import RepositorySearchService
from app.schemas.proposal import (
    DECIDABLE_STATUSES,
    ApprovalDecisionRequest,
    CreateProposalRequest,
    ProposalDiffResponse,
    ProposalListResponse,
    ProposalResponse,
    ProposalStatus,
    ValidationResult,
)
from app.services.conversation_store import ConversationStoreError
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError
from app.services.pinecone_service import PineconeService, PineconeServiceError
from app.services.proposal_store import ProposalStoreError

router = APIRouter(prefix="/api/v1/proposals", tags=["proposals"])


def _require_mongo(settings: Settings) -> None:
    if not settings.mongodb_uri:
        raise HTTPException(
            status_code=503,
            detail=(
                "MongoDB is not configured. Proposals require persistence (pausing for "
                "human approval has no meaningful ephemeral mode) — set MONGODB_URI in .env."
            ),
        )


async def _build_graph_dependencies(settings: Settings):
    """Shared service construction for create/approve/reject — identical to chat/investigate's pattern."""
    try:
        embedding_service = EmbeddingService(api_key=settings.google_api_key, model=settings.google_embedding_model)
        pinecone_service = PineconeService(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
        chat_model = get_chat_model(
            api_key=settings.google_api_key, model=settings.google_chat_model, temperature=settings.google_temperature
        )
    except (EmbeddingServiceError, PineconeServiceError, ChatModelError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    search_service = RepositorySearchService(
        store=repository_store, embedding_service=embedding_service, pinecone_service=pinecone_service
    )
    tools = build_tools(search_service=search_service, store=repository_store)
    return chat_model, tools


def _proposal_from_result(
    proposal_id: str, thread_id: str, repository_id: str, bug_report: str, state: ProposalState
) -> ProposalResponse:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    report = state.get("investigation_report")
    draft = state.get("proposal_draft")
    changes = state.get("proposed_changes") or []
    validation = state.get("validation_result") or ValidationResult(valid=False)
    status = ProposalStatus(state.get("status", "FAILED"))

    return ProposalResponse(
        proposal_id=proposal_id,
        thread_id=thread_id,
        repository_id=repository_id,
        bug_description=bug_report,
        root_cause=report.root_cause if report else "",
        expected_behavior=draft.expected_behavior if draft else (report.expected_behavior if report else ""),
        explanation=draft.explanation if draft else "",
        assumptions=draft.assumptions if draft else [],
        proposed_changes=changes,
        affected_files=[c.file_path for c in changes] or (report.affected_files if report else []),
        unified_diff="\n".join(c.unified_diff for c in changes),
        risk_level=draft.risk_level if draft else None,
        validation_results=validation,
        status=status,
        error=state.get("error"),
        decision=state.get("decision"),
        decision_reason=state.get("decision_reason"),
        created_at=now,
        updated_at=now,
    )


def _check_ownership(document: Optional[dict], session_id: str) -> dict:
    if document is None or document.get("session_id") != session_id:
        raise HTTPException(status_code=404, detail="Proposal not found.")
    return document


@router.post("", response_model=ProposalResponse, status_code=201)
async def create_proposal(
    payload: CreateProposalRequest,
    settings: Settings = Depends(get_settings),
    session_id: str = Depends(get_session_id),
) -> ProposalResponse:
    _require_mongo(settings)

    repository = repository_store.get_current()
    if repository is None:
        raise HTTPException(
            status_code=409,
            detail="No repository has been ingested yet. Call POST /api/v1/repositories/ingest first.",
        )
    repository_id = f"{repository.owner}/{repository.repo}"

    chat_model, tools = await _build_graph_dependencies(settings)

    proposal_id = uuid.uuid4().hex  # also used as the LangGraph thread_id — see schemas/proposal.py

    try:
        conversation_store = get_conversation_store(settings)
        await conversation_store.create(
            thread_id=proposal_id,
            session_id=session_id,
            repository_id=repository_id,
            workflow="proposal",
            title=f"Proposal: {payload.bug_report[:80]}",
        )
        checkpointer = get_checkpointer(settings)
    except (MongoDBConnectionError, ConversationStoreError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    graph = build_proposal_graph(
        chat_model=chat_model,
        tools=tools,
        store=repository_store,
        checkpointer=checkpointer,
        investigation_max_rounds=settings.investigation_max_rounds,
        investigation_recursion_limit=settings.investigation_recursion_limit,
    )

    initial_state: ProposalState = {
        "proposal_id": proposal_id,
        "repository_id": repository_id,
        "bug_report": payload.bug_report,
        "investigation_report": None,
        "messages": [],
        "proposal_draft": None,
        "proposed_changes": [],
        "validation_result": None,
        "decision": None,
        "decision_reason": None,
        "status": "DRAFT",
        "error": None,
    }
    config = {
        "configurable": {"thread_id": proposal_id},
        "recursion_limit": settings.investigation_recursion_limit + 10,
    }

    try:
        result = await graph.ainvoke(initial_state, config=config)
    except GraphRecursionError as exc:
        raise HTTPException(status_code=502, detail="Proposal generation did not converge in time.") from exc
    except GoogleRateLimitError as exc:
        raise HTTPException(status_code=429, detail=f"Gemini rate limit exceeded: {exc}") from exc
    except (ChatGoogleGenerativeAIError, GoogleAPIError) as exc:
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {exc}") from exc

    proposal = _proposal_from_result(proposal_id, proposal_id, repository_id, payload.bug_report, result)

    try:
        proposal_store = get_proposal_store(settings)
        await proposal_store.save(proposal, session_id=session_id)
        audit_store = get_proposal_audit_store(settings)
        await audit_store.record(proposal_id, "created", session_id, {"bug_report": payload.bug_report})
        if proposal.status == ProposalStatus.PENDING_APPROVAL:
            await audit_store.record(proposal_id, "pending_approval", session_id, {"risk_level": proposal.risk_level})
        elif proposal.status == ProposalStatus.FAILED:
            await audit_store.record(proposal_id, "validation_failed", session_id, {"error": proposal.error})
        conversation_store = get_conversation_store(settings)
        await conversation_store.touch(proposal_id, status=proposal.status.value)
    except (MongoDBConnectionError, ProposalStoreError, ConversationStoreError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return proposal


@router.get("/{proposal_id}", response_model=ProposalResponse)
async def get_proposal(
    proposal_id: str,
    settings: Settings = Depends(get_settings),
    session_id: str = Depends(get_session_id),
) -> ProposalResponse:
    _require_mongo(settings)
    try:
        document = await get_proposal_store(settings).get(proposal_id)
    except (MongoDBConnectionError, ProposalStoreError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    document = _check_ownership(document, session_id)
    return _document_to_response(document)


@router.get("", response_model=ProposalListResponse)
async def list_proposals(
    repository_id: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    settings: Settings = Depends(get_settings),
    session_id: str = Depends(get_session_id),
) -> ProposalListResponse:
    _require_mongo(settings)
    bounded_limit = max(1, min(limit, settings.conversation_list_max_limit))
    try:
        documents, total = await get_proposal_store(settings).list_for_session(
            session_id=session_id, repository_id=repository_id, limit=bounded_limit, offset=offset
        )
    except (MongoDBConnectionError, ProposalStoreError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ProposalListResponse(
        proposals=[_document_to_response(doc) for doc in documents], total=total, limit=bounded_limit, offset=offset
    )


@router.get("/{proposal_id}/diff", response_model=ProposalDiffResponse)
async def get_proposal_diff(
    proposal_id: str,
    settings: Settings = Depends(get_settings),
    session_id: str = Depends(get_session_id),
) -> ProposalDiffResponse:
    _require_mongo(settings)
    try:
        document = await get_proposal_store(settings).get(proposal_id)
    except (MongoDBConnectionError, ProposalStoreError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    document = _check_ownership(document, session_id)
    proposal = _document_to_response(document)
    return ProposalDiffResponse(
        proposal_id=proposal.proposal_id,
        status=proposal.status,
        unified_diff=proposal.unified_diff,
        files=proposal.proposed_changes,
    )


@router.post("/{proposal_id}/approve", response_model=ProposalResponse)
async def approve_proposal(
    proposal_id: str,
    payload: ApprovalDecisionRequest,
    settings: Settings = Depends(get_settings),
    session_id: str = Depends(get_session_id),
) -> ProposalResponse:
    return await _decide(proposal_id, "approved", payload.reason, settings, session_id)


@router.post("/{proposal_id}/reject", response_model=ProposalResponse)
async def reject_proposal(
    proposal_id: str,
    payload: ApprovalDecisionRequest,
    settings: Settings = Depends(get_settings),
    session_id: str = Depends(get_session_id),
) -> ProposalResponse:
    return await _decide(proposal_id, "rejected", payload.reason, settings, session_id)


async def _decide(
    proposal_id: str, decision: str, reason: Optional[str], settings: Settings, session_id: str
) -> ProposalResponse:
    _require_mongo(settings)

    try:
        proposal_store = get_proposal_store(settings)
        document = await proposal_store.get(proposal_id)
    except (MongoDBConnectionError, ProposalStoreError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    document = _check_ownership(document, session_id)

    current_status = ProposalStatus(document["status"])
    if current_status not in DECIDABLE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Proposal is not awaiting approval (status: {current_status.value}). "
            "A decision cannot be made or replayed on it.",
        )

    repository = repository_store.get_current()
    current_repository_id = f"{repository.owner}/{repository.repo}" if repository else None
    if current_repository_id != document["repository_id"]:
        raise HTTPException(
            status_code=409,
            detail=(
                f"This proposal is scoped to repository '{document['repository_id']}', but the currently "
                f"ingested repository is '{current_repository_id}'. Approval/rejection is blocked to avoid "
                "applying a patch against the wrong repository."
            ),
        )

    chat_model, tools = await _build_graph_dependencies(settings)
    try:
        checkpointer = get_checkpointer(settings)
    except MongoDBConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    graph = build_proposal_graph(
        chat_model=chat_model,
        tools=tools,
        store=repository_store,
        checkpointer=checkpointer,
        investigation_max_rounds=settings.investigation_max_rounds,
        investigation_recursion_limit=settings.investigation_recursion_limit,
    )
    config = {
        "configurable": {"thread_id": proposal_id},
        "recursion_limit": settings.investigation_recursion_limit + 10,
    }

    try:
        result = await graph.ainvoke(Command(resume={"decision": decision, "reason": reason}), config=config)
    except GraphRecursionError as exc:
        raise HTTPException(status_code=502, detail="Applying the decision did not converge in time.") from exc

    proposal = _proposal_from_result(
        proposal_id, proposal_id, document["repository_id"], document["bug_description"], result
    )
    # Preserve the original creation timestamp rather than resetting it.
    proposal.created_at = document["created_at"]

    try:
        await proposal_store.save(proposal, session_id=session_id)
        audit_store = get_proposal_audit_store(settings)
        await audit_store.record(proposal_id, decision, session_id, {"reason": reason})
        if proposal.status == ProposalStatus.APPLIED:
            await audit_store.record(proposal_id, "applied", session_id, {})
        elif proposal.status == ProposalStatus.FAILED:
            await audit_store.record(proposal_id, "apply_failed", session_id, {"error": proposal.error})
        conversation_store = get_conversation_store(settings)
        await conversation_store.touch(proposal_id, status=proposal.status.value)
    except (MongoDBConnectionError, ProposalStoreError, ConversationStoreError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return proposal


def _document_to_response(document: dict) -> ProposalResponse:
    data = {k: v for k, v in document.items() if k not in {"_id", "session_id"}}
    return ProposalResponse(**data)
