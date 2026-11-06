"""
Bug-investigation endpoint (Milestone 6; Milestone 7 adds optional persistence).

A dedicated endpoint rather than a "mode" flag on /api/v1/chat: the request
(a bug report) and response (a structured, multi-field investigation report)
shapes are different enough from plain chat Q&A that folding them into
ChatRequest/ChatResponse would either bloat that contract or require a large
optional/union shape most chat callers never use. This keeps the Milestone 5
chat endpoint's contract completely untouched.

Milestone 7 design decision — investigation threads never resume mid-workflow:
this graph runs start-to-finish in a single `ainvoke` call (there is no valid
"paused mid-investigation" state in this design). So reusing a thread_id here
means "start a fresh investigation, keep grouping it under this conversation
record for history," not "continue where it left off." Before invoking, any
prior checkpoint for that thread_id is explicitly cleared
(`checkpointer.adelete_thread`), guaranteeing a new bug report never inherits
stale evidence, hypotheses, or a stale root-cause analysis from a previous
run on the same thread.
"""

from langchain_core.messages import HumanMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError, GoogleAPIError, GoogleRateLimitError
from langgraph.errors import GraphRecursionError

from fastapi import APIRouter, Depends, HTTPException

from app.agents.investigation_graph import build_investigation_graph
from app.agents.investigation_state import InvestigationState
from app.agents.model import ChatModelError, get_chat_model
from app.agents.tools import build_tools
from app.core.config import Settings, get_settings
from app.core.mongodb import MongoDBConnectionError
from app.core.persistence import get_checkpointer, get_conversation_store, get_memory_service
from app.core.session import get_session_id
from app.ingestion.store import repository_store
from app.retrieval.search_service import RepositorySearchService
from app.schemas.investigation import InvestigateRequest, InvestigationResponse
from app.services.conversation_store import ConversationStoreError
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError
from app.services.memory_store import MemoryStoreError
from app.services.pinecone_service import PineconeService, PineconeServiceError

router = APIRouter(prefix="/api/v1", tags=["investigation"])


@router.post("/investigate", response_model=InvestigationResponse)
async def investigate(
    payload: InvestigateRequest,
    settings: Settings = Depends(get_settings),
    session_id: str = Depends(get_session_id),
) -> InvestigationResponse:
    repository = repository_store.get_current()
    if repository is None:
        raise HTTPException(
            status_code=409,
            detail="No repository has been ingested yet. Call POST /api/v1/repositories/ingest first.",
        )
    repository_id = f"{repository.owner}/{repository.repo}"

    try:
        embedding_service = EmbeddingService(
            api_key=settings.google_api_key,
            model=settings.google_embedding_model,
        )
        pinecone_service = PineconeService(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
        chat_model = get_chat_model(
            api_key=settings.google_api_key,
            model=settings.google_chat_model,
            temperature=settings.google_temperature,
        )
    except (EmbeddingServiceError, PineconeServiceError, ChatModelError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    checkpointer = None
    if payload.thread_id is not None:
        checkpointer = await _prepare_thread_for_investigation(
            payload.thread_id, repository_id, settings, session_id
        )

    search_service = RepositorySearchService(
        store=repository_store,
        embedding_service=embedding_service,
        pinecone_service=pinecone_service,
    )
    tools = build_tools(search_service=search_service, store=repository_store)
    graph = build_investigation_graph(
        chat_model=chat_model, tools=tools, store=repository_store, checkpointer=checkpointer
    )

    initial_state: InvestigationState = {
        "repository_id": repository_id,
        "bug_report": payload.bug_report,
        "max_investigation_rounds": settings.investigation_max_rounds,
        "investigation_round": 0,
        "bug_understanding": None,
        "investigation_plan": None,
        "messages": [],
        "evidence": [],
        "evidence_assessment": None,
        "root_cause_analysis": None,
        "final_report": None,
        "status": "understanding",
    }
    config = {"recursion_limit": settings.investigation_recursion_limit}
    if payload.thread_id is not None:
        config["configurable"] = {"thread_id": payload.thread_id}

    try:
        final_state: InvestigationState = await graph.ainvoke(initial_state, config=config)
    except GraphRecursionError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "The investigation did not converge within "
                f"{settings.investigation_recursion_limit} steps. Try a more specific bug report."
            ),
        ) from exc
    except GoogleRateLimitError as exc:
        raise HTTPException(status_code=429, detail=f"Gemini rate limit exceeded: {exc}") from exc
    except (ChatGoogleGenerativeAIError, GoogleAPIError) as exc:
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {exc}") from exc

    report = final_state.get("final_report")
    if report is None:
        raise HTTPException(status_code=502, detail="Investigation did not produce a report.")
    report.thread_id = payload.thread_id

    if payload.thread_id is not None:
        try:
            conversation_store = get_conversation_store(settings)
            await conversation_store.touch(payload.thread_id, status=report.status)
        except (MongoDBConnectionError, ConversationStoreError):
            pass  # best-effort; must not fail an otherwise-successful investigation

    await _remember_confirmed_findings(settings, session_id, repository_id, payload.thread_id, report)

    return report


async def _prepare_thread_for_investigation(thread_id: str, repository_id: str, settings: Settings, session_id: str):
    """Ownership + workflow checks, then clears any prior checkpoint and returns a ready checkpointer."""
    if not settings.mongodb_uri:
        raise HTTPException(
            status_code=503,
            detail="MongoDB is not configured; thread_id cannot be used. Set MONGODB_URI in .env.",
        )
    try:
        conversation_store = get_conversation_store(settings)
        conversation = await conversation_store.get(thread_id)
    except (MongoDBConnectionError, ConversationStoreError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if conversation is None or conversation["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    if conversation["workflow"] != "investigation":
        raise HTTPException(
            status_code=409, detail="This thread_id belongs to a chat conversation, not an investigation."
        )
    if conversation["repository_id"] != repository_id:
        raise HTTPException(
            status_code=409,
            detail=(
                f"This investigation thread is scoped to repository '{conversation['repository_id']}', but "
                f"the currently ingested repository is '{repository_id}'."
            ),
        )

    try:
        checkpointer = get_checkpointer(settings)
        # Always start fresh — see module docstring for why investigation
        # threads never resume mid-workflow.
        await checkpointer.adelete_thread(thread_id)
    except MongoDBConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return checkpointer


async def _remember_confirmed_findings(
    settings: Settings,
    session_id: str,
    repository_id: str,
    thread_id: str | None,
    report: InvestigationResponse,
) -> None:
    """
    Deterministic, zero-extra-LLM-call memory write: only `confirmed_findings`
    (never `hypotheses`) from a *complete* investigation become long-term
    memories — reusing output the graph already produced and verified,
    rather than making a separate judgment call about what's "useful."
    Best-effort: a Mongo hiccup here must not fail an otherwise-successful
    investigation response.
    """
    if not settings.mongodb_uri or report.status != "complete" or not report.confirmed_findings:
        return
    try:
        memory_service = get_memory_service(settings)
        source = f"investigation:{thread_id}" if thread_id else "investigation:ephemeral"
        for finding in report.confirmed_findings:
            await memory_service.remember(
                session_id=session_id,
                repository_id=repository_id,
                content=finding,
                kind="confirmed_finding",
                source=source,
            )
    except (MongoDBConnectionError, MemoryStoreError):
        pass
