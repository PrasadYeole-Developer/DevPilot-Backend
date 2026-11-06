"""
Bug-investigation endpoint (Milestone 6).

A dedicated endpoint rather than a "mode" flag on /api/v1/chat: the request
(a bug report) and response (a structured, multi-field investigation report)
shapes are different enough from plain chat Q&A that folding them into
ChatRequest/ChatResponse would either bloat that contract or require a large
optional/union shape most chat callers never use. This keeps the Milestone 5
chat endpoint's contract completely untouched.
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
from app.ingestion.store import repository_store
from app.retrieval.search_service import RepositorySearchService
from app.schemas.investigation import InvestigateRequest, InvestigationResponse
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError
from app.services.pinecone_service import PineconeService, PineconeServiceError

router = APIRouter(prefix="/api/v1", tags=["investigation"])


@router.post("/investigate", response_model=InvestigationResponse)
async def investigate(
    payload: InvestigateRequest,
    settings: Settings = Depends(get_settings),
) -> InvestigationResponse:
    repository = repository_store.get_current()
    if repository is None:
        raise HTTPException(
            status_code=409,
            detail="No repository has been ingested yet. Call POST /api/v1/repositories/ingest first.",
        )

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

    search_service = RepositorySearchService(
        store=repository_store,
        embedding_service=embedding_service,
        pinecone_service=pinecone_service,
    )
    tools = build_tools(search_service=search_service, store=repository_store)
    graph = build_investigation_graph(chat_model=chat_model, tools=tools, store=repository_store)

    initial_state: InvestigationState = {
        "repository_id": f"{repository.owner}/{repository.repo}",
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

    try:
        final_state: InvestigationState = await graph.ainvoke(
            initial_state,
            config={"recursion_limit": settings.investigation_recursion_limit},
        )
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

    return report
