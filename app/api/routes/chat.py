"""Chat endpoint: answers questions about the currently ingested repository via the tool-using agent graph."""

from langchain_core.messages import HumanMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError, GoogleAPIError, GoogleRateLimitError
from langgraph.errors import GraphRecursionError

from fastapi import APIRouter, Depends, HTTPException

from app.agents.graph import build_agent_graph
from app.agents.message_utils import collect_search_sources, collect_tool_calls, final_answer
from app.agents.model import ChatModelError, get_chat_model
from app.agents.state import AgentState
from app.agents.tools import build_tools
from app.core.config import Settings, get_settings
from app.ingestion.store import repository_store
from app.retrieval.search_service import RepositorySearchService
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError
from app.services.pinecone_service import PineconeService, PineconeServiceError

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
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
    graph = build_agent_graph(chat_model=chat_model, tools=tools)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=payload.question)],
        "repository_id": f"{repository.owner}/{repository.repo}",
    }

    try:
        final_state: AgentState = await graph.ainvoke(
            initial_state,
            config={"recursion_limit": settings.agent_recursion_limit},
        )
    except GraphRecursionError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "The agent did not converge on an answer within "
                f"{settings.agent_recursion_limit} steps. Try a more specific question."
            ),
        ) from exc
    except GoogleRateLimitError as exc:
        raise HTTPException(status_code=429, detail=f"Gemini rate limit exceeded: {exc}") from exc
    except (ChatGoogleGenerativeAIError, GoogleAPIError) as exc:
        raise HTTPException(status_code=502, detail=f"Gemini request failed: {exc}") from exc

    messages = final_state["messages"]
    answer = final_answer(messages)
    if answer is None:
        raise HTTPException(status_code=502, detail="Agent did not produce a final answer.")

    return ChatResponse(
        answer=answer,
        sources=[ChatSource(**source) for source in collect_search_sources(messages)],
        tool_calls=collect_tool_calls(messages),
    )
