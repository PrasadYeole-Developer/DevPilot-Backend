"""
Chat endpoint: answers questions about the currently ingested repository via
the tool-using agent graph.

Milestone 7 addition: an optional `thread_id` enables persisted, multi-turn
conversations backed by MongoDB. Omitting it (or leaving MONGODB_URI unset)
reproduces the exact Milestone 5 ephemeral behavior — no DB reads or writes
happen at all in that path.
"""

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
from app.core.mongodb import MongoDBConnectionError
from app.core.persistence import get_checkpointer, get_conversation_store, get_memory_service
from app.core.session import get_session_id
from app.ingestion.store import repository_store
from app.retrieval.search_service import RepositorySearchService
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource
from app.services.conversation_store import ConversationStoreError
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError
from app.services.pinecone_service import PineconeService, PineconeServiceError

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    settings: Settings = Depends(get_settings),
    session_id: str = Depends(get_session_id),
) -> ChatResponse:
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
        checkpointer = await _validate_thread_for_chat(payload.thread_id, repository_id, settings, session_id)

    extra_context = await _recall_memories_as_context(settings, session_id, repository_id)

    search_service = RepositorySearchService(
        store=repository_store,
        embedding_service=embedding_service,
        pinecone_service=pinecone_service,
    )
    tools = build_tools(search_service=search_service, store=repository_store)
    graph = build_agent_graph(
        chat_model=chat_model, tools=tools, checkpointer=checkpointer, extra_context=extra_context
    )

    initial_state: AgentState = {
        "messages": [HumanMessage(content=payload.question)],
        "repository_id": repository_id,
    }
    config = {"recursion_limit": settings.agent_recursion_limit}
    if payload.thread_id is not None:
        config["configurable"] = {"thread_id": payload.thread_id}

    try:
        final_state: AgentState = await graph.ainvoke(initial_state, config=config)
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

    if payload.thread_id is not None:
        try:
            conversation_store = get_conversation_store(settings)
            await conversation_store.touch(payload.thread_id)
        except (MongoDBConnectionError, ConversationStoreError):
            pass  # best-effort timestamp refresh; must not fail the actual chat response

    return ChatResponse(
        answer=answer,
        sources=[ChatSource(**source) for source in collect_search_sources(messages)],
        tool_calls=collect_tool_calls(messages),
        thread_id=payload.thread_id,
    )


async def _validate_thread_for_chat(thread_id: str, repository_id: str, settings: Settings, session_id: str):
    """Ownership + workflow + repository-consistency checks, then returns a ready checkpointer."""
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
    if conversation["workflow"] != "chat":
        raise HTTPException(
            status_code=409, detail="This thread_id belongs to an investigation, not a chat conversation."
        )
    if conversation["repository_id"] != repository_id:
        raise HTTPException(
            status_code=409,
            detail=(
                f"This conversation is scoped to repository '{conversation['repository_id']}', but the "
                f"currently ingested repository is '{repository_id}'. Re-ingest the original repository "
                "to continue this conversation."
            ),
        )

    try:
        return get_checkpointer(settings)
    except MongoDBConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def _recall_memories_as_context(settings: Settings, session_id: str, repository_id: str) -> str | None:
    """Best-effort: a Mongo hiccup here must never break an otherwise-working chat request."""
    if not settings.mongodb_uri:
        return None
    try:
        memory_service = get_memory_service(settings)
        memories = await memory_service.recall(session_id=session_id, repository_id=repository_id)
    except MongoDBConnectionError:
        return None

    if not memories:
        return None
    lines = "\n".join(f"- [{m['kind']}] {m['content']}" for m in memories)
    return (
        "Known context about this repository, remembered from previous conversations/investigations "
        "(this is prior context, not something you have verified yourself in this conversation — "
        "re-check against the actual code with your tools before relying on it, and say so if you do):\n"
        f"{lines}"
    )
