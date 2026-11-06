"""
API-level tests for Milestone 7's chat-persistence integration
(/api/v1/chat with thread_id).

All Mongo-backed services are monkeypatched at the chat-route-module level;
Gemini/Pinecone are replaced exactly as in tests/test_api_contract.py. No
real network calls are made.
"""

import app.api.routes.chat as chat_module
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import Settings, get_settings
from app.core.session import get_session_id
from app.main import app as fastapi_app
from app.services.conversation_store import ConversationStore
from tests.conftest import FakeCollection, FakeLongTermStore


def _settings_with_mongo() -> Settings:
    real = get_settings()
    return Settings(**{**real.model_dump(), "mongodb_uri": "mongodb://fake"})


class _DummyService:
    def __init__(self, *args, **kwargs):
        pass


class _EchoingChatModel:
    """Replies with a message that proves whether memory context reached it."""

    def __init__(self):
        self.last_messages = None

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.last_messages = messages
        return AIMessage(content="This repo uses the context you gave me.", tool_calls=[])


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    fastapi_app.dependency_overrides.clear()


def _patch_gemini_pinecone(monkeypatch, fake_search_service, chat_model):
    monkeypatch.setattr(chat_module, "EmbeddingService", _DummyService)
    monkeypatch.setattr(chat_module, "PineconeService", _DummyService)
    monkeypatch.setattr(chat_module, "RepositorySearchService", lambda **kwargs: fake_search_service)
    monkeypatch.setattr(chat_module, "get_chat_model", lambda **kwargs: chat_model)


def _patch_mongo(monkeypatch):
    """Configures settings.mongodb_uri (via dependency override) and fakes every Mongo-backed factory."""
    fastapi_app.dependency_overrides[get_settings] = _settings_with_mongo

    collection = FakeCollection()
    checkpointer = InMemorySaver()
    memory_store = FakeLongTermStore()
    from app.services.memory_store import LongTermMemoryService

    monkeypatch.setattr(chat_module, "get_conversation_store", lambda settings: ConversationStore(collection))
    monkeypatch.setattr(chat_module, "get_checkpointer", lambda settings: checkpointer)
    monkeypatch.setattr(chat_module, "get_memory_service", lambda settings: LongTermMemoryService(memory_store))
    return collection, checkpointer, memory_store


def test_chat_without_thread_id_is_still_fully_ephemeral(monkeypatch, sample_store, fake_search_service):
    """No thread_id -> zero Mongo calls at all, even if Mongo isn't patched/configured."""
    monkeypatch.setattr(chat_module, "repository_store", sample_store)
    _patch_gemini_pinecone(monkeypatch, fake_search_service, _EchoingChatModel())
    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"

    client = TestClient(fastapi_app)
    response = client.post("/api/v1/chat", json={"question": "Hello"})

    assert response.status_code == 200
    assert response.json()["thread_id"] is None


def test_chat_with_unknown_thread_id_returns_404(monkeypatch, sample_store, fake_search_service):
    monkeypatch.setattr(chat_module, "repository_store", sample_store)
    _patch_gemini_pinecone(monkeypatch, fake_search_service, _EchoingChatModel())
    _patch_mongo(monkeypatch)
    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"

    client = TestClient(fastapi_app)
    response = client.post("/api/v1/chat", json={"question": "Hi", "thread_id": "does-not-exist"})

    assert response.status_code == 404


def test_chat_rejects_a_thread_owned_by_another_session(monkeypatch, sample_store, fake_search_service):
    monkeypatch.setattr(chat_module, "repository_store", sample_store)
    _patch_gemini_pinecone(monkeypatch, fake_search_service, _EchoingChatModel())
    collection, _, _ = _patch_mongo(monkeypatch)

    conversation_store = ConversationStore(collection)
    import asyncio

    asyncio.run(
        conversation_store.create(
            thread_id="thread-1", session_id="session-owner", repository_id="acme/webapp", workflow="chat"
        )
    )

    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-intruder"
    client = TestClient(fastapi_app)
    response = client.post("/api/v1/chat", json={"question": "Hi", "thread_id": "thread-1"})

    assert response.status_code == 404  # not 403 -- do not confirm existence


def test_chat_rejects_a_thread_belonging_to_a_different_workflow(monkeypatch, sample_store, fake_search_service):
    monkeypatch.setattr(chat_module, "repository_store", sample_store)
    _patch_gemini_pinecone(monkeypatch, fake_search_service, _EchoingChatModel())
    collection, _, _ = _patch_mongo(monkeypatch)

    conversation_store = ConversationStore(collection)
    import asyncio

    asyncio.run(
        conversation_store.create(
            thread_id="thread-1", session_id="session-a", repository_id="acme/webapp", workflow="investigation"
        )
    )

    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)
    response = client.post("/api/v1/chat", json={"question": "Hi", "thread_id": "thread-1"})

    assert response.status_code == 409


def test_chat_rejects_a_thread_scoped_to_a_different_repository(monkeypatch, sample_store, fake_search_service):
    monkeypatch.setattr(chat_module, "repository_store", sample_store)
    _patch_gemini_pinecone(monkeypatch, fake_search_service, _EchoingChatModel())
    collection, _, _ = _patch_mongo(monkeypatch)

    conversation_store = ConversationStore(collection)
    import asyncio

    asyncio.run(
        conversation_store.create(
            thread_id="thread-1", session_id="session-a", repository_id="some/other-repo", workflow="chat"
        )
    )

    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)
    response = client.post("/api/v1/chat", json={"question": "Hi", "thread_id": "thread-1"})

    assert response.status_code == 409


def test_chat_with_valid_thread_id_persists_and_restores_across_turns(monkeypatch, sample_store, fake_search_service):
    monkeypatch.setattr(chat_module, "repository_store", sample_store)
    chat_model = _EchoingChatModel()
    _patch_gemini_pinecone(monkeypatch, fake_search_service, chat_model)
    collection, checkpointer, _ = _patch_mongo(monkeypatch)

    conversation_store = ConversationStore(collection)
    import asyncio

    asyncio.run(
        conversation_store.create(
            thread_id="thread-1", session_id="session-a", repository_id="acme/webapp", workflow="chat"
        )
    )

    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)

    first = client.post("/api/v1/chat", json={"question": "First question", "thread_id": "thread-1"})
    assert first.status_code == 200
    assert first.json()["thread_id"] == "thread-1"

    second = client.post("/api/v1/chat", json={"question": "Second question", "thread_id": "thread-1"})
    assert second.status_code == 200

    # The second call's model invocation must have seen the first turn's
    # HumanMessage too -- proof the checkpointer actually restored history.
    from langchain_core.messages import HumanMessage

    human_texts = [m.content for m in chat_model.last_messages if isinstance(m, HumanMessage)]
    assert human_texts == ["First question", "Second question"]


def test_chat_injects_long_term_memory_as_context(monkeypatch, sample_store, fake_search_service):
    monkeypatch.setattr(chat_module, "repository_store", sample_store)
    chat_model = _EchoingChatModel()
    _patch_gemini_pinecone(monkeypatch, fake_search_service, chat_model)
    _, _, memory_store = _patch_mongo(monkeypatch)

    from app.services.memory_store import LongTermMemoryService

    import asyncio

    memory_service = LongTermMemoryService(memory_store)
    asyncio.run(
        memory_service.remember(
            session_id="session-a",
            repository_id="acme/webapp",
            content="This app uses Flask and a Postgres database.",
            kind="confirmed_finding",
            source="investigation:earlier-thread",
        )
    )

    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)
    response = client.post("/api/v1/chat", json={"question": "What stack does this use?"})

    assert response.status_code == 200
    from langchain_core.messages import SystemMessage

    system_message = next(m for m in chat_model.last_messages if isinstance(m, SystemMessage))
    assert "Flask and a Postgres database" in system_message.content


def test_chat_endpoint_requires_mongo_configured_for_thread_id(monkeypatch, sample_store, fake_search_service):
    """thread_id supplied but MONGODB_URI unset (real settings) -> clear 503, not a crash."""
    monkeypatch.setattr(chat_module, "repository_store", sample_store)
    _patch_gemini_pinecone(monkeypatch, fake_search_service, _EchoingChatModel())
    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"

    client = TestClient(fastapi_app)
    response = client.post("/api/v1/chat", json={"question": "Hi", "thread_id": "whatever"})

    assert response.status_code == 503
