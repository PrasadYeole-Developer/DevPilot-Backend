"""
API-level tests for /api/v1/conversations (Milestone 7).

Mongo is faked at the service-factory level (get_conversation_store /
get_checkpointer are monkeypatched to return a FakeCollection-backed
ConversationStore and an InMemorySaver), and `get_settings`/`get_session_id`
are overridden via FastAPI's dependency_overrides — no real MongoDB and no
real session header juggling needed.
"""

import app.api.routes.conversations as conversations_module
import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import Settings, get_settings
from app.core.session import get_session_id
from app.ingestion.store import RepositoryStore
from app.main import app as fastapi_app
from app.services.conversation_store import ConversationStore
from tests.conftest import FakeCollection


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    fastapi_app.dependency_overrides.clear()


def _settings_with_mongo() -> Settings:
    real = get_settings()
    return Settings(**{**real.model_dump(), "mongodb_uri": "mongodb://fake"})


def _override_session(session_id: str) -> None:
    fastapi_app.dependency_overrides[get_session_id] = lambda: session_id


def _enable_fake_mongo(monkeypatch) -> tuple[FakeCollection, InMemorySaver]:
    fastapi_app.dependency_overrides[get_settings] = _settings_with_mongo
    collection = FakeCollection()
    checkpointer = InMemorySaver()
    monkeypatch.setattr(conversations_module, "get_conversation_store", lambda settings: ConversationStore(collection))
    monkeypatch.setattr(conversations_module, "get_checkpointer", lambda settings: checkpointer)
    return collection, checkpointer


def test_conversations_endpoints_require_mongo_configured():
    """With the real (test-environment) settings, MONGODB_URI is unset -> 503, not a crash."""
    _override_session("session-a")
    client = TestClient(fastapi_app)
    response = client.post("/api/v1/conversations", json={"workflow": "chat"})
    assert response.status_code == 503


def test_create_conversation_requires_a_repository(monkeypatch):
    monkeypatch.setattr(conversations_module, "repository_store", RepositoryStore())  # empty
    _enable_fake_mongo(monkeypatch)
    _override_session("session-a")

    client = TestClient(fastapi_app)
    response = client.post("/api/v1/conversations", json={"workflow": "chat"})
    assert response.status_code == 409


def test_create_list_get_delete_conversation_lifecycle(monkeypatch, sample_store):
    monkeypatch.setattr(conversations_module, "repository_store", sample_store)
    _enable_fake_mongo(monkeypatch)
    _override_session("session-a")
    client = TestClient(fastapi_app)

    create_response = client.post("/api/v1/conversations", json={"workflow": "chat", "title": "Auth bug chat"})
    assert create_response.status_code == 201
    created = create_response.json()
    assert created["repository_id"] == "acme/webapp"  # defaulted from the currently ingested repo
    assert created["title"] == "Auth bug chat"
    thread_id = created["thread_id"]

    list_response = client.get("/api/v1/conversations")
    assert list_response.status_code == 200
    listed = list_response.json()
    assert listed["total"] == 1
    assert listed["conversations"][0]["thread_id"] == thread_id

    detail_response = client.get(f"/api/v1/conversations/{thread_id}")
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["thread_id"] == thread_id
    assert detail["messages"] == []  # nothing has run against this thread yet

    delete_response = client.delete(f"/api/v1/conversations/{thread_id}")
    assert delete_response.status_code == 204

    missing_response = client.get(f"/api/v1/conversations/{thread_id}")
    assert missing_response.status_code == 404


def test_conversation_ownership_blocks_other_sessions(monkeypatch, sample_store):
    monkeypatch.setattr(conversations_module, "repository_store", sample_store)
    _enable_fake_mongo(monkeypatch)
    _override_session("session-a")
    client = TestClient(fastapi_app)

    thread_id = client.post("/api/v1/conversations", json={"workflow": "chat"}).json()["thread_id"]

    _override_session("session-b")
    get_response = client.get(f"/api/v1/conversations/{thread_id}")
    assert get_response.status_code == 404  # not 403 -- do not confirm the thread exists

    delete_response = client.delete(f"/api/v1/conversations/{thread_id}")
    assert delete_response.status_code == 404

    list_response = client.get("/api/v1/conversations")
    assert list_response.json()["total"] == 0  # session-b sees none of session-a's conversations


def test_list_conversations_pagination(monkeypatch, sample_store):
    monkeypatch.setattr(conversations_module, "repository_store", sample_store)
    _enable_fake_mongo(monkeypatch)
    _override_session("session-a")
    client = TestClient(fastapi_app)

    for i in range(5):
        client.post("/api/v1/conversations", json={"workflow": "chat", "title": f"Conversation {i}"})

    page_1 = client.get("/api/v1/conversations", params={"limit": 2, "offset": 0}).json()
    page_2 = client.get("/api/v1/conversations", params={"limit": 2, "offset": 2}).json()

    assert page_1["total"] == 5
    assert len(page_1["conversations"]) == 2
    assert len(page_2["conversations"]) == 2
    ids_page_1 = {c["thread_id"] for c in page_1["conversations"]}
    ids_page_2 = {c["thread_id"] for c in page_2["conversations"]}
    assert ids_page_1.isdisjoint(ids_page_2)


def test_get_conversation_includes_restored_messages(monkeypatch, sample_store):
    from langchain_core.messages import AIMessage, HumanMessage

    monkeypatch.setattr(conversations_module, "repository_store", sample_store)
    collection, checkpointer = _enable_fake_mongo(monkeypatch)
    _override_session("session-a")
    client = TestClient(fastapi_app)

    thread_id = client.post("/api/v1/conversations", json={"workflow": "chat"}).json()["thread_id"]

    # Simulate a chat turn having already happened on this thread (without
    # needing a real Gemini call): write directly via the same checkpointer
    # the route will read from.
    import asyncio

    async def _seed():
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        # channel_values is only materialized for channels listed in
        # channel_versions — InMemorySaver (and real checkpointers) key off
        # that to know which channels actually have data in this checkpoint.
        versions = {"messages": "1", "repository_id": "1"}
        checkpoint = {
            "v": 1,
            "id": "seed",
            "ts": "2024-01-01T00:00:00+00:00",
            "channel_values": {
                "messages": [HumanMessage(content="Hi"), AIMessage(content="Hello!", tool_calls=[])],
                "repository_id": "acme/webapp",
            },
            "channel_versions": versions,
            "versions_seen": {},
        }
        await checkpointer.aput(config, checkpoint, {"source": "input", "step": 0, "parents": {}}, versions)

    asyncio.run(_seed())

    detail = client.get(f"/api/v1/conversations/{thread_id}").json()
    assert [m["role"] for m in detail["messages"]] == ["human", "ai"]
    assert detail["messages"][0]["content"] == "Hi"
    assert detail["messages"][1]["content"] == "Hello!"
