"""
API-level tests confirming:
- The Milestone 5 /api/v1/chat response contract is unchanged (plus the
  additive tool_calls field).
- The new /api/v1/investigate endpoint behaves sanely (auth/precondition
  errors) without needing real Gemini/Pinecone credentials.

Dependencies are monkeypatched at the route-module level (not via FastAPI's
dependency_overrides, since these services are constructed directly inside
the route functions) — no real network calls are made.
"""

import app.api.routes.chat as chat_module
import app.api.routes.investigate as investigate_module
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.ingestion.store import RepositoryStore
from app.main import app


class _DummyService:
    def __init__(self, *args, **kwargs):
        pass


class _FakeChatModelForChat:
    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(
            content="Sessions are validated by checking the token (see app/auth/session.py).",
            tool_calls=[],
        )


def test_chat_endpoint_response_shape(monkeypatch, sample_store, fake_search_service):
    monkeypatch.setattr(chat_module, "repository_store", sample_store)
    monkeypatch.setattr(chat_module, "EmbeddingService", _DummyService)
    monkeypatch.setattr(chat_module, "PineconeService", _DummyService)
    monkeypatch.setattr(chat_module, "RepositorySearchService", lambda **kwargs: fake_search_service)
    monkeypatch.setattr(chat_module, "get_chat_model", lambda **kwargs: _FakeChatModelForChat())

    client = TestClient(app)
    response = client.post("/api/v1/chat", json={"question": "How are sessions validated?"})

    assert response.status_code == 200
    body = response.json()
    # Milestone 5 contract (answer, sources) unchanged; tool_calls is additive.
    assert set(body.keys()) == {"answer", "sources", "tool_calls"}
    assert isinstance(body["answer"], str) and body["answer"]
    assert isinstance(body["sources"], list)
    assert isinstance(body["tool_calls"], list)


def test_chat_endpoint_requires_ingested_repository(monkeypatch):
    monkeypatch.setattr(chat_module, "repository_store", RepositoryStore())

    client = TestClient(app)
    response = client.post("/api/v1/chat", json={"question": "anything"})

    assert response.status_code == 409


def test_investigate_endpoint_requires_ingested_repository(monkeypatch):
    monkeypatch.setattr(investigate_module, "repository_store", RepositoryStore())

    client = TestClient(app)
    response = client.post("/api/v1/investigate", json={"bug_report": "something is broken"})

    assert response.status_code == 409


def test_investigate_endpoint_validates_empty_bug_report(monkeypatch, sample_store):
    monkeypatch.setattr(investigate_module, "repository_store", sample_store)

    client = TestClient(app)
    response = client.post("/api/v1/investigate", json={"bug_report": ""})

    assert response.status_code == 422  # Pydantic min_length=1 on InvestigateRequest
