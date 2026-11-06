"""
API-level tests for Milestone 7's investigation-persistence integration
(/api/v1/investigate with thread_id) and automatic long-term memory writes.
"""

import asyncio

import app.api.routes.investigate as investigate_module
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import Settings, get_settings
from app.core.session import get_session_id
from app.main import app as fastapi_app
from app.schemas.investigation import BugUnderstanding, EvidenceAssessment, InvestigationPlan, RootCauseAnalysis
from app.services.conversation_store import ConversationStore
from app.services.memory_store import LongTermMemoryService
from tests.conftest import FakeCollection, FakeLongTermStore, ScriptedChatModel

BUG_REPORT = "Deactivated staff members can still access the application after their accounts are disabled."


class _DummyService:
    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    fastapi_app.dependency_overrides.clear()


def _settings_with_mongo() -> Settings:
    real = get_settings()
    return Settings(**{**real.model_dump(), "mongodb_uri": "mongodb://fake"})


def _patch_gemini_pinecone(monkeypatch, fake_search_service, chat_model):
    monkeypatch.setattr(investigate_module, "EmbeddingService", _DummyService)
    monkeypatch.setattr(investigate_module, "PineconeService", _DummyService)
    monkeypatch.setattr(investigate_module, "RepositorySearchService", lambda **kwargs: fake_search_service)
    monkeypatch.setattr(investigate_module, "get_chat_model", lambda **kwargs: chat_model)


def _patch_mongo(monkeypatch):
    fastapi_app.dependency_overrides[get_settings] = _settings_with_mongo
    collection = FakeCollection()
    checkpointer = InMemorySaver()
    memory_store = FakeLongTermStore()
    monkeypatch.setattr(investigate_module, "get_conversation_store", lambda settings: ConversationStore(collection))
    monkeypatch.setattr(investigate_module, "get_checkpointer", lambda settings: checkpointer)
    monkeypatch.setattr(
        investigate_module, "get_memory_service", lambda settings: LongTermMemoryService(memory_store)
    )
    return collection, checkpointer, memory_store


def _successful_investigation_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        structured_responses={
            BugUnderstanding: BugUnderstanding(
                expected_behavior="e", actual_behavior="a", keywords=["session"], investigation_areas=["auth"]
            ),
            InvestigationPlan: InvestigationPlan(summary="s", steps=[]),
            EvidenceAssessment: EvidenceAssessment(
                sufficient=True, verified_findings=[], contradictions=[], missing_evidence=[], reasoning="r"
            ),
            RootCauseAnalysis: RootCauseAnalysis(
                root_cause="Sessions are not revoked on deactivation.",
                explanation="deactivate_user never invalidates the session.",
                confirmed_findings=["deactivate_user() never revokes sessions (app/auth/deactivate.py)"],
                hypotheses=[],
                confidence_level="high",
                recommended_fix_direction="Revoke sessions in deactivate_user.",
                remaining_unknowns=[],
            ),
        },
        tool_responses=[
            AIMessage(content="", tool_calls=[{"name": "search_codebase", "args": {"query": "session"}, "id": "c1"}]),
            AIMessage(content="Confirmed deactivate_user never revokes sessions.", tool_calls=[]),
        ],
    )


def test_investigate_without_thread_id_still_writes_long_term_memory(monkeypatch, sample_store, fake_search_service):
    """Ephemeral (no thread_id) investigations still contribute to cross-conversation memory."""
    monkeypatch.setattr(investigate_module, "repository_store", sample_store)
    _patch_gemini_pinecone(monkeypatch, fake_search_service, _successful_investigation_model())
    _, _, memory_store = _patch_mongo(monkeypatch)

    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)
    response = client.post("/api/v1/investigate", json={"bug_report": BUG_REPORT})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "complete"
    assert body["thread_id"] is None

    memories = asyncio.run(
        LongTermMemoryService(memory_store).recall(session_id="session-a", repository_id="acme/webapp")
    )
    assert len(memories) == 1
    assert "deactivate_user" in memories[0]["content"]
    assert memories[0]["source"] == "investigation:ephemeral"


def test_investigate_with_thread_id_tags_memory_source_with_thread(monkeypatch, sample_store, fake_search_service):
    monkeypatch.setattr(investigate_module, "repository_store", sample_store)
    _patch_gemini_pinecone(monkeypatch, fake_search_service, _successful_investigation_model())
    collection, _, memory_store = _patch_mongo(monkeypatch)

    conversation_store = ConversationStore(collection)
    asyncio.run(
        conversation_store.create(
            thread_id="inv-1", session_id="session-a", repository_id="acme/webapp", workflow="investigation"
        )
    )

    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)
    response = client.post("/api/v1/investigate", json={"bug_report": BUG_REPORT, "thread_id": "inv-1"})

    assert response.status_code == 200
    assert response.json()["thread_id"] == "inv-1"

    memories = asyncio.run(
        LongTermMemoryService(memory_store).recall(session_id="session-a", repository_id="acme/webapp")
    )
    assert memories[0]["source"] == "investigation:inv-1"

    # The conversation's status was updated to match the investigation outcome.
    updated = asyncio.run(conversation_store.get("inv-1"))
    assert updated["status"] == "complete"


def test_reusing_an_investigation_thread_clears_prior_checkpoint_state(monkeypatch, sample_store, fake_search_service):
    """
    Starting a second investigation on the same thread_id must never let the
    first run's evidence/hypotheses leak into the second run's result —
    regardless of what the (possibly different) bug report says this time.
    """
    monkeypatch.setattr(investigate_module, "repository_store", sample_store)
    collection, checkpointer, _ = _patch_mongo(monkeypatch)

    conversation_store = ConversationStore(collection)
    asyncio.run(
        conversation_store.create(
            thread_id="inv-1", session_id="session-a", repository_id="acme/webapp", workflow="investigation"
        )
    )

    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)

    # First run.
    first_model = _successful_investigation_model()
    monkeypatch.setattr(investigate_module, "EmbeddingService", _DummyService)
    monkeypatch.setattr(investigate_module, "PineconeService", _DummyService)
    monkeypatch.setattr(investigate_module, "RepositorySearchService", lambda **kwargs: fake_search_service)
    monkeypatch.setattr(investigate_module, "get_chat_model", lambda **kwargs: first_model)
    first = client.post("/api/v1/investigate", json={"bug_report": BUG_REPORT, "thread_id": "inv-1"})
    assert first.status_code == 200
    assert first.json()["confirmed_findings"] == [
        "deactivate_user() never revokes sessions (app/auth/deactivate.py)"
    ]

    # The checkpoint for inv-1 must exist after the first run.
    state_after_first = asyncio.run(
        checkpointer.aget_tuple({"configurable": {"thread_id": "inv-1", "checkpoint_ns": ""}})
    )
    assert state_after_first is not None

    # Second run, same thread_id, a different (unrelated, inconclusive) bug report.
    second_model = ScriptedChatModel(
        structured_responses={
            BugUnderstanding: BugUnderstanding(
                expected_behavior="e2", actual_behavior="a2", keywords=[], investigation_areas=[]
            ),
            InvestigationPlan: InvestigationPlan(summary="s2", steps=[]),
            EvidenceAssessment: EvidenceAssessment(
                sufficient=False, verified_findings=[], contradictions=[], missing_evidence=["nothing found"], reasoning="r2"
            ),
        },
        # Always-insufficient EvidenceAssessment means every round loops back
        # via continue_investigation -> investigate_agent until rounds are
        # exhausted — enough identical responses for investigation_max_rounds + 1.
        tool_responses=[AIMessage(content="Found nothing relevant.", tool_calls=[]) for _ in range(6)],
    )
    monkeypatch.setattr(investigate_module, "get_chat_model", lambda **kwargs: second_model)
    second = client.post(
        "/api/v1/investigate",
        json={"bug_report": "Completely unrelated report.", "thread_id": "inv-1"},
    )

    assert second.status_code == 200
    body = second.json()
    assert body["status"] == "inconclusive"
    # The first run's confirmed finding must NOT leak into the second run's report.
    assert body["confirmed_findings"] == []
    assert "deactivate_user" not in str(body)


def test_investigate_rejects_a_thread_belonging_to_a_different_workflow(monkeypatch, sample_store, fake_search_service):
    monkeypatch.setattr(investigate_module, "repository_store", sample_store)
    _patch_gemini_pinecone(monkeypatch, fake_search_service, _successful_investigation_model())
    collection, _, _ = _patch_mongo(monkeypatch)

    conversation_store = ConversationStore(collection)
    asyncio.run(
        conversation_store.create(
            thread_id="thread-1", session_id="session-a", repository_id="acme/webapp", workflow="chat"
        )
    )

    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)
    response = client.post("/api/v1/investigate", json={"bug_report": BUG_REPORT, "thread_id": "thread-1"})

    assert response.status_code == 409


def test_investigate_rejects_a_thread_owned_by_another_session(monkeypatch, sample_store, fake_search_service):
    monkeypatch.setattr(investigate_module, "repository_store", sample_store)
    _patch_gemini_pinecone(monkeypatch, fake_search_service, _successful_investigation_model())
    collection, _, _ = _patch_mongo(monkeypatch)

    conversation_store = ConversationStore(collection)
    asyncio.run(
        conversation_store.create(
            thread_id="inv-1", session_id="session-owner", repository_id="acme/webapp", workflow="investigation"
        )
    )

    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-intruder"
    client = TestClient(fastapi_app)
    response = client.post("/api/v1/investigate", json={"bug_report": BUG_REPORT, "thread_id": "inv-1"})

    assert response.status_code == 404
