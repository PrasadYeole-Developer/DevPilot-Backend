"""
API-level tests for /api/v1/proposals (Milestone 8).

All Mongo-backed services are monkeypatched at the proposals-route-module
level (fake collections + a real InMemorySaver checkpointer, same pattern as
tests/test_chat_api_persistence.py); Gemini/Pinecone are replaced the same
way as elsewhere. No real network calls are made.
"""

import asyncio

import app.agents.nodes.apply_patch as apply_patch_module
import app.api.routes.proposals as proposals_module
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.core.config import Settings, get_settings
from app.core.session import get_session_id
from app.main import app as fastapi_app
from app.schemas.investigation import BugUnderstanding, EvidenceAssessment, InvestigationPlan, RootCauseAnalysis
from app.schemas.proposal import ProposalDraft, ProposedFileChangeDraft
from app.services.conversation_store import ConversationStore
from app.services.github_write_client import GitHubWriteClient
from app.services.proposal_store import ProposalAuditStore, ProposalStore
from tests.conftest import FakeCollection, FakeGitHubWriteClient, ScriptedChatModel

BUG_REPORT = "Deactivated staff members can still access the application after their account is disabled."


class _DummyService:
    def __init__(self, *args, **kwargs):
        pass


def _happy_path_responses() -> dict:
    return {
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
        ProposalDraft: ProposalDraft(
            explanation="Clear the session token on deactivation.",
            expected_behavior="Deactivated users are logged out immediately.",
            risk_level="low",
            assumptions=[],
            additional_files_needed=[],
            changes=[
                ProposedFileChangeDraft(
                    file_path="app/auth/deactivate.py",
                    change_summary="Clear session_token.",
                    proposed_content=(
                        "def deactivate_user(user):\n"
                        "    user.is_active = False\n"
                        "    user.session_token = None\n"
                        "    db.save(user)\n"
                    ),
                )
            ],
        ),
    }


def _read_call(file_path: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "read_repository_file", "args": {"file_path": file_path}, "id": "c1"}])


def _summary(text: str) -> AIMessage:
    return AIMessage(content=text, tool_calls=[])


def _happy_path_model() -> ScriptedChatModel:
    return ScriptedChatModel(
        structured_responses=_happy_path_responses(),
        tool_responses=[_read_call("app/auth/deactivate.py"), _summary("Confirmed.")],
    )


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    yield
    fastapi_app.dependency_overrides.clear()


def _settings_with_mongo(*, github_write_token: str | None = "fake-write-token") -> Settings:
    real = get_settings()
    return Settings(**{**real.model_dump(), "mongodb_uri": "mongodb://fake", "github_write_token": github_write_token})


def _patch_everything(monkeypatch, sample_store, fake_search_service, chat_model, *, write_client=None):
    fastapi_app.dependency_overrides[get_settings] = _settings_with_mongo

    conversations = FakeCollection()
    proposals = FakeCollection()
    audit = FakeCollection()
    checkpointer = InMemorySaver()
    write_client = write_client if write_client is not None else FakeGitHubWriteClient()
    write_client.seed_default_branch("main", commit_sha="base-sha-1", tree_sha="base-tree-1")

    monkeypatch.setattr(proposals_module, "repository_store", sample_store)
    monkeypatch.setattr(proposals_module, "EmbeddingService", _DummyService)
    monkeypatch.setattr(proposals_module, "PineconeService", _DummyService)
    monkeypatch.setattr(proposals_module, "RepositorySearchService", lambda **kwargs: fake_search_service)
    monkeypatch.setattr(proposals_module, "get_chat_model", lambda **kwargs: chat_model)
    monkeypatch.setattr(proposals_module, "get_conversation_store", lambda settings: ConversationStore(conversations))
    monkeypatch.setattr(proposals_module, "get_checkpointer", lambda settings: checkpointer)
    monkeypatch.setattr(proposals_module, "get_proposal_store", lambda settings: ProposalStore(proposals))
    monkeypatch.setattr(proposals_module, "get_proposal_audit_store", lambda settings: ProposalAuditStore(audit))
    # apply_patch_node imports GitHubWriteClient at module level (same
    # convention as EmbeddingService/PineconeService above) -- never a real
    # GitHub call in these tests.
    monkeypatch.setattr(apply_patch_module, "GitHubWriteClient", lambda **kwargs: write_client)

    return conversations, proposals, audit, checkpointer, write_client


# --- 7. Proposal persistence; creation reaches PENDING_APPROVAL ---


def test_create_proposal_persists_and_reaches_pending_approval(monkeypatch, sample_store, fake_search_service):
    _patch_everything(monkeypatch, sample_store, fake_search_service, _happy_path_model())
    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)

    response = client.post("/api/v1/proposals", json={"bug_report": BUG_REPORT})
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "PENDING_APPROVAL"
    assert body["proposal_id"] == body["thread_id"]
    assert body["affected_files"] == ["app/auth/deactivate.py"]
    assert "session_token = None" in body["unified_diff"]

    # Persisted, not just returned once.
    fetched = client.get(f"/api/v1/proposals/{body['proposal_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "PENDING_APPROVAL"


def test_create_proposal_requires_mongo_configured(sample_store):
    # Explicitly force mongodb_uri=None rather than relying on the ambient
    # .env lacking one -- this project's real .env has a genuine MONGODB_URI
    # configured (from Milestone 7's live Atlas testing), so the "unconfigured"
    # case must be constructed, not assumed.
    def _settings_without_mongo() -> Settings:
        real = get_settings()
        return Settings(**{**real.model_dump(), "mongodb_uri": None})

    fastapi_app.dependency_overrides[get_settings] = _settings_without_mongo
    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)
    response = client.post("/api/v1/proposals", json={"bug_report": BUG_REPORT})
    assert response.status_code == 503


def test_create_proposal_requires_a_repository(monkeypatch):
    from app.ingestion.store import RepositoryStore

    _patch_everything(monkeypatch, RepositoryStore(), None, _happy_path_model())
    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)
    response = client.post("/api/v1/proposals", json={"bug_report": BUG_REPORT})
    assert response.status_code == 409


# --- 12. Duplicate approval/rejection requests are safely handled ---


def test_duplicate_approval_is_rejected(monkeypatch, sample_store, fake_search_service):
    _patch_everything(monkeypatch, sample_store, fake_search_service, _happy_path_model())
    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)

    proposal_id = client.post("/api/v1/proposals", json={"bug_report": BUG_REPORT}).json()["proposal_id"]

    first = client.post(f"/api/v1/proposals/{proposal_id}/approve", json={"reason": "ok"})
    assert first.status_code == 200
    assert first.json()["status"] == "APPLIED"

    second = client.post(f"/api/v1/proposals/{proposal_id}/approve", json={"reason": "again"})
    assert second.status_code == 409  # already decided -- not silently re-applied

    reject_after_approve = client.post(f"/api/v1/proposals/{proposal_id}/reject", json={"reason": "changed my mind"})
    assert reject_after_approve.status_code == 409


def test_reject_then_approve_is_rejected(monkeypatch, sample_store, fake_search_service):
    _patch_everything(monkeypatch, sample_store, fake_search_service, _happy_path_model())
    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)

    proposal_id = client.post("/api/v1/proposals", json={"bug_report": BUG_REPORT}).json()["proposal_id"]
    rejected = client.post(f"/api/v1/proposals/{proposal_id}/reject", json={"reason": "no"})
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "REJECTED"

    approve_after_reject = client.post(f"/api/v1/proposals/{proposal_id}/approve", json={"reason": "actually yes"})
    assert approve_after_reject.status_code == 409
    # The patch must never have been applied.
    final = client.get(f"/api/v1/proposals/{proposal_id}").json()
    assert final["status"] == "REJECTED"


# --- 13. Unauthorized proposal access is rejected ---


def test_unauthorized_session_cannot_access_or_decide(monkeypatch, sample_store, fake_search_service):
    _patch_everything(monkeypatch, sample_store, fake_search_service, _happy_path_model())
    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-owner"
    client = TestClient(fastapi_app)
    proposal_id = client.post("/api/v1/proposals", json={"bug_report": BUG_REPORT}).json()["proposal_id"]

    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-intruder"
    assert client.get(f"/api/v1/proposals/{proposal_id}").status_code == 404
    assert client.get(f"/api/v1/proposals/{proposal_id}/diff").status_code == 404
    assert client.post(f"/api/v1/proposals/{proposal_id}/approve", json={}).status_code == 404
    assert client.post(f"/api/v1/proposals/{proposal_id}/reject", json={}).status_code == 404
    assert client.get("/api/v1/proposals").json()["total"] == 0  # intruder sees none of the owner's proposals


# --- 14. Cross-repository proposal access (decision) is rejected ---


def test_approval_rejected_if_repository_context_changed(monkeypatch, sample_store, fake_search_service):
    _patch_everything(monkeypatch, sample_store, fake_search_service, _happy_path_model())
    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)
    proposal_id = client.post("/api/v1/proposals", json={"bug_report": BUG_REPORT}).json()["proposal_id"]

    # Simulate a different repository having since been ingested.
    from app.ingestion.store import IngestedRepository

    sample_store.save(IngestedRepository(owner="someone-else", repo="other-repo", default_branch="main", files=[]))

    response = client.post(f"/api/v1/proposals/{proposal_id}/approve", json={"reason": "ok"})
    assert response.status_code == 409


# --- 15/16. Approved patch only ever exists as data; original repo content never mutated ---


def test_approved_patch_does_not_mutate_original_repository(monkeypatch, sample_store, fake_search_service):
    _patch_everything(monkeypatch, sample_store, fake_search_service, _happy_path_model())
    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)

    original_content = next(f.content for f in sample_store.get_current().files if f.path == "app/auth/deactivate.py")

    proposal_id = client.post("/api/v1/proposals", json={"bug_report": BUG_REPORT}).json()["proposal_id"]
    approved = client.post(f"/api/v1/proposals/{proposal_id}/approve", json={"reason": "ok"})
    assert approved.status_code == 200
    assert approved.json()["status"] == "APPLIED"

    # The "applied" content exists only inside the proposal's own data.
    diff_response = client.get(f"/api/v1/proposals/{proposal_id}/diff")
    assert "session_token = None" in diff_response.json()["unified_diff"]

    # The original RepositoryStore snapshot is completely unchanged.
    current_content = next(f.content for f in sample_store.get_current().files if f.path == "app/auth/deactivate.py")
    assert current_content == original_content
    assert "session_token = None" not in current_content


# --- 17. Audit history records every decision ---


def test_audit_history_records_creation_and_decision(monkeypatch, sample_store, fake_search_service):
    _, _, audit_collection, _, _ = _patch_everything(monkeypatch, sample_store, fake_search_service, _happy_path_model())
    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)

    proposal_id = client.post("/api/v1/proposals", json={"bug_report": BUG_REPORT}).json()["proposal_id"]
    client.post(f"/api/v1/proposals/{proposal_id}/approve", json={"reason": "ok"})

    audit_store = ProposalAuditStore(audit_collection)
    history = asyncio.run(audit_store.history(proposal_id))
    events = [entry["event"] for entry in history]

    assert "created" in events
    assert "pending_approval" in events
    assert "approved" in events
    assert "applied" in events
    # Append-only: the "created" record is still there after later events.
    assert events[0] == "created"


# --- Milestone 9: GitHubWriteClient integration ---


def test_approval_missing_write_token_fails_without_any_github_call(monkeypatch, sample_store, fake_search_service):
    _, _, _, _, write_client = _patch_everything(monkeypatch, sample_store, fake_search_service, _happy_path_model())
    # Undo the fake for this one test: use the REAL GitHubWriteClient class,
    # so a missing token is caught by its own constructor, not by the fake.
    monkeypatch.setattr(apply_patch_module, "GitHubWriteClient", GitHubWriteClient)
    fastapi_app.dependency_overrides[get_settings] = lambda: _settings_with_mongo(github_write_token=None)
    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)

    proposal_id = client.post("/api/v1/proposals", json={"bug_report": BUG_REPORT}).json()["proposal_id"]
    response = client.post(f"/api/v1/proposals/{proposal_id}/approve", json={"reason": "ok"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "FAILED"
    assert "GITHUB_WRITE_TOKEN" in body["error"]
    assert body["apply_result"] is None
    assert write_client.calls == []  # the fake was swapped out; nothing on it was ever called either way


def test_approval_resume_performs_github_writes_exactly_once(monkeypatch, sample_store, fake_search_service):
    _, _, _, _, write_client = _patch_everything(monkeypatch, sample_store, fake_search_service, _happy_path_model())
    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)
    proposal_id = client.post("/api/v1/proposals", json={"bug_report": BUG_REPORT}).json()["proposal_id"]

    first = client.post(f"/api/v1/proposals/{proposal_id}/approve", json={"reason": "ok"})
    assert first.status_code == 200
    assert first.json()["status"] == "APPLIED"
    calls_after_first = list(write_client.calls)
    assert len(calls_after_first) > 0

    second = client.post(f"/api/v1/proposals/{proposal_id}/approve", json={"reason": "again"})
    assert second.status_code == 409  # already decided -- the graph is never even re-invoked

    # No additional GitHub write/read calls happened on the second attempt.
    assert write_client.calls == calls_after_first


def test_persisted_proposal_contains_branch_commit_and_pr_details(monkeypatch, sample_store, fake_search_service):
    _patch_everything(monkeypatch, sample_store, fake_search_service, _happy_path_model())
    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)
    proposal_id = client.post("/api/v1/proposals", json={"bug_report": BUG_REPORT}).json()["proposal_id"]

    approved = client.post(f"/api/v1/proposals/{proposal_id}/approve", json={"reason": "ok"})
    assert approved.status_code == 200
    assert approved.json()["status"] == "APPLIED"

    # Round-tripped through ProposalStore -- GET fetches it back from the
    # fake Mongo collection, not from the in-memory response object.
    fetched = client.get(f"/api/v1/proposals/{proposal_id}")
    assert fetched.status_code == 200
    apply_result = fetched.json()["apply_result"]
    assert apply_result is not None
    assert apply_result["branch_name"] == f"devpilot/proposal-{proposal_id}"
    assert apply_result["base_branch"] == "main"
    assert apply_result["commit_sha"] is not None
    assert apply_result["pull_request_number"] is not None
    assert apply_result["pull_request_url"] is not None
    assert apply_result["succeeded"] is True


def test_pr_creation_failure_preserves_branch_and_commit_details(monkeypatch, sample_store, fake_search_service):
    failing_client = FakeGitHubWriteClient(fail_at="create_pull_request")
    _patch_everything(monkeypatch, sample_store, fake_search_service, _happy_path_model(), write_client=failing_client)
    fastapi_app.dependency_overrides[get_session_id] = lambda: "session-a"
    client = TestClient(fastapi_app)
    proposal_id = client.post("/api/v1/proposals", json={"bug_report": BUG_REPORT}).json()["proposal_id"]

    response = client.post(f"/api/v1/proposals/{proposal_id}/approve", json={"reason": "ok"})
    assert response.status_code == 200
    body = response.json()

    # Never falsely reported as a full success.
    assert body["status"] == "APPLY_FAILED"
    assert "pull request" in body["error"].lower()
    apply_result = body["apply_result"]
    assert apply_result["branch_name"] == f"devpilot/proposal-{proposal_id}"
    assert apply_result["commit_sha"] is not None  # the commit was pushed before PR creation failed
    assert apply_result["pull_request_number"] is None
    assert apply_result["succeeded"] is False
