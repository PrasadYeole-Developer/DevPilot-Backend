"""
Offline tests for the Milestone 8 human-in-the-loop proposal graph.

Uses the real langgraph.types.interrupt/Command(resume=...) mechanism (via
langgraph.checkpoint.memory.InMemorySaver, which implements the same
BaseCheckpointSaver interface MongoDBSaver does) — not a simulated boolean
check. All Gemini calls are mocked via ScriptedChatModel; Pinecone search is
bypassed via FakeSearchService.
"""

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agents.proposal_graph import build_proposal_graph
from app.agents.tools import build_tools
from app.schemas.investigation import BugUnderstanding, EvidenceAssessment, InvestigationPlan, RootCauseAnalysis
from app.schemas.proposal import ProposalDraft, ProposedFileChangeDraft
from tests.conftest import ScriptedChatModel

pytestmark = pytest.mark.anyio

BUG_REPORT = "Deactivated staff members can still access the application after their account is disabled."


def _read_call(file_path: str, call_id: str = "call_read") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": "read_repository_file", "args": {"file_path": file_path}, "id": call_id}]
    )


def _summary(text: str) -> AIMessage:
    return AIMessage(content=text, tool_calls=[])


def _happy_path_responses() -> dict:
    return {
        BugUnderstanding: BugUnderstanding(
            expected_behavior="Deactivated users cannot access the app.",
            actual_behavior="Deactivated users can still access the app.",
            keywords=["deactivate", "session"],
            investigation_areas=["authentication"],
        ),
        InvestigationPlan: InvestigationPlan(summary="Check deactivation logic.", steps=[]),
        EvidenceAssessment: EvidenceAssessment(
            sufficient=True, verified_findings=[], contradictions=[], missing_evidence=[], reasoning="r"
        ),
        RootCauseAnalysis: RootCauseAnalysis(
            root_cause="Sessions are not revoked on deactivation.",
            explanation="deactivate_user sets is_active=False but never invalidates the session.",
            confirmed_findings=["deactivate_user() never revokes sessions (app/auth/deactivate.py)"],
            hypotheses=[],
            confidence_level="high",
            recommended_fix_direction="Revoke sessions in deactivate_user.",
            remaining_unknowns=[],
        ),
        ProposalDraft: ProposalDraft(
            explanation="Clear the session token when a user is deactivated.",
            expected_behavior="Deactivated users are immediately logged out.",
            risk_level="low",
            assumptions=["Setting session_token to None is sufficient to invalidate the session."],
            additional_files_needed=[],
            changes=[
                ProposedFileChangeDraft(
                    file_path="app/auth/deactivate.py",
                    change_summary="Clear session_token on deactivation.",
                    proposed_content=(
                        "def deactivate_user(user):\n"
                        "    user.is_active = False\n"
                        "    user.session_token = None\n"
                        "    db.save(user)\n"
                        "    # Existing sessions are not revoked here.\n"
                    ),
                )
            ],
        ),
    }


def _build_graph(chat_model, store, search_service, checkpointer):
    tools = build_tools(search_service=search_service, store=store)
    return build_proposal_graph(chat_model=chat_model, tools=tools, store=store, checkpointer=checkpointer)


def _initial_state(proposal_id: str, repository_id: str, bug_report: str) -> dict:
    return {
        "proposal_id": proposal_id,
        "repository_id": repository_id,
        "bug_report": bug_report,
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


# --- 1. Proposal generation from verified investigation findings; pauses at the interrupt ---


async def test_proposal_generation_pauses_at_approval_interrupt(sample_store, fake_search_service):
    model = ScriptedChatModel(
        structured_responses=_happy_path_responses(),
        tool_responses=[_read_call("app/auth/deactivate.py"), _summary("Confirmed deactivate_user never revokes sessions.")],
    )
    checkpointer = InMemorySaver()
    graph = _build_graph(model, sample_store, fake_search_service, checkpointer)
    config = {"configurable": {"thread_id": "prop-1"}, "recursion_limit": 50}

    result = await graph.ainvoke(_initial_state("prop-1", "acme/webapp", BUG_REPORT), config=config)

    assert "__interrupt__" in result, "graph must pause at request_human_approval, not complete"
    assert result["status"] == "PENDING_APPROVAL"
    assert result["validation_result"].valid is True
    assert result["proposed_changes"][0].file_path == "app/auth/deactivate.py"
    assert "session_token = None" in result["proposed_changes"][0].unified_diff

    # Confirm via the official state API too, not just the raw dict: the
    # graph is genuinely paused before request_human_approval, not finished.
    state = await graph.aget_state(config)
    assert state.next == ("request_human_approval",)


# --- 2. Rejection of unsupported root-cause hypotheses (inconclusive investigation) ---


async def test_no_proposal_generated_from_inconclusive_investigation(sample_store, fake_search_service):
    always_insufficient = EvidenceAssessment(
        sufficient=False, verified_findings=[], contradictions=[], missing_evidence=["nothing found"], reasoning="r"
    )
    model = ScriptedChatModel(
        structured_responses={
            BugUnderstanding: BugUnderstanding(expected_behavior="e", actual_behavior="a", keywords=[], investigation_areas=[]),
            InvestigationPlan: InvestigationPlan(summary="s", steps=[]),
            EvidenceAssessment: always_insufficient,
            # ProposalDraft deliberately NOT provided -- if generate_proposal
            # were reached, this test would fail with a KeyError, proving it wasn't.
        },
        tool_responses=[_summary("Found nothing conclusive.") for _ in range(6)],
    )
    checkpointer = InMemorySaver()
    graph = _build_graph(model, sample_store, fake_search_service, checkpointer)
    config = {"configurable": {"thread_id": "prop-2"}, "recursion_limit": 100}

    result = await graph.ainvoke(_initial_state("prop-2", "acme/webapp", BUG_REPORT), config=config)

    assert "__interrupt__" not in result
    assert result["status"] == "FAILED"
    assert result["proposal_draft"] is None
    assert "unverified hypothesis" in result["error"] or "confirmed root cause" in result["error"]


# --- 9/11. Missing approval never applies changes; approved workflow resumes and applies ---


async def test_approved_decision_resumes_and_applies(sample_store, fake_search_service):
    model = ScriptedChatModel(
        structured_responses=_happy_path_responses(),
        tool_responses=[_read_call("app/auth/deactivate.py"), _summary("Confirmed.")],
    )
    checkpointer = InMemorySaver()
    graph = _build_graph(model, sample_store, fake_search_service, checkpointer)
    config = {"configurable": {"thread_id": "prop-3"}, "recursion_limit": 50}

    paused = await graph.ainvoke(_initial_state("prop-3", "acme/webapp", BUG_REPORT), config=config)
    assert "__interrupt__" in paused
    assert paused["status"] == "PENDING_APPROVAL"  # no approval yet -> definitely not applied

    original_before = next(f.content for f in sample_store.get_current().files if f.path == "app/auth/deactivate.py")

    result = await graph.ainvoke(Command(resume={"decision": "approved", "reason": "looks good"}), config=config)

    assert "__interrupt__" not in result
    assert result["decision"] == "approved"
    assert result["status"] == "APPLIED"

    # 16. The original ingested repository snapshot is never mutated.
    original_after = next(f.content for f in sample_store.get_current().files if f.path == "app/auth/deactivate.py")
    assert original_after == original_before
    assert "session_token = None" not in original_after


# --- 10. Rejected workflow never applies changes ---


async def test_rejected_decision_never_applies(sample_store, fake_search_service):
    model = ScriptedChatModel(
        structured_responses=_happy_path_responses(),
        tool_responses=[_read_call("app/auth/deactivate.py"), _summary("Confirmed.")],
    )
    checkpointer = InMemorySaver()
    graph = _build_graph(model, sample_store, fake_search_service, checkpointer)
    config = {"configurable": {"thread_id": "prop-4"}, "recursion_limit": 50}

    await graph.ainvoke(_initial_state("prop-4", "acme/webapp", BUG_REPORT), config=config)
    result = await graph.ainvoke(Command(resume={"decision": "rejected", "reason": "not the right fix"}), config=config)

    assert result["decision"] == "rejected"
    assert result["status"] == "REJECTED"
    # apply_patch must never have run: no APPLIED status, and the graph's
    # own structure (request_human_approval's conditional edge) makes it
    # unreachable on this branch -- confirmed via the final state's "next".
    state = await graph.aget_state(config)
    assert state.next == ()  # graph terminated, nothing left to run


# --- 19. Restart simulation: a new graph object, same checkpointer, resumes correctly ---


async def test_resume_works_after_simulated_restart(sample_store, fake_search_service):
    """
    Every real request builds a brand-new graph object (see
    app/api/routes/proposals.py) -- there is never a long-lived graph
    instance to begin with. So "survives a restart" means: a second,
    independently-built graph object sharing only the checkpointer (standing
    in for MongoDB) can resume a thread paused by the first.
    """
    model_before = ScriptedChatModel(
        structured_responses=_happy_path_responses(),
        tool_responses=[_read_call("app/auth/deactivate.py"), _summary("Confirmed.")],
    )
    checkpointer = InMemorySaver()  # stands in for "the same Mongo database"
    graph_before_restart = _build_graph(model_before, sample_store, fake_search_service, checkpointer)
    config = {"configurable": {"thread_id": "prop-restart"}, "recursion_limit": 50}

    paused = await graph_before_restart.ainvoke(
        _initial_state("prop-restart", "acme/webapp", BUG_REPORT), config=config
    )
    assert "__interrupt__" in paused

    # Simulate "the app restarted": a brand-new ScriptedChatModel and a
    # brand-new graph object, built independently, sharing only the
    # checkpointer.
    model_after = ScriptedChatModel(structured_responses=_happy_path_responses(), tool_responses=[])
    graph_after_restart = _build_graph(model_after, sample_store, fake_search_service, checkpointer)

    state = await graph_after_restart.aget_state(config)
    assert state.next == ("request_human_approval",), "paused state must survive across independent graph objects"

    result = await graph_after_restart.ainvoke(Command(resume={"decision": "approved", "reason": None}), config=config)

    assert "__interrupt__" not in result
    assert result["status"] == "APPLIED"
    # model_after.tool_call_count stayed 0 -- resuming never re-ran
    # investigate_bug/generate_proposal, proving it genuinely resumed from
    # the checkpoint rather than restarting the whole workflow.
    assert model_after.tool_call_count == 0
