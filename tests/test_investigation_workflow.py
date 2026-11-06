"""
Offline tests for the Milestone 6 bug-investigation workflow.

All Gemini calls are mocked via ScriptedChatModel (tests/conftest.py); Pinecone
embeddings are bypassed via FakeSearchService, which does plain substring
matching against the in-memory repository store. No real API calls are made.
"""

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agents.investigation_graph import build_investigation_graph
from app.agents.state import AgentState
from app.agents.nodes.analyze_root_cause import analyze_root_cause_node
from app.agents.nodes.generate_report import generate_report_node
from app.agents.nodes.plan_investigation import plan_investigation_node
from app.agents.nodes.understand_bug import understand_bug_node
from app.agents.nodes.verify_evidence import extract_evidence
from app.agents.tools import build_tools
from app.schemas.investigation import (
    BugUnderstanding,
    EvidenceAssessment,
    InvestigationPlan,
    InvestigationPlanStep,
    RootCauseAnalysis,
    VerifiedFinding,
)
from tests.conftest import ScriptedChatModel

pytestmark = pytest.mark.anyio

BUG_REPORT = "Deactivated staff members can still access the application after their accounts are disabled."


def _investigator_summary(text: str) -> AIMessage:
    return AIMessage(content=text, tool_calls=[])


def _search_call(query: str, call_id: str = "call_search") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "search_codebase", "args": {"query": query}, "id": call_id}])


def _read_call(file_path: str, call_id: str = "call_read") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": "read_repository_file", "args": {"file_path": file_path}, "id": call_id}]
    )


def _tool_only_graph(tools):
    """Minimal graph wrapping just a ToolNode — ToolNode.ainvoke() requires the
    runtime context a compiled graph provides, so it can't be invoked bare."""
    graph = StateGraph(AgentState)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return graph.compile()


# --- 1. A bug report can be understood and converted into an investigation plan. ---


async def test_bug_report_is_understood_and_planned(sample_store):
    understanding = BugUnderstanding(
        expected_behavior="Deactivated users cannot access the application.",
        actual_behavior="Deactivated users can still access the application.",
        keywords=["deactivate", "session", "is_active"],
        investigation_areas=["authentication", "session handling"],
    )
    plan = InvestigationPlan(
        summary="Check how sessions are validated and whether deactivation revokes them.",
        steps=[InvestigationPlanStep(area="session validation", rationale="confirm what is actually checked")],
    )
    model = ScriptedChatModel(
        structured_responses={BugUnderstanding: understanding, InvestigationPlan: plan},
        tool_responses=[],
    )

    state = {"bug_report": BUG_REPORT, "messages": []}
    update1 = await understand_bug_node(state, chat_model=model)
    assert update1["bug_understanding"] == understanding
    assert update1["status"] == "planning"

    state.update(update1)
    update2 = await plan_investigation_node(state, chat_model=model)
    assert update2["investigation_plan"] == plan
    assert update2["status"] == "investigating"
    assert len(update2["messages"]) == 1
    kickoff_text = update2["messages"][0].content
    assert "Deactivated staff members" in kickoff_text
    assert "session validation" in kickoff_text


# --- 2. The investigation can call existing repository tools. ---


async def test_investigation_calls_existing_tools(sample_store, fake_search_service):
    tools = build_tools(search_service=fake_search_service, store=sample_store)
    model = ScriptedChatModel(
        structured_responses={
            BugUnderstanding: BugUnderstanding(
                expected_behavior="e", actual_behavior="a", keywords=["session"], investigation_areas=["auth"]
            ),
            InvestigationPlan: InvestigationPlan(summary="s", steps=[]),
            EvidenceAssessment: EvidenceAssessment(
                sufficient=True, verified_findings=[], contradictions=[], missing_evidence=[], reasoning="r"
            ),
            RootCauseAnalysis: RootCauseAnalysis(
                root_cause="rc", explanation="e", confirmed_findings=[], hypotheses=[],
                confidence_level="low", recommended_fix_direction="f", remaining_unknowns=[],
            ),
        },
        tool_responses=[
            _search_call("session"),
            _investigator_summary("Found session validation logic."),
        ],
    )

    graph = build_investigation_graph(chat_model=model, tools=tools, store=sample_store)
    final_state = await graph.ainvoke(
        {
            "repository_id": "acme/webapp",
            "bug_report": BUG_REPORT,
            "max_investigation_rounds": 3,
            "investigation_round": 0,
            "bug_understanding": None,
            "investigation_plan": None,
            "messages": [],
            "evidence": [],
            "evidence_assessment": None,
            "root_cause_analysis": None,
            "final_report": None,
            "status": "understanding",
        },
        config={"recursion_limit": 30},
    )

    assert fake_search_service.calls == [("session", 5)]
    tool_messages = [m for m in final_state["messages"] if getattr(m, "name", None) == "search_codebase"]
    assert len(tool_messages) == 1
    assert final_state["final_report"].status == "complete"


# --- 3. Evidence is collected and associated with the correct file references. ---


async def test_evidence_matches_correct_file_references(sample_store, fake_search_service):
    tools = build_tools(search_service=fake_search_service, store=sample_store)
    graph = _tool_only_graph(tools)

    ai_message = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_codebase", "args": {"query": "session"}, "id": "call_1"},
            {"name": "read_repository_file", "args": {"file_path": "app/auth/deactivate.py"}, "id": "call_2"},
        ],
    )
    result = await graph.ainvoke({"messages": [ai_message], "repository_id": "acme/webapp"})
    messages = result["messages"]

    evidence = extract_evidence(messages, sample_store)
    search_hits = {item.file_path: item for item in evidence if item.source_tool == "search_codebase"}
    read_hits = {item.file_path: item for item in evidence if item.source_tool == "read_repository_file"}

    # "session" matches both files (deactivate.py mentions "sessions" in a comment) —
    # both are legitimately returned; evidence must reflect each correctly.
    assert "app/auth/session.py" in search_hits
    assert "session_token" in search_hits["app/auth/session.py"].snippet
    assert "app/auth/deactivate.py" in search_hits
    assert "is_active = False" in search_hits["app/auth/deactivate.py"].snippet

    assert read_hits["app/auth/deactivate.py"].file_path == "app/auth/deactivate.py"
    # Snippet must come from the actual stored file content, not LLM prose.
    repo = sample_store.get_current()
    actual_content = next(f.content for f in repo.files if f.path == "app/auth/deactivate.py")
    assert read_hits["app/auth/deactivate.py"].snippet == actual_content


async def test_evidence_rejects_tool_call_with_no_matching_file(sample_store, fake_search_service):
    """A read_repository_file call that failed (bad path) must not become fabricated evidence."""
    tools = build_tools(search_service=fake_search_service, store=sample_store)
    graph = _tool_only_graph(tools)

    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "read_repository_file", "args": {"file_path": "does/not/exist.py"}, "id": "call_1"}],
    )
    result = await graph.ainvoke({"messages": [ai_message], "repository_id": "acme/webapp"})
    messages = result["messages"]

    tool_message = next(m for m in messages if m.type == "tool")
    assert tool_message.status == "error"
    evidence = extract_evidence(messages, sample_store)
    assert evidence == []


# --- 4. Verified findings are distinguished from unsupported hypotheses. ---


def test_report_distinguishes_confirmed_findings_from_hypotheses():
    analysis = RootCauseAnalysis(
        root_cause="Deactivation does not revoke existing sessions.",
        explanation="is_session_valid only checks session_token presence; deactivate_user never invalidates it.",
        confirmed_findings=["is_session_valid does not check is_active (app/auth/session.py)"],
        hypotheses=["Sessions may also be cached client-side, compounding the issue"],
        confidence_level="medium",
        recommended_fix_direction="Check is_active in is_session_valid, or revoke sessions on deactivation.",
        remaining_unknowns=["How sessions are stored was not confirmed."],
    )
    state = {
        "bug_report": BUG_REPORT,
        "bug_understanding": BugUnderstanding(
            expected_behavior="e", actual_behavior="a", keywords=[], investigation_areas=[]
        ),
        "evidence": [],
        "evidence_assessment": None,
        "root_cause_analysis": analysis,
        "messages": [],
    }
    update = generate_report_node(state)
    report = update["final_report"]

    assert report.status == "complete"
    assert report.confirmed_findings == analysis.confirmed_findings
    assert report.hypotheses == analysis.hypotheses
    assert report.confirmed_findings != report.hypotheses


# --- 5. Insufficient evidence produces an honest inconclusive report. ---


def test_insufficient_evidence_produces_inconclusive_report():
    assessment = EvidenceAssessment(
        sufficient=False,
        verified_findings=[],
        contradictions=[],
        missing_evidence=["No code related to session revocation was found."],
        reasoning="No tool call ever located deactivation-related logic.",
    )
    state = {
        "bug_report": BUG_REPORT,
        "bug_understanding": BugUnderstanding(
            expected_behavior="e", actual_behavior="a", keywords=[], investigation_areas=[]
        ),
        "evidence": [],
        "evidence_assessment": assessment,
        "root_cause_analysis": None,  # analyze_root_cause never ran
        "messages": [],
    }
    update = generate_report_node(state)
    report = update["final_report"]

    assert report.status == "inconclusive"
    assert report.confidence_level == "inconclusive"
    assert "Insufficient evidence" in report.root_cause
    assert report.remaining_unknowns == assessment.missing_evidence
    assert "No code related to session revocation was found." in report.recommended_fix_direction


# --- 6. The workflow can repeat investigation when more evidence is needed. ---


async def test_workflow_repeats_investigation_when_evidence_insufficient(sample_store, fake_search_service):
    tools = build_tools(search_service=fake_search_service, store=sample_store)
    model = ScriptedChatModel(
        structured_responses={
            BugUnderstanding: BugUnderstanding(
                expected_behavior="e", actual_behavior="a", keywords=["session"], investigation_areas=["auth"]
            ),
            InvestigationPlan: InvestigationPlan(summary="s", steps=[]),
            EvidenceAssessment: [
                EvidenceAssessment(
                    sufficient=False, verified_findings=[], contradictions=[],
                    missing_evidence=["Need to check deactivate_user for session revocation."], reasoning="r1",
                ),
                EvidenceAssessment(
                    sufficient=True,
                    verified_findings=[
                        VerifiedFinding(
                            claim="deactivate_user does not revoke sessions",
                            supporting_evidence_refs=["app/auth/deactivate.py"],
                            confidence="confirmed",
                        )
                    ],
                    contradictions=[], missing_evidence=[], reasoning="r2",
                ),
            ],
            RootCauseAnalysis: RootCauseAnalysis(
                root_cause="Sessions are not revoked on deactivation.",
                explanation="deactivate_user sets is_active=False but never invalidates the session.",
                confirmed_findings=["app/auth/deactivate.py never revokes sessions"],
                hypotheses=[], confidence_level="high",
                recommended_fix_direction="Revoke sessions in deactivate_user.",
                remaining_unknowns=[],
            ),
        },
        tool_responses=[
            _search_call("session"),
            _investigator_summary("Session validation only checks token presence."),
            _read_call("app/auth/deactivate.py"),
            _investigator_summary("Confirmed deactivate_user never revokes sessions."),
        ],
    )

    graph = build_investigation_graph(chat_model=model, tools=tools, store=sample_store)
    final_state = await graph.ainvoke(
        {
            "repository_id": "acme/webapp",
            "bug_report": BUG_REPORT,
            "max_investigation_rounds": 3,
            "investigation_round": 0,
            "bug_understanding": None,
            "investigation_plan": None,
            "messages": [],
            "evidence": [],
            "evidence_assessment": None,
            "root_cause_analysis": None,
            "final_report": None,
            "status": "understanding",
        },
        config={"recursion_limit": 50},
    )

    assert final_state["investigation_round"] == 1, "continue_investigation must have run exactly once"
    assert final_state["final_report"].status == "complete"
    assert final_state["final_report"].confidence_level == "high"
    read_calls = [c.arguments["file_path"] for c in final_state["final_report"].tool_calls if c.tool == "read_repository_file"]
    assert "app/auth/deactivate.py" in read_calls


# --- 7. Investigation terminates safely without an infinite loop. ---


async def test_investigation_terminates_without_infinite_loop(sample_store, fake_search_service):
    tools = build_tools(search_service=fake_search_service, store=sample_store)
    always_insufficient = EvidenceAssessment(
        sufficient=False, verified_findings=[], contradictions=[],
        missing_evidence=["Still nothing concrete."], reasoning="never enough",
    )
    model = ScriptedChatModel(
        structured_responses={
            BugUnderstanding: BugUnderstanding(
                expected_behavior="e", actual_behavior="a", keywords=[], investigation_areas=[]
            ),
            InvestigationPlan: InvestigationPlan(summary="s", steps=[]),
            EvidenceAssessment: always_insufficient,  # repeated every round via itertools.repeat
        },
        tool_responses=[
            _investigator_summary("I found nothing conclusive.")
            for _ in range(10)  # far more than max_investigation_rounds would ever need
        ],
    )

    graph = build_investigation_graph(chat_model=model, tools=tools, store=sample_store)
    final_state = await graph.ainvoke(
        {
            "repository_id": "acme/webapp",
            "bug_report": BUG_REPORT,
            "max_investigation_rounds": 2,
            "investigation_round": 0,
            "bug_understanding": None,
            "investigation_plan": None,
            "messages": [],
            "evidence": [],
            "evidence_assessment": None,
            "root_cause_analysis": None,
            "final_report": None,
            "status": "understanding",
        },
        config={"recursion_limit": 100},
    )

    assert final_state["investigation_round"] == 2, "must stop exactly at max_investigation_rounds"
    assert final_state["final_report"].status == "inconclusive"
    assert final_state["root_cause_analysis"] is None, "analyze_root_cause must never run when evidence is insufficient"
