"""
LangGraph node: generate_report — assembles the final structured investigation
report.

Deliberately makes no LLM call: every field here is pure Python assembly from
pieces already produced by earlier nodes (bug_understanding, evidence,
evidence_assessment, root_cause_analysis) and from the recorded tool-call
history. This guarantees the report itself cannot introduce new fabrication —
it can only reflect what the earlier, already-grounded nodes produced.
"""

from app.agents.investigation_state import InvestigationState
from app.agents.message_utils import collect_tool_calls
from app.schemas.chat import ToolCallRecord
from app.schemas.investigation import InvestigationResponse


def generate_report_node(state: InvestigationState) -> dict:
    understanding = state.get("bug_understanding")
    assessment = state.get("evidence_assessment")
    analysis = state.get("root_cause_analysis")
    evidence = state.get("evidence", [])
    tool_calls = collect_tool_calls(state["messages"])

    report = (
        _complete_report(state, understanding, evidence, analysis, tool_calls)
        if analysis is not None
        else _inconclusive_report(state, understanding, evidence, assessment, tool_calls)
    )

    return {"final_report": report, "status": report.status}


def _complete_report(state, understanding, evidence, analysis, tool_calls) -> InvestigationResponse:
    return InvestigationResponse(
        status="complete",
        bug_summary=state["bug_report"],
        expected_behavior=understanding.expected_behavior if understanding else "",
        actual_behavior=understanding.actual_behavior if understanding else "",
        investigation_performed=_describe_tool_calls(tool_calls),
        root_cause=analysis.root_cause,
        confirmed_findings=analysis.confirmed_findings,
        hypotheses=analysis.hypotheses,
        supporting_evidence=evidence,
        affected_files=_affected_files(evidence),
        confidence_level=analysis.confidence_level,
        recommended_fix_direction=analysis.recommended_fix_direction,
        remaining_unknowns=analysis.remaining_unknowns,
        tool_calls=tool_calls,
    )


def _inconclusive_report(state, understanding, evidence, assessment, tool_calls) -> InvestigationResponse:
    missing = assessment.missing_evidence if assessment else []
    recommended = (
        "Further investigation needed: " + "; ".join(missing)
        if missing
        else "Provide more specific details about the bug (e.g. which part of the app, "
        "what steps reproduce it) or investigate manually — the available tools did not "
        "surface enough evidence."
    )
    return InvestigationResponse(
        status="inconclusive",
        bug_summary=state["bug_report"],
        expected_behavior=understanding.expected_behavior if understanding else "",
        actual_behavior=understanding.actual_behavior if understanding else "",
        investigation_performed=_describe_tool_calls(tool_calls),
        root_cause="Insufficient evidence was found to determine a root cause.",
        supporting_evidence=evidence,
        affected_files=_affected_files(evidence),
        confidence_level="inconclusive",
        recommended_fix_direction=recommended,
        remaining_unknowns=missing,
        tool_calls=tool_calls,
    )


def _affected_files(evidence) -> list[str]:
    return sorted({item.file_path for item in evidence})


def _describe_tool_calls(tool_calls: list[ToolCallRecord]) -> list[str]:
    descriptions = []
    for call in tool_calls:
        args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
        outcome = "ok" if call.status == "success" else f"failed ({call.status})"
        descriptions.append(f"{call.tool}({args}) — {outcome}")
    return descriptions
