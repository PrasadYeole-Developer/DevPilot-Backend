"""
LangGraph node: analyze_root_cause — produces root-cause analysis.

Only reached when verify_evidence judged the evidence sufficient. Takes the
verified findings and raw evidence (never the investigator's unverified
summary alone) as input, so the analysis is grounded in what verify_evidence
already confirmed is actually backed by retrieved content.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.investigation_state import InvestigationState
from app.schemas.investigation import EvidenceAssessment, EvidenceItem, RootCauseAnalysis

SYSTEM_PROMPT = """You explain the likely root cause of a reported bug, based ONLY \
on the verified findings and evidence provided below.

Rules:
- Connect the relevant code paths to explain how they produce the reported behavior.
- Clearly separate confirmed_findings (directly supported by the evidence) from \
hypotheses (plausible but not directly confirmed).
- Set confidence_level honestly: "high" only if the evidence directly demonstrates \
the mechanism; "low" if you are mostly reasoning from partial evidence.
- recommended_fix_direction should describe where/what to look at fixing, not a full \
patch — do not write new code.
- List remaining_unknowns: anything that would still need checking to be fully \
certain, even if you're confident overall."""


async def analyze_root_cause_node(state: InvestigationState, chat_model: BaseChatModel) -> dict:
    assessment = state["evidence_assessment"]
    evidence = state["evidence"]

    structured_model = chat_model.with_structured_output(RootCauseAnalysis)
    prompt = (
        f"Bug report: {state['bug_report']}\n\n"
        f"Verified findings:\n{_format_findings(assessment)}\n\n"
        f"Contradictions noted during verification: {assessment.contradictions or 'none'}\n\n"
        f"Evidence:\n{_format_evidence(evidence)}"
    )
    analysis: RootCauseAnalysis = await structured_model.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    return {"root_cause_analysis": analysis, "status": "analyzing"}


def _format_findings(assessment: EvidenceAssessment) -> str:
    if not assessment.verified_findings:
        return "(none)"
    return "\n".join(
        f"- [{finding.confidence}] {finding.claim} (refs: {', '.join(finding.supporting_evidence_refs) or 'none'})"
        for finding in assessment.verified_findings
    )


def _format_evidence(evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return "(none)"
    return "\n\n".join(
        f"--- {item.file_path} "
        f"({f'lines {item.start_line}-{item.end_line}' if item.start_line else 'full file'}) ---\n{item.snippet}"
        for item in evidence
    )
