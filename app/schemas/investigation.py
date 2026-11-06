"""
Schemas for the bug-investigation workflow (Milestone 6).

These Pydantic models serve double duty: they are both the structured-output
targets passed to `chat_model.with_structured_output(...)` inside the
investigation graph's nodes, and the FastAPI request/response contract for
`POST /api/v1/investigate`. Reusing the same models avoids a third, redundant
set of "internal" shapes.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.chat import ToolCallRecord


class InvestigateRequest(BaseModel):
    bug_report: str = Field(..., min_length=1, description="Free-text description of the reported bug.")
    # Milestone 7, additive: omit for the original Milestone 6 ephemeral behavior.
    # Passing an existing investigation thread_id starts a fresh investigation
    # under that same conversation record (see app.api.routes.investigate for why
    # investigation threads are always reset rather than resumed).
    thread_id: Optional[str] = Field(default=None)


class BugUnderstanding(BaseModel):
    """Output of the understand_bug node — purely an interpretation of the report, no code inspected yet."""

    expected_behavior: str
    actual_behavior: str
    keywords: list[str] = Field(default_factory=list)
    investigation_areas: list[str] = Field(
        default_factory=list, description="Candidate components/areas to investigate, e.g. 'authentication'."
    )


class InvestigationPlanStep(BaseModel):
    area: str
    rationale: str


class InvestigationPlan(BaseModel):
    """Output of the plan_investigation node."""

    summary: str
    steps: list[InvestigationPlanStep] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """
    One piece of evidence. Always derived directly from the in-memory repository
    store or an actual tool call's recorded arguments — never from free-text LLM
    output — so every field here is traceable to real ingested content.
    """

    source_tool: Literal["search_codebase", "read_repository_file"]
    file_path: str
    start_line: int | None = None
    end_line: int | None = None
    snippet: str


class VerifiedFinding(BaseModel):
    claim: str
    supporting_evidence_refs: list[str] = Field(
        default_factory=list, description="References into supporting_evidence, e.g. 'README.md:1-9'."
    )
    confidence: Literal["confirmed", "likely", "hypothesis"]


class EvidenceAssessment(BaseModel):
    """Output of the verify_evidence node."""

    sufficient: bool
    verified_findings: list[VerifiedFinding] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    reasoning: str


class RootCauseAnalysis(BaseModel):
    """Output of the analyze_root_cause node — only reached when evidence was judged sufficient."""

    root_cause: str
    explanation: str
    confirmed_findings: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    confidence_level: Literal["high", "medium", "low"]
    recommended_fix_direction: str
    remaining_unknowns: list[str] = Field(default_factory=list)


class InvestigationResponse(BaseModel):
    """Final structured report — both the graph's `final_report` state value and the API response body."""

    status: Literal["complete", "inconclusive"]
    bug_summary: str
    expected_behavior: str
    actual_behavior: str
    investigation_performed: list[str] = Field(default_factory=list)
    root_cause: str
    confirmed_findings: list[str] = Field(
        default_factory=list, description="Findings directly backed by retrieved evidence."
    )
    hypotheses: list[str] = Field(
        default_factory=list, description="Plausible but not directly confirmed explanations."
    )
    supporting_evidence: list[EvidenceItem] = Field(default_factory=list)
    affected_files: list[str] = Field(default_factory=list)
    confidence_level: Literal["high", "medium", "low", "inconclusive"]
    recommended_fix_direction: str
    remaining_unknowns: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    # Milestone 7, additive: echoes the thread_id used, or None if ephemeral.
    thread_id: Optional[str] = Field(default=None)
