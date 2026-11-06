"""
State for the Milestone 6 bug-investigation graph.

Deliberately separate from app.agents.state.AgentState (the Milestone 5 chat
state) rather than extended from it: the two graphs serve different shapes of
work (one-shot Q&A vs. a multi-stage investigation with its own plan/evidence/
verification bookkeeping), and conflating them would make both harder to
reason about. They do share the same `messages` pattern (Annotated with
add_messages) and reuse the same agent_node/ToolNode/tools_condition
primitives for the tool-calling portion.

Structured fields (BugUnderstanding, InvestigationPlan, EvidenceAssessment,
RootCauseAnalysis, InvestigationResponse) are the exact Pydantic models used
as `with_structured_output(...)` targets — stored directly in state rather
than re-parsed/duplicated into a second shape.
"""

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.schemas.investigation import (
    BugUnderstanding,
    EvidenceAssessment,
    EvidenceItem,
    InvestigationPlan,
    InvestigationResponse,
    RootCauseAnalysis,
)


class InvestigationState(TypedDict):
    repository_id: str
    bug_report: str

    # Safety bound on the verify -> continue_investigation -> investigate loop,
    # independent of (and tighter than) the graph-level recursion_limit, so the
    # workflow stops for a deliberate, reportable reason rather than an opaque
    # GraphRecursionError.
    max_investigation_rounds: int
    investigation_round: int

    bug_understanding: Optional[BugUnderstanding]
    investigation_plan: Optional[InvestigationPlan]

    # The ReAct investigation conversation — reuses the same agent_node/ToolNode
    # primitives as the Milestone 5 chat graph.
    messages: Annotated[list[AnyMessage], add_messages]

    # Recomputed from `messages` each time verify_evidence runs (not
    # accumulated independently), so there is exactly one source of truth for
    # "what was actually retrieved."
    evidence: list[EvidenceItem]
    evidence_assessment: Optional[EvidenceAssessment]
    root_cause_analysis: Optional[RootCauseAnalysis]

    final_report: Optional[InvestigationResponse]
    status: str
