"""
State for the Milestone 8 human-in-the-loop code-change proposal graph.

Separate from InvestigationState/AgentState for the same reason those two
are separate from each other: a different shape of work with its own
bookkeeping (draft/validation/decision/application), even though it reuses
the investigation workflow internally (see app.agents.nodes.investigate_bug)
rather than duplicating it.
"""

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.schemas.investigation import InvestigationResponse
from app.schemas.proposal import ProposalDraft, ProposedFileChange, ValidationResult


class ProposalState(TypedDict):
    proposal_id: str
    repository_id: str
    bug_report: str

    investigation_report: Optional[InvestigationResponse]

    # Used only by generate_proposal's internal context-gathering pass(es) —
    # not a ReAct tool-calling loop, just a place to keep the structured-output
    # exchange if ever inspected/debugged.
    messages: Annotated[list[AnyMessage], add_messages]

    proposal_draft: Optional[ProposalDraft]
    proposed_changes: list[ProposedFileChange]
    validation_result: Optional[ValidationResult]

    decision: Optional[str]  # "approved" | "rejected", set only after the interrupt resumes
    decision_reason: Optional[str]

    status: str  # mirrors ProposalStatus.value
    error: Optional[str]
