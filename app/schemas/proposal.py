"""
Schemas for the human-in-the-loop code-change proposal workflow (Milestone 8).

As in app.schemas.investigation, some models serve double duty: both as
`with_structured_output(...)` targets inside the proposal graph's nodes, and
as the FastAPI request/response contract.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ProposalStatus(str, Enum):
    DRAFT = "DRAFT"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    APPLIED = "APPLIED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"


# Statuses a decision (approve/reject) may be made from. Anything else means
# "already decided" or "never reached approval" — attempting a decision there
# is rejected, not silently accepted. See app/api/routes/proposals.py.
DECIDABLE_STATUSES = {ProposalStatus.PENDING_APPROVAL}


class CreateProposalRequest(BaseModel):
    bug_report: str = Field(..., min_length=1, description="Free-text description of the reported bug.")
    # Unlike chat/investigate, there is no thread_id reuse here: every
    # proposal gets its own freshly-generated thread server-side, since a
    # proposal's lifecycle (draft -> pending -> decided) only ever happens
    # once per thread. Proposals also always require MongoDB (pausing for
    # approval is meaningless without persistence) — there is no ephemeral
    # mode for this workflow at all.


class ApprovalDecisionRequest(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=2000)


# --- LLM-facing structured-output targets (generate_proposal node) ---


class ProposedFileChangeDraft(BaseModel):
    """What Gemini proposes for one file — content only; the diff is always computed server-side."""

    file_path: str
    change_summary: str
    proposed_content: str


class ProposalDraft(BaseModel):
    """Output of the generate_proposal node's structured-output call."""

    explanation: str = Field(description="Why these changes fix the verified root cause.")
    expected_behavior: str = Field(description="What should happen after the fix is applied.")
    risk_level: Literal["low", "medium", "high"]
    assumptions: list[str] = Field(
        default_factory=list, description="Anything taken on faith beyond the verified evidence."
    )
    additional_files_needed: list[str] = Field(
        default_factory=list,
        description="Paths the model wants to see before it can propose confidently (triggers one extra context pass).",
    )
    changes: list[ProposedFileChangeDraft] = Field(default_factory=list)


# --- Server-validated, finalized shapes ---


class ProposedFileChange(BaseModel):
    """
    One finalized, server-validated file change. The diff is always computed
    by app.services.patch_service from (original, proposed) content — never
    trusted from the LLM.
    """

    file_path: str
    original_content_hash: str = Field(description="sha256 of the file's content when the proposal was drafted.")
    proposed_content: str
    unified_diff: str
    change_summary: str


class ValidationIssue(BaseModel):
    code: str
    message: str
    file_path: Optional[str] = None


class ValidationResult(BaseModel):
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class ProposalResponse(BaseModel):
    """Final structured proposal — both the `proposals` collection document shape and the API response body."""

    proposal_id: str
    thread_id: str
    repository_id: str
    bug_description: str
    root_cause: str
    expected_behavior: str
    explanation: str
    assumptions: list[str] = Field(default_factory=list)
    proposed_changes: list[ProposedFileChange] = Field(default_factory=list)
    affected_files: list[str] = Field(default_factory=list)
    unified_diff: str = Field(default="", description="All files' diffs concatenated, for convenience.")
    risk_level: Optional[Literal["low", "medium", "high"]] = None
    validation_results: ValidationResult = Field(default_factory=lambda: ValidationResult(valid=False))
    status: ProposalStatus
    error: Optional[str] = Field(default=None, description="Set when status is FAILED.")
    decision: Optional[Literal["approved", "rejected"]] = None
    decision_reason: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ProposalListResponse(BaseModel):
    proposals: list[ProposalResponse]
    total: int
    limit: int
    offset: int


class ProposalDiffResponse(BaseModel):
    proposal_id: str
    status: ProposalStatus
    unified_diff: str
    files: list[ProposedFileChange] = Field(default_factory=list)


# --- Audit trail entries (the `proposal_audit` collection) ---


class ProposalAuditEvent(BaseModel):
    proposal_id: str
    event: Literal[
        "created", "validated", "validation_failed", "pending_approval",
        "approved", "rejected", "applied", "apply_failed",
    ]
    session_id: str
    detail: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime
