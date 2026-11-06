"""
LangGraph node: validate_proposal.

No LLM call. Independent, deterministic, server-side validation of what
generate_proposal drafted — path safety, file existence, allowed extensions,
diff size limits — via app.services.patch_service. This is the enforcement
point for "the server must independently validate the proposal; never trust
paths, patch content, or validation results supplied only by the LLM."
"""

from app.agents.proposal_state import ProposalState
from app.ingestion.store import RepositoryStore
from app.services.patch_service import validate_proposed_changes


def validate_proposal_node(state: ProposalState, store: RepositoryStore) -> dict:
    draft = state["proposal_draft"]
    repository = store.get_current()
    if repository is None:
        return {"status": "FAILED", "error": "No repository is currently ingested."}

    changes, result = validate_proposed_changes(draft.changes, repository)

    if not result.valid:
        return {
            "proposed_changes": changes,
            "validation_result": result,
            "status": "FAILED",
            "error": "Proposal failed validation: " + "; ".join(issue.message for issue in result.issues),
        }

    return {"proposed_changes": changes, "validation_result": result, "status": "PENDING_APPROVAL"}
