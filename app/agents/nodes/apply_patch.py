"""
LangGraph node: apply_patch.

Only reachable from request_human_approval's "approved" branch (see
app.agents.proposal_graph's conditional edges) — the graph structure itself,
not a flag check, is what guarantees a rejected or still-pending proposal can
never reach here.

No LLM call, no filesystem writes, no modification of the original
RepositoryStore snapshot: "applying" means (1) re-checking that the proposal
isn't stale against whatever is currently ingested, since time may have
passed since generation/validation, then (2) marking it APPLIED. The
"applied" artifact IS the proposal's own proposed_content/unified_diff
(finalized back in validate_proposal) — retrievable via
GET /api/v1/proposals/{id} and .../diff, never merged back into the
in-memory repository snapshot or any real file.
"""

from app.agents.proposal_state import ProposalState
from app.ingestion.store import RepositoryStore
from app.services.patch_service import check_for_staleness


def apply_patch_node(state: ProposalState, store: RepositoryStore) -> dict:
    repository = store.get_current()
    if repository is None or f"{repository.owner}/{repository.repo}" != state["repository_id"]:
        return {
            "status": "FAILED",
            "error": "The currently ingested repository no longer matches this proposal's repository.",
        }

    staleness_issues = check_for_staleness(state["proposed_changes"], repository)
    if staleness_issues:
        return {
            "status": "FAILED",
            "error": "Proposal is stale: " + "; ".join(issue.message for issue in staleness_issues),
        }

    return {"status": "APPLIED"}
