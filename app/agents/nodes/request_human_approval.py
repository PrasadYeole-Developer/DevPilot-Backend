"""
LangGraph node: request_human_approval.

The actual LangGraph human-in-the-loop interrupt (langgraph.types.interrupt),
not a simulated boolean check. On first execution, `interrupt(payload)`
raises internally — LangGraph catches it, persists the paused state via the
graph's checkpointer, and `graph.ainvoke(...)` returns with a `__interrupt__`
key in its result instead of completing. There is no valid code path from
here to apply_patch without that pause actually happening — the interrupt
IS the approval gate, not a check guarding it.

To resume, the caller re-invokes the same thread with
`Command(resume={"decision": ..., "reason": ...})` (see
app/api/routes/proposals.py). LangGraph then re-runs this node from the top;
`interrupt(...)` immediately returns the resume value instead of pausing
again, and the rest of the function executes normally.

Important: everything before the `interrupt()` call re-runs on resume too —
there is nothing here but the interrupt call itself, so that's safe by
construction.
"""

from langgraph.types import interrupt

from app.agents.proposal_state import ProposalState


def request_human_approval_node(state: ProposalState) -> dict:
    decision_payload = interrupt(
        {
            "proposal_id": state["proposal_id"],
            "repository_id": state["repository_id"],
            "root_cause": state["investigation_report"].root_cause,
            "risk_level": state["proposal_draft"].risk_level,
            "affected_files": [change.file_path for change in state["proposed_changes"]],
            "message": "Approve or reject this proposed code change.",
        }
    )

    decision = decision_payload.get("decision")
    reason = decision_payload.get("reason")
    status = "APPROVED" if decision == "approved" else "REJECTED"
    return {"decision": decision, "decision_reason": reason, "status": status}
