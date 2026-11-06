"""
LangGraph node: investigate_bug.

Reuses the entire Milestone 6 investigation graph (app.agents.investigation_graph)
as an ephemeral, nested run — this is composition, not duplication: zero
investigation logic is reimplemented here. The nested run has no checkpointer
of its own; its final report is folded into *this* graph's own checkpointed
state instead of being independently resumable (an investigation is a
single-shot workflow with no valid mid-run resume point — see
app.agents.investigation_graph's own docstring).
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from app.agents.investigation_graph import build_investigation_graph
from app.agents.investigation_state import InvestigationState
from app.agents.proposal_state import ProposalState
from app.ingestion.store import RepositoryStore


async def investigate_bug_node(
    state: ProposalState,
    chat_model: BaseChatModel,
    tools: list[BaseTool],
    store: RepositoryStore,
    max_investigation_rounds: int,
    recursion_limit: int,
) -> dict:
    investigation_graph = build_investigation_graph(chat_model=chat_model, tools=tools, store=store)

    initial_state: InvestigationState = {
        "repository_id": state["repository_id"],
        "bug_report": state["bug_report"],
        "max_investigation_rounds": max_investigation_rounds,
        "investigation_round": 0,
        "bug_understanding": None,
        "investigation_plan": None,
        "messages": [],
        "evidence": [],
        "evidence_assessment": None,
        "root_cause_analysis": None,
        "final_report": None,
        "status": "understanding",
    }
    final_state = await investigation_graph.ainvoke(initial_state, config={"recursion_limit": recursion_limit})
    report = final_state.get("final_report")

    if report is None or report.status != "complete":
        status_word = report.status if report else "none"
        return {
            "investigation_report": report,
            "status": "FAILED",
            "error": (
                f"Investigation did not reach a confirmed root cause (status: {status_word}); "
                "cannot propose a fix from an unverified hypothesis alone."
            ),
        }

    return {"investigation_report": report, "status": "DRAFT"}
