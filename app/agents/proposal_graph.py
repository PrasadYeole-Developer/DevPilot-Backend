"""
Builds the DevPilot human-in-the-loop code-change proposal graph (Milestone 8).

    START -> investigate_bug
                 |
      (status != FAILED)  (status == FAILED, e.g. inconclusive investigation)
                 |                           |
                 v                           v
          generate_proposal                 END
                 |
          validate_proposal
                 |
      (status != FAILED)  (status == FAILED, e.g. path/size/extension violation)
                 |                           |
                 v                           v
      request_human_approval                END
          [INTERRUPT -- pauses here;
           resumes via Command(resume=...)]
                 |
      (decision == "approved")  (decision == "rejected", status already set to REJECTED)
                 |                           |
                 v                           v
            apply_patch                     END
                 |
                END

Reuses Milestone 6's investigation graph wholesale (via investigate_bug_node)
rather than duplicating it, and the structured-output node pattern
established there for generate_proposal/validate_proposal/apply_patch.

A checkpointer is REQUIRED here, not optional: request_human_approval calls
langgraph.types.interrupt(), which needs a checkpointer to persist the paused
state for later resumption via Command(resume=...). There is no meaningful
"ephemeral" mode for a workflow whose entire point is pausing for a human
decision — see app/api/routes/proposals.py, which returns 503 if MongoDB
isn't configured before ever building this graph.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes.apply_patch import apply_patch_node
from app.agents.nodes.generate_proposal import generate_proposal_node
from app.agents.nodes.investigate_bug import investigate_bug_node
from app.agents.nodes.request_human_approval import request_human_approval_node
from app.agents.nodes.validate_proposal import validate_proposal_node
from app.agents.proposal_state import ProposalState
from app.ingestion.store import RepositoryStore

DEFAULT_INVESTIGATION_MAX_ROUNDS = 3
DEFAULT_INVESTIGATION_RECURSION_LIMIT = 60


def build_proposal_graph(
    chat_model: BaseChatModel,
    tools: list[BaseTool],
    store: RepositoryStore,
    checkpointer: BaseCheckpointSaver,
    investigation_max_rounds: int = DEFAULT_INVESTIGATION_MAX_ROUNDS,
    investigation_recursion_limit: int = DEFAULT_INVESTIGATION_RECURSION_LIMIT,
) -> CompiledStateGraph:
    graph = StateGraph(ProposalState)

    async def _investigate_bug(state: ProposalState) -> dict:
        return await investigate_bug_node(
            state,
            chat_model=chat_model,
            tools=tools,
            store=store,
            max_investigation_rounds=investigation_max_rounds,
            recursion_limit=investigation_recursion_limit,
        )

    async def _generate_proposal(state: ProposalState) -> dict:
        return await generate_proposal_node(state, chat_model=chat_model, store=store)

    def _validate_proposal(state: ProposalState) -> dict:
        return validate_proposal_node(state, store=store)

    def _apply_patch(state: ProposalState) -> dict:
        return apply_patch_node(state, store=store)

    graph.add_node("investigate_bug", _investigate_bug)
    graph.add_node("generate_proposal", _generate_proposal)
    graph.add_node("validate_proposal", _validate_proposal)
    graph.add_node("request_human_approval", request_human_approval_node)
    graph.add_node("apply_patch", _apply_patch)

    graph.add_edge(START, "investigate_bug")
    graph.add_conditional_edges(
        "investigate_bug", _route_after_investigation, {"generate_proposal": "generate_proposal", END: END}
    )
    graph.add_edge("generate_proposal", "validate_proposal")
    graph.add_conditional_edges(
        "validate_proposal", _route_after_validation, {"request_human_approval": "request_human_approval", END: END}
    )
    graph.add_conditional_edges(
        "request_human_approval", _route_after_decision, {"apply_patch": "apply_patch", END: END}
    )
    graph.add_edge("apply_patch", END)

    return graph.compile(checkpointer=checkpointer)


def _route_after_investigation(state: ProposalState) -> str:
    return END if state["status"] == "FAILED" else "generate_proposal"


def _route_after_validation(state: ProposalState) -> str:
    return END if state["status"] == "FAILED" else "request_human_approval"


def _route_after_decision(state: ProposalState) -> str:
    return "apply_patch" if state["decision"] == "approved" else END
