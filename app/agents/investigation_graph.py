"""
Builds the DevPilot bug-investigation graph (Milestone 6).

    START -> understand_bug -> plan_investigation -> investigate_agent
                                                         |  ^
                                           (tool calls)  v  | (loop)
                                                    investigate_tools
                                                         |
                                           (no more tool calls)
                                                         v
                                                  verify_evidence
                                                         |
                                   +---------------------+----------------------+
                                   | sufficient           | insufficient,        | insufficient,
                                   v                       | rounds remain       | rounds exhausted
                         analyze_root_cause                v                     v
                                   |              continue_investigation   generate_report
                                   v                       |
                           generate_report -> END   investigate_agent (loop)

The investigate_agent/investigate_tools pair reuses the exact same
agent_node/ToolNode/tools_condition primitives as the Milestone 5 chat graph
(app.agents.graph.build_agent_graph) — just with an investigator system prompt
and a different exit destination (verify_evidence instead of END).

Dependencies (chat_model, tools, store) are injected by the caller, consistent
with the rest of the agent module.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agents.investigation_state import InvestigationState
from app.agents.nodes.analyze_root_cause import analyze_root_cause_node
from app.agents.nodes.continue_investigation import continue_investigation_node
from app.agents.nodes.generate_report import generate_report_node
from app.agents.nodes.investigate_code import investigate_code_node
from app.agents.nodes.plan_investigation import plan_investigation_node
from app.agents.nodes.understand_bug import understand_bug_node
from app.agents.nodes.verify_evidence import verify_evidence_node
from app.ingestion.store import RepositoryStore

DEFAULT_MAX_INVESTIGATION_ROUNDS = 3


def build_investigation_graph(
    chat_model: BaseChatModel,
    tools: list[BaseTool],
    store: RepositoryStore,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """
    checkpointer: Milestone 7 addition. None (the default) preserves exact
    Milestone 6 behavior. When provided, the full investigation conversation
    (understand_bug through generate_report) is persisted per `thread_id` for
    later history retrieval via GET /api/v1/conversations/{thread_id}.

    Note: unlike the chat graph, a checkpointed investigation thread is never
    resumed mid-workflow by this graph itself — app.api.routes.investigate
    always clears any prior checkpoint for a reused thread_id before invoking,
    since this workflow has no valid "continue where it left off" state (it
    runs start-to-finish in one call). The checkpointer here exists purely so
    a finished investigation's full message trace can be reviewed later.
    """
    bound_model = chat_model.bind_tools(tools)

    graph = StateGraph(InvestigationState)

    async def _understand_bug(state: InvestigationState) -> dict:
        return await understand_bug_node(state, chat_model=chat_model)

    async def _plan_investigation(state: InvestigationState) -> dict:
        return await plan_investigation_node(state, chat_model=chat_model)

    async def _investigate_agent(state: InvestigationState) -> dict:
        return await investigate_code_node(state, chat_model=bound_model)

    async def _verify_evidence(state: InvestigationState) -> dict:
        return await verify_evidence_node(state, chat_model=chat_model, store=store)

    async def _analyze_root_cause(state: InvestigationState) -> dict:
        return await analyze_root_cause_node(state, chat_model=chat_model)

    graph.add_node("understand_bug", _understand_bug)
    graph.add_node("plan_investigation", _plan_investigation)
    graph.add_node("investigate_agent", _investigate_agent)
    # handle_tool_errors=True: same reasoning as Milestone 5 — a tool failure
    # becomes a visible status="error" ToolMessage, never a silent fake success.
    graph.add_node("investigate_tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_node("verify_evidence", _verify_evidence)
    graph.add_node("continue_investigation", continue_investigation_node)
    graph.add_node("analyze_root_cause", _analyze_root_cause)
    graph.add_node("generate_report", generate_report_node)

    graph.add_edge(START, "understand_bug")
    graph.add_edge("understand_bug", "plan_investigation")
    graph.add_edge("plan_investigation", "investigate_agent")

    graph.add_conditional_edges(
        "investigate_agent",
        tools_condition,
        {"tools": "investigate_tools", END: "verify_evidence"},
    )
    graph.add_edge("investigate_tools", "investigate_agent")

    graph.add_conditional_edges(
        "verify_evidence",
        _route_after_verification,
        {
            "analyze_root_cause": "analyze_root_cause",
            "continue_investigation": "continue_investigation",
            "generate_report": "generate_report",
        },
    )
    graph.add_edge("continue_investigation", "investigate_agent")
    graph.add_edge("analyze_root_cause", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile(checkpointer=checkpointer)


def _route_after_verification(state: InvestigationState) -> str:
    assessment = state["evidence_assessment"]
    if assessment.sufficient:
        return "analyze_root_cause"
    max_rounds = state.get("max_investigation_rounds", DEFAULT_MAX_INVESTIGATION_ROUNDS)
    if state["investigation_round"] >= max_rounds:
        return "generate_report"
    return "continue_investigation"
