"""
Builds the DevPilot tool-calling agent graph.

The Milestone 4 fixed two-node pipeline (retrieve_context -> generate_answer)
is replaced by a ReAct-style loop:

    START -> agent -> (tools_condition) -> tools -> agent -> ... -> END

The agent node calls Gemini with tools bound; if the response contains tool
calls, ToolNode executes them (search_codebase, list_repository_files,
read_repository_file — all read-only, scoped to the currently ingested
repository) and loops back to the agent node with the results appended to the
conversation. Once Gemini responds with no further tool calls, the graph
ends. The caller is responsible for passing a recursion_limit on invoke to
guard against runaway tool-call loops (see app.core.config.agent_recursion_limit).

Dependencies (chat_model, tools) are injected by the caller rather than
constructed inside node functions, consistent with the rest of the agent
module — this keeps nodes testable and avoids constructing new clients per
node execution.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agents.nodes.agent_node import SYSTEM_PROMPT, agent_node
from app.agents.state import AgentState


def build_agent_graph(
    chat_model: BaseChatModel,
    tools: list[BaseTool],
    checkpointer: BaseCheckpointSaver | None = None,
    extra_context: str | None = None,
) -> CompiledStateGraph:
    """
    checkpointer: Milestone 7 addition. None (the default) preserves exact
    Milestone 5 behavior — an uncompiled-with-memory, single-shot graph. When
    provided (a MongoDBSaver in production), LangGraph persists/restores
    `messages` per `thread_id` passed via `config={"configurable": {...}}`
    on invoke.

    extra_context: Milestone 7 addition. Appended to the default system
    prompt — used to inject relevant long-term memories for this repository
    without changing agent_node itself or losing its base behavioral rules.
    """
    bound_model = chat_model.bind_tools(tools)
    system_prompt = SYSTEM_PROMPT if not extra_context else f"{SYSTEM_PROMPT}\n\n{extra_context}"

    graph = StateGraph(AgentState)

    async def _agent(state: AgentState) -> dict:
        return await agent_node(state, chat_model=bound_model, system_prompt=system_prompt)

    graph.add_node("agent", _agent)
    # handle_tool_errors=True: any exception raised by a tool (bad args, path
    # not found, upstream service failure) becomes a ToolMessage with
    # status="error" instead of crashing the graph — Gemini sees the failure
    # explicitly rather than it being silently treated as a successful result.
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)
