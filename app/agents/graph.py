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
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agents.nodes.agent_node import agent_node
from app.agents.state import AgentState


def build_agent_graph(chat_model: BaseChatModel, tools: list[BaseTool]) -> CompiledStateGraph:
    bound_model = chat_model.bind_tools(tools)

    graph = StateGraph(AgentState)

    async def _agent(state: AgentState) -> dict:
        return await agent_node(state, chat_model=bound_model)

    graph.add_node("agent", _agent)
    # handle_tool_errors=True: any exception raised by a tool (bad args, path
    # not found, upstream service failure) becomes a ToolMessage with
    # status="error" instead of crashing the graph — Gemini sees the failure
    # explicitly rather than it being silently treated as a successful result.
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()
