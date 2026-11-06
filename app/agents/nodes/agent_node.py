"""
LangGraph node: agent.

Invokes Gemini (with tools already bound by the caller) on the running
conversation. The system prompt is prepended fresh on every call rather than
persisted into state — it's static and cheap to rebuild, and this avoids any
risk of it being duplicated as the agent/tools loop appends messages.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from app.agents.state import AgentState

SYSTEM_PROMPT = """You are DevPilot, a software development assistant.

You can inspect the currently ingested GitHub repository using the tools provided:
- search_codebase: semantic search over the repository's indexed source code.
- list_repository_files: list file paths available in the repository.
- read_repository_file: read one file's full content by its exact path.

Rules:
- Use tools whenever answering requires information about the repository's actual
  files or code. Do not answer from assumption when a tool could confirm the facts.
- Never invent files, functions, configuration, or implementation details that you
  have not actually seen in a tool result in this conversation.
- Do not claim to have read a file unless you actually called read_repository_file
  (or saw its content via search_codebase) in this conversation.
- Cite the relevant file path (and line numbers, when available) for any claim drawn
  from a tool result.
- If the available evidence is insufficient to answer confidently, say so plainly
  instead of guessing.
- You may call tools multiple times in sequence (e.g. search, then read a specific
  file) when one call isn't enough to answer well.
- You cannot modify the repository or execute any of its code — your tools are
  strictly read-only."""


async def agent_node(
    state: AgentState, chat_model: BaseChatModel, system_prompt: str = SYSTEM_PROMPT
) -> dict:
    """
    system_prompt defaults to the Milestone 5 chat assistant prompt, so existing
    callers (app.agents.graph.build_agent_graph) are unaffected. The Milestone 6
    investigation graph reuses this same function with an investigator-focused
    prompt instead of duplicating the ReAct loop logic.
    """
    messages = [SystemMessage(content=system_prompt), *state["messages"]]
    response = await chat_model.ainvoke(messages)
    return {"messages": [response]}
