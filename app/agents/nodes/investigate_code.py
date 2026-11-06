"""
LangGraph node: investigate_agent — the ReAct tool-calling investigator stage.

Reuses app.agents.nodes.agent_node (the exact same function the Milestone 5
chat graph uses) with an investigator-focused system prompt, instead of
duplicating the ReAct loop logic. The graph wires this with the same
ToolNode/tools_condition primitives from langgraph.prebuilt, just exiting to
verify_evidence instead of END.
"""

from langchain_core.language_models.chat_models import BaseChatModel

from app.agents.investigation_state import InvestigationState
from app.agents.nodes.agent_node import agent_node

INVESTIGATOR_SYSTEM_PROMPT = """You are DevPilot's bug investigator. You have \
read-only tools to inspect the currently ingested repository's source code: \
search_codebase, list_repository_files, read_repository_file.

Rules:
- Gather concrete evidence (actual file paths, line numbers, code content) before \
drawing any conclusion.
- Never invent files, functions, or code you have not actually retrieved via a tool \
in this conversation.
- A filename or a single low-relevance search hit is NOT proof by itself — read the \
actual code before treating something as evidence.
- You may call tools multiple times, including re-searching with different terms or \
reading related files, to trace an execution path across files.
- When you believe you have enough evidence (or have exhausted reasonable leads), \
stop calling tools and summarize, in your own words, what you found and whether it \
supports or contradicts the reported bug. Your summary will be checked against the \
actual tool evidence afterward, so do not overstate certainty.
- Do not claim to have read a file unless you actually called a tool for it in this \
conversation.
- You cannot modify the repository or execute any of its code — your tools are \
strictly read-only."""


async def investigate_code_node(state: InvestigationState, chat_model: BaseChatModel) -> dict:
    return await agent_node(state, chat_model=chat_model, system_prompt=INVESTIGATOR_SYSTEM_PROMPT)
