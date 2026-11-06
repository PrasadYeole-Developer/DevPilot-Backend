"""LangGraph node: understand_bug — interprets the raw bug report before any code is inspected."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.investigation_state import InvestigationState
from app.schemas.investigation import BugUnderstanding

SYSTEM_PROMPT = """You analyze a user's bug report about a software repository.

Extract:
- The expected behavior (what should happen).
- The actual/reported behavior (what the user says happens instead).
- Keywords worth searching for in the codebase.
- Candidate investigation areas/components (e.g. "authentication", "session
  handling", "user deactivation") that seem relevant to this report.

Base this purely on what the bug report itself says. You have not looked at any
code yet — do not guess at a root cause or assume the bug is real."""


async def understand_bug_node(state: InvestigationState, chat_model: BaseChatModel) -> dict:
    structured_model = chat_model.with_structured_output(BugUnderstanding)
    understanding: BugUnderstanding = await structured_model.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=state["bug_report"])]
    )
    return {"bug_understanding": understanding, "status": "planning"}
