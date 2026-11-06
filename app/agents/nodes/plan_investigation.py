"""LangGraph node: plan_investigation — turns the bug understanding into a focused investigation plan."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.investigation_state import InvestigationState
from app.schemas.investigation import BugUnderstanding, InvestigationPlan

SYSTEM_PROMPT = """You plan a codebase investigation for a reported bug.

Given the expected/actual behavior and candidate areas, propose a short, focused
plan: which parts of the codebase to examine and why, in a sensible order. Do not
propose a root cause yet — only what to look at. Keep it to at most 5 steps."""


async def plan_investigation_node(state: InvestigationState, chat_model: BaseChatModel) -> dict:
    understanding = state["bug_understanding"]
    structured_model = chat_model.with_structured_output(InvestigationPlan)
    plan: InvestigationPlan = await structured_model.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=_format_understanding(understanding))]
    )

    kickoff_message = HumanMessage(
        content=(
            f"Bug report: {state['bug_report']}\n\n"
            f"Expected behavior: {understanding.expected_behavior}\n"
            f"Actual behavior: {understanding.actual_behavior}\n\n"
            f"Investigation plan:\n{_format_plan(plan)}\n\n"
            "Investigate the repository using your tools to gather evidence for or "
            "against this bug. Stop calling tools once you have either found concrete "
            "supporting evidence or exhausted the reasonable leads from the plan."
        )
    )
    return {"investigation_plan": plan, "messages": [kickoff_message], "status": "investigating"}


def _format_understanding(understanding: BugUnderstanding) -> str:
    return (
        f"Expected behavior: {understanding.expected_behavior}\n"
        f"Actual behavior: {understanding.actual_behavior}\n"
        f"Keywords: {', '.join(understanding.keywords) or '(none extracted)'}\n"
        f"Candidate areas: {', '.join(understanding.investigation_areas) or '(none extracted)'}"
    )


def _format_plan(plan: InvestigationPlan) -> str:
    if not plan.steps:
        return plan.summary
    steps = "\n".join(f"{i + 1}. {step.area} — {step.rationale}" for i, step in enumerate(plan.steps))
    return f"{plan.summary}\n{steps}"
