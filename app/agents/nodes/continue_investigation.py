"""
LangGraph node: continue_investigation — bridges verify_evidence back into the
ReAct loop with concrete direction on what's still missing, when evidence was
judged insufficient but rounds remain.
"""

from langchain_core.messages import HumanMessage

from app.agents.investigation_state import InvestigationState


def continue_investigation_node(state: InvestigationState) -> dict:
    assessment = state["evidence_assessment"]
    missing = assessment.missing_evidence if assessment else []

    if missing:
        guidance = (
            "The evidence gathered so far is not sufficient yet. Specifically still "
            "missing:\n" + "\n".join(f"- {item}" for item in missing) + "\n\n"
            "Use your tools to address these gaps before summarizing again."
        )
    else:
        guidance = (
            "The evidence gathered so far is not sufficient yet. Try different search "
            "terms, or read files you haven't opened yet, before summarizing again."
        )

    return {
        "investigation_round": state["investigation_round"] + 1,
        "messages": [HumanMessage(content=guidance)],
    }
