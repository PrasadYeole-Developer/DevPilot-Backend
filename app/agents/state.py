"""Shared state definition for the DevPilot tool-calling agent graph."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class SourceReference(TypedDict):
    """The subset of a retrieved chunk exposed to API callers as a citation."""

    file_path: str
    start_line: int
    end_line: int
    similarity_score: float


class AgentState(TypedDict):
    # `add_messages` appends new messages (and de-dupes by id) instead of
    # overwriting — this is what lets tool calls and tool results accumulate
    # across the agent <-> tools loop instead of being lost between turns.
    messages: Annotated[list[AnyMessage], add_messages]
    repository_id: str
