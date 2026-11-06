"""Request/response schemas for conversation management (Milestone 7)."""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

ConversationWorkflow = Literal["chat", "investigation", "proposal"]


class ConversationCreateRequest(BaseModel):
    # If omitted, the currently ingested repository is used.
    repository_id: Optional[str] = Field(default=None)
    workflow: ConversationWorkflow = Field(default="chat")
    title: Optional[str] = Field(default=None, max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationSummary(BaseModel):
    thread_id: str
    repository_id: str
    workflow: ConversationWorkflow
    title: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]
    total: int
    limit: int
    offset: int


class ConversationMessage(BaseModel):
    """A simplified view of one LangGraph message — never the raw checkpoint document."""

    role: Literal["human", "ai", "tool", "system"]
    content: str
    tool_name: Optional[str] = None


class ConversationDetail(ConversationSummary):
    messages: list[ConversationMessage] = Field(default_factory=list)
