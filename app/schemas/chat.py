"""Request/response schemas for the chat endpoint."""

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    # Milestone 7, additive: omit for the original Milestone 5 ephemeral behavior
    # (no persistence, no ownership check). Pass a thread_id obtained from
    # POST /api/v1/conversations to continue a persisted conversation.
    thread_id: Optional[str] = Field(default=None)


class ChatSource(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    similarity_score: float


class ToolCallRecord(BaseModel):
    """One tool call the agent made while answering, for transparency/debugging."""

    tool: str
    arguments: dict
    status: str = Field(description="'success' or 'error', as reported by the tool execution.")


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    # Additive, defaults to [] — existing clients that only read answer/sources
    # are unaffected.
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    # Milestone 7, additive: echoes the thread_id used, or None if this call
    # was ephemeral (no thread_id supplied / Mongo not configured).
    thread_id: Optional[str] = Field(default=None)
