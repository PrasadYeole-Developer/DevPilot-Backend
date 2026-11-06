"""Request/response schemas for the chat endpoint."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)


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
