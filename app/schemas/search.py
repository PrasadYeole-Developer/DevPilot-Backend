"""Request/response schemas for the semantic search endpoint."""

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    path: str
    chunk_index: int
    start_line: int
    end_line: int
    score: float
    text: str


class SearchResponse(BaseModel):
    repository_id: str
    query: str
    results: list[SearchResult]
