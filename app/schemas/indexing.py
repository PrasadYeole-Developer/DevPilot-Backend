"""Response schema for the repository indexing endpoint."""

from pydantic import BaseModel, Field


class IndexRepositoryResponse(BaseModel):
    repository_id: str
    total_source_files: int
    total_chunks: int
    embedded_chunks: int
    indexed_vectors: int
    failed_chunks: int
    errors: list[str] = Field(default_factory=list)
