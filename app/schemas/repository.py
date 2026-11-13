"""Request/response schemas for the repository ingestion endpoint."""

from pydantic import BaseModel, Field


class IngestRepositoryRequest(BaseModel):
    repository_url: str = Field(
        ...,
        description="GitHub repository URL, e.g. https://github.com/owner/repository",
    )


class SkippedFile(BaseModel):
    path: str
    reason: str


class IngestRepositoryResponse(BaseModel):
    owner: str
    repository: str
    default_branch: str
    total_files_discovered: int
    eligible_files: int
    retrieved_files: int
    skipped_files: list[SkippedFile]
    skipped_files_truncated: bool = Field(
        default=False,
        description="True if skipped_files was capped and more skips exist than are listed.",
    )
    tree_truncated: bool = Field(
        default=False,
        description="True if GitHub truncated the repository tree (repo too large to list fully).",
    )
    errors: list[str] = Field(default_factory=list)


class CurrentRepositoryInfo(IngestRepositoryResponse):
    """Same shape as IngestRepositoryResponse, plus the derived repository_id
    the frontend already uses elsewhere (owner/repository)."""

    repository_id: str


class CurrentRepositoryResponse(BaseModel):
    repository: CurrentRepositoryInfo | None = Field(
        default=None,
        description="The currently ingested repository, or null if none has been ingested "
        "(or the backend process has restarted since the last ingest).",
    )
