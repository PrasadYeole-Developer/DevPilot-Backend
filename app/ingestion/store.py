"""
Holds the contents of the most recently ingested repository in memory.

DevPilot connects to one repository at a time, so a single slot (overwritten
on each ingest) is sufficient — there is no multi-repo cache or persistence
here. This is intentionally temporary scaffolding for the next milestone
(chunking/embeddings); it is lost on process restart.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class RetrievedFile:
    path: str
    content: str
    size: int
    sha: str


@dataclass(frozen=True)
class IngestedRepository:
    owner: str
    repo: str
    default_branch: str
    files: list[RetrievedFile] = field(default_factory=list)
    ingested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RepositoryStore:
    def __init__(self) -> None:
        self._current: IngestedRepository | None = None

    def save(self, repository: IngestedRepository) -> None:
        self._current = repository

    def get_current(self) -> IngestedRepository | None:
        return self._current


repository_store = RepositoryStore()
