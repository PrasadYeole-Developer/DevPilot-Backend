"""Orchestrates semantic search: embed the query -> Pinecone similarity search -> map to results."""

from dataclasses import dataclass
from typing import Any

from app.ingestion.store import RepositoryStore
from app.retrieval.indexing_service import NoRepositoryIngestedError
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService


@dataclass
class SearchMatch:
    path: str
    chunk_index: int
    start_line: int
    end_line: int
    score: float
    text: str


class RepositorySearchService:
    def __init__(
        self,
        store: RepositoryStore,
        embedding_service: EmbeddingService,
        pinecone_service: PineconeService,
    ):
        self._store = store
        self._embedding_service = embedding_service
        self._pinecone_service = pinecone_service

    async def search(self, query: str, top_k: int) -> list[SearchMatch]:
        repository = self._store.get_current()
        if repository is None:
            raise NoRepositoryIngestedError(
                "No repository has been ingested yet. Call POST /api/v1/repositories/ingest first."
            )
        repository_id = f"{repository.owner}/{repository.repo}"

        query_vector = await self._embedding_service.embed_query(query)
        host = await self._pinecone_service.get_index_host()

        matches = await self._pinecone_service.query(
            host=host,
            vector=query_vector,
            top_k=top_k,
            metadata_filter={"repository_id": {"$eq": repository_id}},
        )

        return [_to_search_match(match) for match in matches]


def _to_search_match(match: Any) -> SearchMatch:
    metadata = match.metadata or {}
    return SearchMatch(
        path=metadata.get("path", "unknown"),
        chunk_index=metadata.get("chunk_index", -1),
        start_line=metadata.get("start_line", -1),
        end_line=metadata.get("end_line", -1),
        score=match.score,
        text=metadata.get("text", ""),
    )
