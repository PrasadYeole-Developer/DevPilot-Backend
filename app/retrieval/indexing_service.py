"""Orchestrates the index pipeline: ingestion store -> chunk -> embed -> Pinecone upsert."""

import hashlib
from dataclasses import dataclass

from app.ingestion.store import IngestedRepository, RepositoryStore
from app.retrieval.chunking import CodeChunk, chunk_files
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError
from app.services.pinecone_service import PineconeService, PineconeServiceError

# Pinecone caps total metadata per vector at 40KB. The chunk text is by far the
# largest field, so it alone is truncated defensively if a misconfigured
# CHUNK_SIZE (or an unusually dense file) would otherwise push a vector over
# the limit.
MAX_METADATA_TEXT_BYTES = 35_000


class NoRepositoryIngestedError(Exception):
    """Raised when /index or /search is called before /ingest."""


@dataclass
class IndexingResult:
    repository_id: str
    total_source_files: int
    total_chunks: int
    embedded_chunks: int
    indexed_vectors: int
    failed_chunks: int
    errors: list[str]


class RepositoryIndexingService:
    def __init__(
        self,
        store: RepositoryStore,
        embedding_service: EmbeddingService,
        pinecone_service: PineconeService,
        chunk_size: int,
        chunk_overlap: int,
        embedding_batch_size: int = 32,
    ):
        self._store = store
        self._embedding_service = embedding_service
        self._pinecone_service = pinecone_service
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._embedding_batch_size = embedding_batch_size

    async def index_current_repository(self) -> IndexingResult:
        repository = self._require_current_repository()
        repository_id = f"{repository.owner}/{repository.repo}"

        chunks = chunk_files(repository.files, self._chunk_size, self._chunk_overlap)
        if not chunks:
            return IndexingResult(
                repository_id=repository_id,
                total_source_files=len(repository.files),
                total_chunks=0,
                embedded_chunks=0,
                indexed_vectors=0,
                failed_chunks=0,
                errors=["No eligible source content to index."],
            )

        embedded_chunks, embedded_vectors, failed_chunks, errors = await self._embed_all(chunks)

        if not embedded_chunks:
            return IndexingResult(
                repository_id=repository_id,
                total_source_files=len(repository.files),
                total_chunks=len(chunks),
                embedded_chunks=0,
                indexed_vectors=0,
                failed_chunks=failed_chunks,
                errors=errors or ["No chunks were successfully embedded."],
            )

        dimension = self._embedding_service.dimension
        assert dimension is not None  # guaranteed once at least one embed call succeeded

        try:
            host = await self._pinecone_service.ensure_index(dimension)
            pinecone_vectors = [
                _build_vector(repository, chunk, vector)
                for chunk, vector in zip(embedded_chunks, embedded_vectors)
            ]
            indexed_count = await self._pinecone_service.upsert(host, pinecone_vectors)
        except PineconeServiceError as exc:
            errors.append(str(exc))
            return IndexingResult(
                repository_id=repository_id,
                total_source_files=len(repository.files),
                total_chunks=len(chunks),
                embedded_chunks=len(embedded_chunks),
                indexed_vectors=0,
                failed_chunks=failed_chunks + len(embedded_chunks),
                errors=errors,
            )

        return IndexingResult(
            repository_id=repository_id,
            total_source_files=len(repository.files),
            total_chunks=len(chunks),
            embedded_chunks=len(embedded_chunks),
            indexed_vectors=indexed_count,
            failed_chunks=failed_chunks,
            errors=errors,
        )

    async def _embed_all(
        self, chunks: list[CodeChunk]
    ) -> tuple[list[CodeChunk], list[list[float]], int, list[str]]:
        successful_chunks: list[CodeChunk] = []
        vectors: list[list[float]] = []
        failed_chunks = 0
        errors: list[str] = []

        for start in range(0, len(chunks), self._embedding_batch_size):
            batch = chunks[start : start + self._embedding_batch_size]
            try:
                batch_vectors = await self._embedding_service.embed_documents(
                    [chunk.text for chunk in batch]
                )
            except EmbeddingServiceError as exc:
                failed_chunks += len(batch)
                errors.append(f"Embedding failed for a batch of {len(batch)} chunks: {exc}")
                continue
            successful_chunks.extend(batch)
            vectors.extend(batch_vectors)

        return successful_chunks, vectors, failed_chunks, errors

    def _require_current_repository(self) -> IngestedRepository:
        repository = self._store.get_current()
        if repository is None:
            raise NoRepositoryIngestedError(
                "No repository has been ingested yet. Call POST /api/v1/repositories/ingest first."
            )
        return repository


def vector_id(repository: IngestedRepository, chunk: CodeChunk) -> str:
    """
    Deterministic vector ID from (owner, repo, branch, path, chunk_index) so
    re-indexing the same repository upserts (overwrites) existing vectors
    instead of creating duplicates.
    """
    raw = f"{repository.owner}/{repository.repo}/{repository.default_branch}/{chunk.path}#{chunk.chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_vector(repository: IngestedRepository, chunk: CodeChunk, vector: list[float]) -> dict:
    text = chunk.text
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_METADATA_TEXT_BYTES:
        text = encoded[:MAX_METADATA_TEXT_BYTES].decode("utf-8", errors="ignore")

    return {
        "id": vector_id(repository, chunk),
        "values": vector,
        "metadata": {
            "repository_id": f"{repository.owner}/{repository.repo}",
            "owner": repository.owner,
            "repository": repository.repo,
            "branch": repository.default_branch,
            "path": chunk.path,
            "chunk_index": chunk.chunk_index,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "text": text,
        },
    }
