"""Repository ingestion, indexing, and semantic search endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings, get_settings
from app.ingestion.repository_service import RepositoryIngestionService
from app.ingestion.store import repository_store
from app.ingestion.url_parser import InvalidRepositoryURLError
from app.retrieval.indexing_service import NoRepositoryIngestedError, RepositoryIndexingService
from app.retrieval.search_service import RepositorySearchService
from app.schemas.indexing import IndexRepositoryResponse
from app.schemas.repository import IngestRepositoryRequest, IngestRepositoryResponse, SkippedFile
from app.schemas.search import SearchRequest, SearchResponse, SearchResult
from app.services.embedding_service import (
    EmbeddingRateLimitError,
    EmbeddingService,
    EmbeddingServiceError,
)
from app.services.github_client import GitHubAPIError, GitHubClient, GitHubNotFoundError, GitHubRateLimitError
from app.services.pinecone_service import PineconeService, PineconeServiceError

router = APIRouter(prefix="/api/v1/repositories", tags=["repositories"])


@router.post("/ingest", response_model=IngestRepositoryResponse)
async def ingest_repository(
    payload: IngestRepositoryRequest,
    settings: Settings = Depends(get_settings),
) -> IngestRepositoryResponse:
    try:
        async with GitHubClient(
            token=settings.github_token,
            timeout_seconds=settings.github_request_timeout_seconds,
        ) as client:
            service = RepositoryIngestionService(
                github_client=client,
                max_file_size_bytes=settings.github_max_file_size_bytes,
                max_files=settings.github_max_files,
                max_concurrent_requests=settings.github_max_concurrent_requests,
                store=repository_store,
            )
            result = await service.ingest(payload.repository_url)
    except InvalidRepositoryURLError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GitHubNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="Repository not found (or private without a valid token)."
        ) from exc
    except GitHubRateLimitError as exc:
        raise HTTPException(
            status_code=429, detail="GitHub API rate limit exceeded. Try again later or provide a GITHUB_TOKEN."
        ) from exc
    except GitHubAPIError as exc:
        raise HTTPException(status_code=502, detail=f"GitHub API error: {exc}") from exc

    return IngestRepositoryResponse(
        owner=result.owner,
        repository=result.repo,
        default_branch=result.default_branch,
        total_files_discovered=result.total_files_discovered,
        eligible_files=result.eligible_files,
        retrieved_files=result.retrieved_files,
        skipped_files=[SkippedFile(path=s.path, reason=s.reason) for s in result.skipped_files],
        skipped_files_truncated=result.skipped_files_truncated,
        tree_truncated=result.tree_truncated,
        errors=result.errors,
    )


@router.post("/index", response_model=IndexRepositoryResponse)
async def index_repository(
    settings: Settings = Depends(get_settings),
) -> IndexRepositoryResponse:
    try:
        embedding_service = EmbeddingService(
            api_key=settings.google_api_key,
            model=settings.google_embedding_model,
        )
        pinecone_service = PineconeService(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
    except (EmbeddingServiceError, PineconeServiceError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    service = RepositoryIndexingService(
        store=repository_store,
        embedding_service=embedding_service,
        pinecone_service=pinecone_service,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    try:
        result = await service.index_current_repository()
    except NoRepositoryIngestedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EmbeddingRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except EmbeddingServiceError as exc:
        raise HTTPException(status_code=502, detail=f"Embedding generation failed: {exc}") from exc
    except PineconeServiceError as exc:
        raise HTTPException(status_code=502, detail=f"Pinecone error: {exc}") from exc

    return IndexRepositoryResponse(
        repository_id=result.repository_id,
        total_source_files=result.total_source_files,
        total_chunks=result.total_chunks,
        embedded_chunks=result.embedded_chunks,
        indexed_vectors=result.indexed_vectors,
        failed_chunks=result.failed_chunks,
        errors=result.errors,
    )


@router.post("/search", response_model=SearchResponse)
async def search_repository(
    payload: SearchRequest,
    settings: Settings = Depends(get_settings),
) -> SearchResponse:
    repository = repository_store.get_current()
    if repository is None:
        raise HTTPException(
            status_code=409,
            detail="No repository has been ingested yet. Call POST /api/v1/repositories/ingest first.",
        )

    try:
        embedding_service = EmbeddingService(
            api_key=settings.google_api_key,
            model=settings.google_embedding_model,
        )
        pinecone_service = PineconeService(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        )
    except (EmbeddingServiceError, PineconeServiceError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    service = RepositorySearchService(
        store=repository_store,
        embedding_service=embedding_service,
        pinecone_service=pinecone_service,
    )

    try:
        matches = await service.search(payload.query, payload.top_k)
    except NoRepositoryIngestedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except EmbeddingRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except EmbeddingServiceError as exc:
        raise HTTPException(status_code=502, detail=f"Embedding generation failed: {exc}") from exc
    except PineconeServiceError as exc:
        raise HTTPException(status_code=502, detail=f"Pinecone error: {exc}") from exc

    return SearchResponse(
        repository_id=f"{repository.owner}/{repository.repo}",
        query=payload.query,
        results=[
            SearchResult(
                path=m.path,
                chunk_index=m.chunk_index,
                start_line=m.start_line,
                end_line=m.end_line,
                score=m.score,
                text=m.text,
            )
            for m in matches
        ],
    )
