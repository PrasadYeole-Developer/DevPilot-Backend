"""
Thin async wrapper around Pinecone for index lifecycle, vector upsert, and similarity search.

Uses the native async client (PineconeAsyncio / AsyncIndex) since the rest of
the app is async FastAPI — no thread-pool shims needed. Index management
(create/describe) opens a short-lived PineconeAsyncio client; data operations
(upsert/query) open a short-lived AsyncIndex bound to the index's host, which
Pinecone requires looking up separately from the index name.
"""

from typing import Any

from pinecone import AsyncIndex, PineconeAsyncio, ServerlessSpec
from pinecone.exceptions import PineconeException


class PineconeServiceError(Exception):
    """Base error for any failed Pinecone operation."""


class PineconeService:
    def __init__(self, api_key: str | None, index_name: str, cloud: str, region: str):
        if not api_key:
            raise PineconeServiceError(
                "PINECONE_API_KEY is not configured. Set it in your .env file."
            )
        self._api_key = api_key
        self._index_name = index_name
        self._cloud = cloud
        self._region = region

    async def ensure_index(self, dimension: int) -> str:
        """
        Create the index if it doesn't exist yet (serverless, cosine similarity).
        If it already exists, verify its dimension matches the embedding model's
        actual output — a silent mismatch would otherwise fail at upsert time
        with a much less obvious error. Returns the index host.
        """
        try:
            async with PineconeAsyncio(api_key=self._api_key) as pc:
                if not await pc.has_index(self._index_name):
                    await pc.create_index(
                        name=self._index_name,
                        dimension=dimension,
                        metric="cosine",
                        spec=ServerlessSpec(cloud=self._cloud, region=self._region),
                    )
                description = await pc.describe_index(self._index_name)
        except PineconeException as exc:
            raise PineconeServiceError(f"Failed to create/describe Pinecone index: {exc}") from exc

        if description.dimension != dimension:
            raise PineconeServiceError(
                f"Pinecone index '{self._index_name}' already exists with dimension "
                f"{description.dimension}, but the embedding model produced dimension "
                f"{dimension}. Use a different PINECONE_INDEX_NAME or delete the existing index."
            )

        return description.host

    async def get_index_host(self) -> str:
        """Look up the host of an existing index, without creating one."""
        try:
            async with PineconeAsyncio(api_key=self._api_key) as pc:
                if not await pc.has_index(self._index_name):
                    raise PineconeServiceError(
                        f"Pinecone index '{self._index_name}' does not exist yet. "
                        "Run POST /api/v1/repositories/index first."
                    )
                description = await pc.describe_index(self._index_name)
        except PineconeException as exc:
            raise PineconeServiceError(f"Failed to describe Pinecone index: {exc}") from exc
        return description.host

    async def upsert(self, host: str, vectors: list[dict[str, Any]]) -> int:
        try:
            async with AsyncIndex(api_key=self._api_key, host=host) as index:
                response = await index.upsert(vectors=vectors, batch_size=100)
        except PineconeException as exc:
            raise PineconeServiceError(f"Pinecone upsert failed: {exc}") from exc
        return response.upserted_count

    async def query(
        self,
        host: str,
        vector: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        try:
            async with AsyncIndex(api_key=self._api_key, host=host) as index:
                response = await index.query(
                    vector=vector,
                    top_k=top_k,
                    filter=metadata_filter,
                    include_metadata=True,
                )
        except PineconeException as exc:
            raise PineconeServiceError(f"Pinecone query failed: {exc}") from exc
        return response.matches
