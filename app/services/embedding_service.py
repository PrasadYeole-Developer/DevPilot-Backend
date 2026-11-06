"""
Generates embeddings via Google's GenAI SDK (google-genai).

Model: gemini-embedding-001, the currently supported Gemini embedding model
(confirmed against the installed google-genai==3.0.0 SDK's own test fixtures —
there is no stable, documented way to list supported embedding models at
runtime). Output dimensionality is never assumed: it is read off the first
real API response and exposed via `.dimension`, so callers (Pinecone index
creation) use the actual value instead of a hardcoded guess.

Document chunks and search queries are embedded with different task_type
values (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY) — Gemini's embedding model is
asymmetric and tuned for this, which measurably improves retrieval quality
over embedding both sides the same way.
"""

from google import genai
from google.genai import errors as genai_errors


class EmbeddingServiceError(Exception):
    """Base error for any failed embedding request."""


class EmbeddingRateLimitError(EmbeddingServiceError):
    """Raised when the Gemini API reports a rate limit (HTTP 429)."""


class EmbeddingService:
    def __init__(self, api_key: str | None, model: str, batch_size: int = 32):
        if not api_key:
            raise EmbeddingServiceError(
                "GOOGLE_API_KEY is not configured. Set it in your .env file."
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._batch_size = batch_size
        self._dimension: int | None = None

    @property
    def dimension(self) -> int | None:
        """Actual output dimension of the last successful embedding call, or None if none yet."""
        return self._dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([text], task_type="RETRIEVAL_QUERY")
        return vectors[0]

    async def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                response = await self._client.aio.models.embed_content(
                    model=self._model,
                    contents=batch,
                    config={"task_type": task_type},
                )
            except genai_errors.APIError as exc:
                if exc.code == 429:
                    raise EmbeddingRateLimitError(
                        f"Gemini embedding rate limit exceeded: {exc}"
                    ) from exc
                raise EmbeddingServiceError(f"Gemini embedding request failed: {exc}") from exc

            if response.embeddings is None:
                raise EmbeddingServiceError("Gemini embedding response contained no embeddings.")

            for item in response.embeddings:
                vector = list(item.values or [])
                if self._dimension is None:
                    self._dimension = len(vector)
                all_vectors.append(vector)

        return all_vectors
