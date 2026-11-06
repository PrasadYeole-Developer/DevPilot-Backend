"""
Tool: search_codebase.

Thin wrapper around the existing RepositorySearchService (Milestone 3) — no
embedding or Pinecone query logic is duplicated here. Uses
response_format="content_and_artifact" so the tool result carries both a
text summary for Gemini and a structured `artifact` (file_path/line
range/similarity_score) that the chat route later collects into the API
response's `sources` list.
"""

from langchain_core.tools import BaseTool, tool

from app.retrieval.indexing_service import NoRepositoryIngestedError
from app.retrieval.search_service import RepositorySearchService
from app.services.embedding_service import EmbeddingRateLimitError, EmbeddingServiceError
from app.services.pinecone_service import PineconeServiceError

DEFAULT_TOP_K = 5
MAX_TOP_K = 15


def make_search_codebase_tool(search_service: RepositorySearchService) -> BaseTool:
    @tool(response_format="content_and_artifact")
    async def search_codebase(query: str, top_k: int = DEFAULT_TOP_K) -> tuple[str, list[dict]]:
        """Semantically search the currently ingested GitHub repository's source code.

        Use this to find code or documentation relevant to a question before
        answering. Returns the most relevant chunks with their file path, line
        range, and similarity score. Call again with a refined query if the
        first results aren't sufficient.

        Args:
            query: Natural-language description of what to look for.
            top_k: Number of results to return (1-15, default 5).
        """
        bounded_top_k = max(1, min(top_k, MAX_TOP_K))
        try:
            matches = await search_service.search(query, bounded_top_k)
        except NoRepositoryIngestedError as exc:
            raise ValueError(str(exc)) from exc
        except EmbeddingRateLimitError as exc:
            raise ValueError(f"Embedding rate limit exceeded: {exc}") from exc
        except EmbeddingServiceError as exc:
            raise ValueError(f"Embedding generation failed: {exc}") from exc
        except PineconeServiceError as exc:
            raise ValueError(f"Pinecone search failed: {exc}") from exc

        if not matches:
            return "No relevant source code was found for this query.", []

        artifact = [
            {
                "file_path": match.path,
                "start_line": match.start_line,
                "end_line": match.end_line,
                "similarity_score": match.score,
            }
            for match in matches
        ]
        content = "\n\n".join(
            f"--- {match.path} (lines {match.start_line}-{match.end_line}, "
            f"similarity {match.score:.3f}) ---\n{match.text}"
            for match in matches
        )
        return content, artifact

    return search_codebase
