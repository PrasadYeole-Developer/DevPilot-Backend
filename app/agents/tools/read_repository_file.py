"""Tool: read_repository_file — reads one file's content from the in-memory repository store."""

from langchain_core.tools import BaseTool, tool

from app.ingestion.store import RepositoryStore

MAX_CONTENT_CHARS = 20_000


def make_read_repository_file_tool(store: RepositoryStore) -> BaseTool:
    @tool
    def read_repository_file(file_path: str) -> str:
        """Read the full content of one file from the currently ingested repository.

        file_path must exactly match a path returned by list_repository_files or
        by search_codebase results. Paths outside the ingested repository —
        including any path-traversal attempt (e.g. containing "..") or an
        absolute path — are rejected, since this only ever looks up the exact
        path against the known set of files actually retrieved during ingestion
        (it never touches the filesystem).

        Args:
            file_path: The exact repository-relative path of the file to read.
        """
        repository = store.get_current()
        if repository is None:
            raise ValueError(
                "No repository has been ingested yet. Call POST /api/v1/repositories/ingest first."
            )

        requested = file_path.strip()
        files_by_path = {file.path: file for file in repository.files}
        matched = files_by_path.get(requested)

        if matched is None:
            raise ValueError(
                f"File '{requested}' was not found in the ingested repository "
                f"{repository.owner}/{repository.repo}. "
                "Use list_repository_files to see available paths."
            )

        content = matched.content
        truncated = len(content) > MAX_CONTENT_CHARS
        if truncated:
            content = content[:MAX_CONTENT_CHARS]

        header = f"File: {matched.path} ({matched.size} bytes)"
        if truncated:
            header += " [content truncated]"
        return f"{header}\n\n{content}"

    return read_repository_file
