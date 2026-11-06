"""Tool: list_repository_files — lists eligible file paths from the in-memory repository store."""

from langchain_core.tools import BaseTool, tool

from app.ingestion.store import RepositoryStore

MAX_FILES_RETURNED = 300


def make_list_repository_files_tool(store: RepositoryStore) -> BaseTool:
    @tool
    def list_repository_files(path_prefix: str = "") -> str:
        """List file paths available in the currently ingested repository.

        Returns only file paths, never file contents — call read_repository_file
        afterward to read a specific file. Optionally filter to files under a
        directory prefix, e.g. "src/" or "app/services".

        Args:
            path_prefix: Only return files whose path starts with this prefix.
                Leave empty to list all eligible files.
        """
        repository = store.get_current()
        if repository is None:
            raise ValueError(
                "No repository has been ingested yet. Call POST /api/v1/repositories/ingest first."
            )

        normalized_prefix = path_prefix.strip().lstrip("/")
        paths = sorted(
            file.path for file in repository.files if file.path.startswith(normalized_prefix)
        )

        if not paths:
            if normalized_prefix:
                return f"No files found under prefix '{normalized_prefix}'."
            return "No eligible files were retrieved for this repository."

        truncated = len(paths) > MAX_FILES_RETURNED
        shown = paths[:MAX_FILES_RETURNED]
        header = f"{len(paths)} file(s) found" + (
            f" (showing first {MAX_FILES_RETURNED})" if truncated else ""
        )
        return header + ":\n" + "\n".join(shown)

    return list_repository_files
