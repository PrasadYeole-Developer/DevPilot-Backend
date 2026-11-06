"""Read-only tools the DevPilot agent can call against the currently ingested repository."""

from langchain_core.tools import BaseTool

from app.agents.tools.list_repository_files import make_list_repository_files_tool
from app.agents.tools.read_repository_file import make_read_repository_file_tool
from app.agents.tools.search_codebase import make_search_codebase_tool
from app.ingestion.store import RepositoryStore
from app.retrieval.search_service import RepositorySearchService


def build_tools(
    search_service: RepositorySearchService, store: RepositoryStore
) -> list[BaseTool]:
    return [
        make_search_codebase_tool(search_service),
        make_list_repository_files_tool(store),
        make_read_repository_file_tool(store),
    ]
