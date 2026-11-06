"""Orchestrates a full repository ingestion: validate URL -> fetch tree -> filter -> fetch contents."""

import asyncio
from dataclasses import dataclass

from app.ingestion.file_filter import SkippedEntry, TreeEntry, filter_tree_entries
from app.ingestion.store import IngestedRepository, RepositoryStore, RetrievedFile
from app.ingestion.url_parser import parse_github_url
from app.services.github_client import GitHubClient, GitHubRateLimitError

# Caps how many skipped-file entries are echoed back in the API response,
# independent of how many are actually tracked internally — a repo with
# thousands of excluded files (e.g. a huge node_modules tree) must not
# blow up the response payload.
MAX_SKIPPED_FILES_IN_RESPONSE = 200


@dataclass
class IngestionResult:
    owner: str
    repo: str
    default_branch: str
    total_files_discovered: int
    eligible_files: int
    retrieved_files: int
    skipped_files: list[SkippedEntry]
    skipped_files_truncated: bool
    tree_truncated: bool
    errors: list[str]


class RepositoryIngestionService:
    def __init__(
        self,
        github_client: GitHubClient,
        max_file_size_bytes: int,
        max_files: int,
        max_concurrent_requests: int,
        store: RepositoryStore,
    ):
        self._client = github_client
        self._max_file_size_bytes = max_file_size_bytes
        self._max_files = max_files
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._store = store

    async def ingest(self, repository_url: str) -> IngestionResult:
        owner, repo = parse_github_url(repository_url)

        repo_info = await self._client.get_repository(owner, repo)
        default_branch = repo_info["default_branch"]

        tree = await self._client.get_tree(owner, repo, default_branch)
        tree_truncated = bool(tree.get("truncated", False))

        entries = [
            TreeEntry(path=item["path"], sha=item["sha"], size=item.get("size", 0))
            for item in tree.get("tree", [])
            if item.get("type") == "blob"
        ]
        total_files_discovered = len(entries)

        eligible, skipped = filter_tree_entries(
            entries, self._max_file_size_bytes, self._max_files
        )

        retrieved_files, fetch_skipped, errors = await self._fetch_contents(owner, repo, eligible)
        skipped.extend(fetch_skipped)

        self._store.save(
            IngestedRepository(
                owner=owner,
                repo=repo,
                default_branch=default_branch,
                files=retrieved_files,
            )
        )

        skipped_files_truncated = len(skipped) > MAX_SKIPPED_FILES_IN_RESPONSE

        return IngestionResult(
            owner=owner,
            repo=repo,
            default_branch=default_branch,
            total_files_discovered=total_files_discovered,
            eligible_files=len(eligible),
            retrieved_files=len(retrieved_files),
            skipped_files=skipped[:MAX_SKIPPED_FILES_IN_RESPONSE],
            skipped_files_truncated=skipped_files_truncated,
            tree_truncated=tree_truncated,
            errors=errors,
        )

    async def _fetch_contents(
        self, owner: str, repo: str, eligible: list[TreeEntry]
    ) -> tuple[list[RetrievedFile], list[SkippedEntry], list[str]]:
        async def fetch_one(entry: TreeEntry) -> RetrievedFile | SkippedEntry:
            async with self._semaphore:
                text = await self._client.get_blob_text(owner, repo, entry.sha)
            if text is None:
                return SkippedEntry(path=entry.path, reason="binary_or_undecodable_content")
            return RetrievedFile(path=entry.path, content=text, size=entry.size, sha=entry.sha)

        results = await asyncio.gather(
            *(fetch_one(entry) for entry in eligible), return_exceptions=True
        )

        retrieved: list[RetrievedFile] = []
        skipped: list[SkippedEntry] = []
        errors: list[str] = []
        rate_limited = False

        for entry, result in zip(eligible, results):
            if isinstance(result, RetrievedFile):
                retrieved.append(result)
            elif isinstance(result, SkippedEntry):
                skipped.append(result)
            elif isinstance(result, GitHubRateLimitError):
                skipped.append(SkippedEntry(path=entry.path, reason="rate_limited"))
                rate_limited = True
            elif isinstance(result, Exception):
                skipped.append(SkippedEntry(path=entry.path, reason="fetch_error"))

        if rate_limited:
            errors.append(
                "GitHub API rate limit was reached while fetching file contents; "
                "some eligible files were skipped. Provide a GITHUB_TOKEN to raise the limit."
            )

        return retrieved, skipped, errors
