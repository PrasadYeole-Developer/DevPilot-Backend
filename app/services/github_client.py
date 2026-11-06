"""
Minimal async GitHub REST API client.

Only the read-only endpoints DevPilot needs are implemented:
repository metadata, the recursive git tree, and blob content.
No write/mutating calls exist here by design.
"""

from __future__ import annotations

import base64
from typing import Any, Optional

import httpx

GITHUB_API_BASE_URL = "https://api.github.com"


class GitHubAPIError(Exception):
    """Base error for any failed GitHub API call."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class GitHubNotFoundError(GitHubAPIError):
    """Repository, branch, or resource does not exist (or is private without a token)."""


class GitHubRateLimitError(GitHubAPIError):
    """GitHub rate limit was hit. reset_at_epoch is the Unix timestamp when it resets, if known."""

    def __init__(self, message: str, reset_at_epoch: Optional[int] = None):
        super().__init__(message, status_code=403)
        self.reset_at_epoch = reset_at_epoch


class GitHubClient:
    """Thin async wrapper around the GitHub REST API. Use as an async context manager."""

    def __init__(self, token: Optional[str], timeout_seconds: float):
        self._token = token
        self._timeout_seconds = timeout_seconds
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "GitHubClient":
        self._client = httpx.AsyncClient(
            base_url=GITHUB_API_BASE_URL,
            headers=self._build_headers(),
            timeout=self._timeout_seconds,
        )
        return self

    async def __aexit__(self, *_exc_info) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        # Never log this header — it carries the caller's token when present.
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        assert self._client is not None, "GitHubClient must be used as an async context manager"
        try:
            response = await self._client.get(path, params=params)
        except httpx.TimeoutException as exc:
            raise GitHubAPIError(f"GitHub API request timed out: {path}") from exc
        except httpx.HTTPError as exc:
            raise GitHubAPIError(f"GitHub API request failed: {path} ({exc})") from exc

        if response.status_code == 404:
            raise GitHubNotFoundError(f"Not found: {path}", status_code=404)

        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            reset_header = response.headers.get("X-RateLimit-Reset")
            reset_at = int(reset_header) if reset_header else None
            raise GitHubRateLimitError("GitHub API rate limit exceeded", reset_at_epoch=reset_at)

        if response.status_code >= 400:
            raise GitHubAPIError(
                f"GitHub API error {response.status_code} for {path}",
                status_code=response.status_code,
            )

        return response.json()

    async def get_repository(self, owner: str, repo: str) -> dict[str, Any]:
        """Fetch repository metadata (used to resolve the default branch)."""
        return await self._get(f"/repos/{owner}/{repo}")

    async def get_tree(self, owner: str, repo: str, tree_sha: str) -> dict[str, Any]:
        """Fetch the recursive git tree for a branch name or commit/tree SHA."""
        return await self._get(
            f"/repos/{owner}/{repo}/git/trees/{tree_sha}",
            params={"recursive": "1"},
        )

    async def get_blob_text(self, owner: str, repo: str, sha: str) -> Optional[str]:
        """
        Fetch a blob's content and decode it as UTF-8 text.

        Returns None if the blob is not valid UTF-8 text (treated as binary)
        or if GitHub reports the content as unavailable/truncated.
        """
        blob = await self._get(f"/repos/{owner}/{repo}/git/blobs/{sha}")

        if blob.get("encoding") != "base64" or blob.get("content") is None:
            return None

        try:
            raw_bytes = base64.b64decode(blob["content"])
        except (ValueError, TypeError):
            return None

        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return None
