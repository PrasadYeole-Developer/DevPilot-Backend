"""
Async GitHub REST API client for WRITE operations: branches, commits, pull requests.

Deliberately separate from app.services.github_client.GitHubClient, which is
read-only by design (see its module docstring) and is never touched here.
This client requires its own, explicitly write-scoped token
(settings.github_write_token) so a read-only DevPilot deployment can never
accidentally hold a token capable of mutating a repository.

Everything goes through GitHub's REST/Git Data API (blobs/trees/commits/refs,
plus the pulls endpoint) — no local git clone, no shell/subprocess git calls.
This mirrors how app.ingestion already fetches content through the API
rather than cloning.

Retries are deliberately NOT implemented here: branch/ref/commit/PR creation
are not safely idempotent to retry blindly (a retried POST could create a
duplicate commit/PR, or race a ref update). Callers that want retry
semantics must do so explicitly at a layer that can reason about idempotency
(e.g. by checking whether the desired end state already exists).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.services.github_client import (
    GITHUB_API_BASE_URL,
    GitHubAPIError,
    GitHubNotFoundError,
    GitHubRateLimitError,
)

# --- Errors -----------------------------------------------------------------


class GitHubWriteConfigurationError(Exception):
    """Raised when GitHubWriteClient is used without a configured write token.

    Mirrors app.services.embedding_service.EmbeddingServiceError and
    app.services.pinecone_service.PineconeServiceError: a config problem,
    distinct from GitHubAPIError (which means a call to GitHub was actually
    attempted and failed).
    """


class GitHubBranchExistsError(GitHubAPIError):
    """
    The branch ref already exists (GitHub returned 422 "Reference already
    exists" from POST .../git/refs).

    Raised instead of silently succeeding: this client has no way to know
    whether the existing branch points at the SHA the caller wanted, so
    treating "already exists" as success would risk masking a real conflict.
    Callers decide how to handle it (e.g. reuse the branch after checking
    where it points, or fail the operation with a clear message).
    """


class InvalidBranchNameError(ValueError):
    """Raised when a branch name fails local validation, before any API call.

    Mirrors app.ingestion.url_parser.InvalidRepositoryURLError's pattern:
    a plain ValueError subclass for a client-side input validation failure.
    """


# --- Branch name validation ---------------------------------------------
#
# A practical subset of `git check-ref-format --branch` rules — enough to
# reject the inputs that would otherwise produce a confusing GitHub 422 or,
# worse, something that means something unexpected to git (e.g. "@{",
# leading "-" which looks like a flag to some tools).

_MAX_BRANCH_NAME_LENGTH = 250
_FORBIDDEN_BRANCH_CHARS = set(" ~^:?*[\\")


def validate_branch_name(branch_name: str) -> str:
    """Return the trimmed branch name if valid, else raise InvalidBranchNameError."""
    name = branch_name.strip() if branch_name else ""

    if not name:
        raise InvalidBranchNameError("Branch name must not be empty.")
    if len(name) > _MAX_BRANCH_NAME_LENGTH:
        raise InvalidBranchNameError(f"Branch name exceeds {_MAX_BRANCH_NAME_LENGTH} characters.")
    if name in {"@", "HEAD"}:
        raise InvalidBranchNameError(f"'{name}' is not a valid branch name.")
    if name.startswith("/") or name.endswith("/") or "//" in name:
        raise InvalidBranchNameError("Branch name must not start/end with '/' or contain '//'.")
    if name.startswith("-"):
        raise InvalidBranchNameError("Branch name must not start with '-'.")
    if name.endswith(".") or name.endswith(".lock"):
        raise InvalidBranchNameError("Branch name must not end with '.' or '.lock'.")
    if ".." in name:
        raise InvalidBranchNameError("Branch name must not contain '..'.")
    if "@{" in name:
        raise InvalidBranchNameError("Branch name must not contain '@{'.")
    if any(part.startswith(".") for part in name.split("/")):
        raise InvalidBranchNameError("No path component of a branch name may start with '.'.")
    if any(ch in _FORBIDDEN_BRANCH_CHARS for ch in name):
        raise InvalidBranchNameError(
            "Branch name must not contain any of: space ~ ^ : ? * [ \\"
        )
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
        raise InvalidBranchNameError("Branch name must not contain control characters.")

    return name


# --- Result shapes ------------------------------------------------------


@dataclass(frozen=True)
class TreeFileEntry:
    """One file to include in a new Git tree. See GitHubWriteClient.create_tree."""

    path: str
    blob_sha: str
    mode: str = "100644"
    type: str = "blob"


@dataclass(frozen=True)
class PullRequestResult:
    number: int
    url: str
    state: str


# --- Client ---------------------------------------------------------------


class GitHubWriteClient:
    """Thin async wrapper around GitHub's write endpoints. Use as an async context manager."""

    def __init__(
        self,
        token: Optional[str],
        timeout_seconds: float,
        bot_name: str,
        bot_email: str,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        if not token:
            raise GitHubWriteConfigurationError(
                "GITHUB_WRITE_TOKEN is not configured. Branch/commit/pull-request operations "
                "require a token with repository write access (classic PAT with 'repo' scope, "
                "or a fine-grained PAT with Contents: Read & write + Pull requests: Read & write). "
                "Set GITHUB_WRITE_TOKEN in your .env file -- GITHUB_TOKEN is reserved for "
                "read-only ingestion and is never reused for writes."
            )
        self._token = token
        self._timeout_seconds = timeout_seconds
        self._bot_name = bot_name
        self._bot_email = bot_email
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "GitHubWriteClient":
        self._client = httpx.AsyncClient(
            base_url=GITHUB_API_BASE_URL,
            headers=self._build_headers(),
            timeout=self._timeout_seconds,
            transport=self._transport,
        )
        return self

    async def __aexit__(self, *_exc_info) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _build_headers(self) -> dict[str, str]:
        # Never log this header -- it carries the write-scoped token.
        return {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Authorization": f"Bearer {self._token}",
        }

    async def _request(self, method: str, path: str, json: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        assert self._client is not None, "GitHubWriteClient must be used as an async context manager"
        try:
            response = await self._client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            raise GitHubAPIError(f"GitHub API request timed out: {method} {path}") from exc
        except httpx.HTTPError as exc:
            raise GitHubAPIError(f"GitHub API request failed: {method} {path} ({exc})") from exc

        if response.status_code == 404:
            raise GitHubNotFoundError(f"Not found: {method} {path}", status_code=404)

        if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
            reset_header = response.headers.get("X-RateLimit-Reset")
            reset_at = int(reset_header) if reset_header else None
            raise GitHubRateLimitError("GitHub API rate limit exceeded", reset_at_epoch=reset_at)

        if response.status_code >= 400:
            raise GitHubAPIError(
                f"GitHub API error {response.status_code} for {method} {path}: {_github_message(response)}",
                status_code=response.status_code,
            )

        return response.json() if response.content else {}

    # --- Reads used to discover the correct base for a write ------------
    #
    # These are GET calls, not writes -- kept here (rather than in
    # app.services.github_client, which must stay strictly read-only in its
    # own right per that module's design) because they exist purely to
    # support this client's own write preparation: finding the commit a
    # branch currently points at, and a commit's tree SHA, so callers never
    # have to assume a blob/tree SHA is a commit SHA or vice versa.

    async def get_branch_head_commit_sha(self, owner: str, repo: str, branch_name: str) -> str:
        """Resolve a branch name to the commit SHA it currently points at."""
        data = await self._request("GET", f"/repos/{owner}/{repo}/git/refs/heads/{branch_name}")
        return data["object"]["sha"]

    async def get_commit(self, owner: str, repo: str, commit_sha: str) -> dict[str, Any]:
        """Fetch a commit object: its tree SHA, parent commit SHA(s), and message."""
        return await self._request("GET", f"/repos/{owner}/{repo}/git/commits/{commit_sha}")

    # --- 1. Branches ---------------------------------------------------

    async def create_branch(self, owner: str, repo: str, branch_name: str, base_sha: str) -> dict[str, Any]:
        """Create refs/heads/{branch_name} pointing at base_sha.

        Raises GitHubBranchExistsError (not a generic GitHubAPIError) if the
        branch already exists, so callers can decide what to do rather than
        this client silently treating it as success.
        """
        name = validate_branch_name(branch_name)
        try:
            return await self._request(
                "POST",
                f"/repos/{owner}/{repo}/git/refs",
                json={"ref": f"refs/heads/{name}", "sha": base_sha},
            )
        except GitHubAPIError as exc:
            if exc.status_code == 422 and "already exists" in str(exc).lower():
                raise GitHubBranchExistsError(
                    f"Branch 'refs/heads/{name}' already exists.", status_code=422
                ) from exc
            raise

    # --- 2. Blobs --------------------------------------------------------

    async def create_blob(self, owner: str, repo: str, content: str) -> str:
        """Create a blob from complete UTF-8 file content. Returns the blob SHA."""
        data = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/git/blobs",
            json={"content": content, "encoding": "utf-8"},
        )
        return data["sha"]

    # --- 3. Trees ---------------------------------------------------------

    async def create_tree(
        self, owner: str, repo: str, base_tree_sha: str, files: list[TreeFileEntry]
    ) -> str:
        """Create a tree layering `files` on top of base_tree_sha. Returns the tree SHA."""
        if not files:
            raise ValueError("create_tree requires at least one file entry.")

        tree_items = [
            {"path": f.path, "mode": f.mode, "type": f.type, "sha": f.blob_sha} for f in files
        ]
        data = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/git/trees",
            json={"base_tree": base_tree_sha, "tree": tree_items},
        )
        return data["sha"]

    # --- 4. Commits ---------------------------------------------------------

    async def create_commit(
        self, owner: str, repo: str, message: str, tree_sha: str, parent_sha: str
    ) -> str:
        """Create a commit with the configured bot identity as author and committer."""
        identity = {"name": self._bot_name, "email": self._bot_email}
        data = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/git/commits",
            json={
                "message": message,
                "tree": tree_sha,
                "parents": [parent_sha],
                "author": identity,
                "committer": identity,
            },
        )
        return data["sha"]

    # --- 5. Update branch reference -----------------------------------

    async def update_branch_reference(self, owner: str, repo: str, branch_name: str, commit_sha: str) -> dict[str, Any]:
        """Point refs/heads/{branch_name} at commit_sha. Never force-updates."""
        name = validate_branch_name(branch_name)
        return await self._request(
            "PATCH",
            f"/repos/{owner}/{repo}/git/refs/heads/{name}",
            json={"sha": commit_sha, "force": False},
        )

    # --- 6. Pull requests -------------------------------------------------

    async def create_pull_request(
        self, owner: str, repo: str, title: str, body: str, head_branch: str, base_branch: str
    ) -> PullRequestResult:
        data = await self._request(
            "POST",
            f"/repos/{owner}/{repo}/pulls",
            json={"title": title, "body": body, "head": head_branch, "base": base_branch},
        )
        return PullRequestResult(number=data["number"], url=data["html_url"], state=data["state"])


def _github_message(response: httpx.Response) -> str:
    """Best-effort extraction of GitHub's error `message` field for a clearer exception string."""
    try:
        data = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(data, dict) and isinstance(data.get("message"), str):
        return data["message"]
    return response.text[:200]
