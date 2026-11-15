"""
MCP client for GitHub pull-request creation ONLY (Phase 1 of the GitHub ->
MCP migration).

Scope is deliberately narrow, per the Phase 0/0-continued audit of the
official `github/github-mcp-server` (verified by reading its actual Go
source, not just its README):

- `create_pull_request` is the one write operation with a safe, verified
  MCP equivalent -- its input schema (owner, repo, title, head, base,
  body, draft, maintainer_can_modify, reviewers) matches what
  GitHubWriteClient.create_pull_request already sends.
- Every other GitHub operation DevPilot performs (repo/tree/blob reads;
  branch/blob/tree/commit/ref-update writes) stays on the existing REST
  clients (app.services.github_client, app.services.github_write_client).
  The official server has no tool that can create a branch from an exact,
  pre-validated commit SHA (create_branch only takes a branch *name* and
  resolves it to "whatever that branch currently points to"), no tool that
  sets a custom commit author/committer identity (push_files's internal
  CreateCommit call never sets Author/Committer), and get_commit's output
  type (MinimalCommit) never exposes a tree SHA or parent SHA(s) -- so
  DevPilot's branch-collision verification and bot commit identity simply
  cannot be reproduced through this server for those operations.

Connects to the official remote GitHub MCP server
(https://api.githubcopilot.com/mcp/, Streamable HTTP transport) via the
official `mcp` Python SDK (`mcp.Client` + `mcp.client.streamable_http.
streamable_http_client` -- verified against the actual 2.3.0 wheel on
PyPI, not just the SDK's docs). The connection is scoped to exactly one
tool via the server's documented `X-MCP-Tools` header (see
github/github-mcp-server's docs/remote-server.md), so this client cannot
be used for anything else even by mistake.

Reuses `settings.github_write_token` -- the same write-scoped credential
GitHubWriteClient already uses, not a new credential surface. Never
logged (see _build_headers' token usage below).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import httpx
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

from app.services.github_client import GitHubAPIError
from app.services.github_write_client import PullRequestResult

# Scoping the connection to exactly this one tool via the server's
# X-MCP-Tools header means a bug in this client cannot accidentally reach
# any other GitHub MCP tool, regardless of what gets called.
GITHUB_MCP_PULL_REQUEST_TOOL = "create_pull_request"


# --- Errors -----------------------------------------------------------------


class GitHubMCPConfigurationError(Exception):
    """Raised when GitHubMCPClient is used without a configured write token.

    Mirrors app.services.github_write_client.GitHubWriteConfigurationError:
    a config problem, distinct from the GitHubAPIError family below (which
    means a call to the MCP server was actually attempted and failed).
    """


class GitHubMCPConnectionError(GitHubAPIError):
    """The MCP connection to the GitHub MCP server could not be established,
    or the tool call failed at the transport/protocol level (not a tool-
    reported error -- see GitHubMCPToolError for that).

    Subclasses GitHubAPIError so existing `except GitHubAPIError` call
    sites (e.g. apply_patch_node's PR-creation step) keep working
    unchanged -- see app.agents.nodes.apply_patch.
    """


class GitHubMCPTimeoutError(GitHubAPIError):
    """An MCP connection or tool call exceeded the configured timeout."""


class GitHubMCPToolError(GitHubAPIError):
    """The create_pull_request MCP tool itself reported a failure
    (CallToolResult.is_error=True) -- e.g. GitHub rejected the PR creation
    (no commits between head/base, branch not found, etc). The GitHub MCP
    server reports these as a normal (non-exceptional) tool result with
    is_error=True and a human-readable message in content[0].text, not as
    an MCP protocol-level error -- verified from
    github/github-mcp-server's pkg/utils/result.go.
    """


# --- Client ------------------------------------------------------------


class GitHubMCPClient:
    """Thin async wrapper around the official GitHub MCP server's
    create_pull_request tool. Use as an async context manager.
    """

    def __init__(self, token: Optional[str], server_url: str, timeout_seconds: float):
        if not token:
            raise GitHubMCPConfigurationError(
                "GITHUB_WRITE_TOKEN is not configured. Pull-request creation via the GitHub "
                "MCP server requires the same write-scoped token GitHubWriteClient uses -- "
                "set GITHUB_WRITE_TOKEN in your .env file."
            )
        self._token = token
        self._server_url = server_url
        self._timeout_seconds = timeout_seconds
        self._client: Optional[Client] = None

    async def __aenter__(self) -> "GitHubMCPClient":
        # Never log this header -- it carries the write-scoped token.
        http_client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {self._token}",
                "X-MCP-Tools": GITHUB_MCP_PULL_REQUEST_TOOL,
            },
        )
        transport = streamable_http_client(self._server_url, http_client=http_client)
        client = Client(transport, read_timeout_seconds=self._timeout_seconds)
        try:
            await asyncio.wait_for(client.__aenter__(), timeout=self._timeout_seconds)
        except TimeoutError as exc:
            raise GitHubMCPTimeoutError(
                f"Timed out connecting to the GitHub MCP server at {self._server_url}"
            ) from exc
        except Exception as exc:
            raise GitHubMCPConnectionError(
                f"Could not connect to the GitHub MCP server at {self._server_url}: {exc}"
            ) from exc
        self._client = client
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            await client.__aexit__(*exc_info)

    async def create_pull_request(
        self, owner: str, repo: str, title: str, body: str, head_branch: str, base_branch: str
    ) -> PullRequestResult:
        assert self._client is not None, "GitHubMCPClient must be used as an async context manager"

        arguments = {
            "owner": owner,
            "repo": repo,
            "title": title,
            "head": head_branch,
            "base": base_branch,
            "body": body,
        }

        try:
            result = await asyncio.wait_for(
                self._client.call_tool(
                    GITHUB_MCP_PULL_REQUEST_TOOL, arguments, read_timeout_seconds=self._timeout_seconds
                ),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as exc:
            raise GitHubMCPTimeoutError(
                f"Timed out waiting for the GitHub MCP server to respond to "
                f"'{GITHUB_MCP_PULL_REQUEST_TOOL}'"
            ) from exc
        except GitHubAPIError:
            raise
        except Exception as exc:
            raise GitHubMCPConnectionError(
                f"GitHub MCP call to '{GITHUB_MCP_PULL_REQUEST_TOOL}' failed: {exc}"
            ) from exc

        text = _first_text_block(result)
        if result.is_error:
            raise GitHubMCPToolError(
                f"GitHub MCP tool '{GITHUB_MCP_PULL_REQUEST_TOOL}' reported a failure: {text}"
            )

        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise GitHubMCPToolError(
                f"GitHub MCP tool '{GITHUB_MCP_PULL_REQUEST_TOOL}' returned an unparseable "
                f"response: {text!r}"
            ) from exc

        return _pull_request_result_from_mcp(data)


def _first_text_block(result: Any) -> str:
    for block in result.content:
        text = getattr(block, "text", None)
        if text is not None:
            return text
    return ""


def _pull_request_result_from_mcp(data: dict[str, Any]) -> PullRequestResult:
    """
    github/github-mcp-server's create_pull_request tool returns only
    {"id": "<string>", "url": "<string>"} (pkg/github/minimal_types.go's
    MinimalResponse, confirmed from source) -- a materially different shape
    than the REST API's response, which the existing GitHubWriteClient maps
    directly:

    - `id` is the PR's global database ID, NOT its repo-scoped number
      (GitHub's REST "number" field, e.g. the 42 in ".../pull/42") -- the
      tool never returns `number` at all.
    - No `state` field is returned at all.

    The PR number is recovered from the trailing path segment of `url`
    instead -- GitHub's pull-request HTML URLs are always
    "https://github.com/{owner}/{repo}/pull/{number}", a long-standing
    public URL format, not an internal implementation detail. `state` is
    set to "open" because a create call that didn't raise an error always
    results in a newly opened PR; GitHub has no other initial state.
    """
    url = data.get("url") or ""
    number: Optional[int] = None
    tail = url.rstrip("/").rsplit("/", 1)[-1] if url else ""
    if tail.isdigit():
        number = int(tail)
    return PullRequestResult(number=number, url=url, state="open")
