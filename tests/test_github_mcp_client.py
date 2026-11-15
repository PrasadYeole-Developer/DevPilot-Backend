"""
Unit tests for app.services.github_mcp_client.GitHubMCPClient.

GitHubMCPClient's dependency on `mcp.Client` is monkeypatched at the point
of use (the name imported into this module) -- exactly the same convention
as every other external service in this codebase (EmbeddingService,
PineconeService, GitHubWriteClient, etc.). No real MCP connection, no real
network call, no real GitHub write happens in any test here.
"""

from typing import Any, Optional

import app.services.github_mcp_client as github_mcp_client_module
import pytest

from app.services.github_mcp_client import (
    GitHubMCPClient,
    GitHubMCPConfigurationError,
    GitHubMCPConnectionError,
    GitHubMCPTimeoutError,
    GitHubMCPToolError,
)

pytestmark = pytest.mark.anyio

OWNER = "PrasadYeole-Developer"
REPO = "MojitoCocktails"


class _FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeCallToolResult:
    """Stand-in for mcp.types.CallToolResult -- only the fields GitHubMCPClient reads."""

    def __init__(self, text: str, is_error: bool = False):
        self.content = [_FakeTextBlock(text)]
        self.is_error = is_error


class _FakeMCPClient:
    """Stand-in for mcp.Client -- the one dependency GitHubMCPClient actually drives."""

    def __init__(
        self,
        *,
        enter_exc: Optional[BaseException] = None,
        call_result: Optional[_FakeCallToolResult] = None,
        call_exc: Optional[BaseException] = None,
    ):
        self._enter_exc = enter_exc
        self._call_result = call_result
        self._call_exc = call_exc
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> "_FakeMCPClient":
        if self._enter_exc is not None:
            raise self._enter_exc
        return self

    async def __aexit__(self, *_exc_info: Any) -> None:
        return None

    async def call_tool(self, name: str, arguments: dict[str, Any], read_timeout_seconds: Optional[float] = None):
        self.calls.append((name, arguments))
        if self._call_exc is not None:
            raise self._call_exc
        return self._call_result


def _client(monkeypatch, **fake_kwargs) -> tuple[GitHubMCPClient, _FakeMCPClient]:
    fake = _FakeMCPClient(**fake_kwargs)
    # GitHubMCPClient.__aenter__ calls Client(transport, read_timeout_seconds=...);
    # the transport argument (built from the real streamable_http_client) is never
    # touched by our fake, so no real HTTP client is ever entered.
    monkeypatch.setattr(github_mcp_client_module, "Client", lambda *args, **kwargs: fake)
    client = GitHubMCPClient(token="fake-token", server_url="https://example.invalid/mcp/", timeout_seconds=5.0)
    return client, fake


# --- Configuration -----------------------------------------------------


def test_missing_token_raises_configuration_error():
    with pytest.raises(GitHubMCPConfigurationError):
        GitHubMCPClient(token=None, server_url="https://example.invalid/mcp/", timeout_seconds=5.0)


def test_missing_token_error_names_github_write_token():
    with pytest.raises(GitHubMCPConfigurationError, match="GITHUB_WRITE_TOKEN"):
        GitHubMCPClient(token="", server_url="https://example.invalid/mcp/", timeout_seconds=5.0)


# --- Successful PR creation + response mapping --------------------------


async def test_successful_create_pull_request_maps_response(monkeypatch):
    client, fake = _client(
        monkeypatch,
        call_result=_FakeCallToolResult(
            '{"id": "2543112345", "url": "https://github.com/%s/%s/pull/42"}' % (OWNER, REPO)
        ),
    )

    async with client:
        result = await client.create_pull_request(OWNER, REPO, "My fix", "body text", "devpilot/fix-1", "main")

    assert result.number == 42
    assert result.url == f"https://github.com/{OWNER}/{REPO}/pull/42"
    assert result.state == "open"
    assert fake.calls == [
        (
            "create_pull_request",
            {
                "owner": OWNER,
                "repo": REPO,
                "title": "My fix",
                "head": "devpilot/fix-1",
                "base": "main",
                "body": "body text",
            },
        )
    ]


async def test_response_without_parseable_pr_number_leaves_number_none(monkeypatch):
    client, _ = _client(monkeypatch, call_result=_FakeCallToolResult('{"id": "1", "url": ""}'))

    async with client:
        result = await client.create_pull_request(OWNER, REPO, "t", "b", "h", "base")

    assert result.number is None
    assert result.url == ""
    assert result.state == "open"


# --- Connection failure --------------------------------------------------


async def test_connection_failure_on_enter_raises_connection_error(monkeypatch):
    client, _ = _client(monkeypatch, enter_exc=ConnectionError("connection refused"))

    with pytest.raises(GitHubMCPConnectionError):
        async with client:
            pass


async def test_generic_exception_during_call_raises_connection_error(monkeypatch):
    client, _ = _client(monkeypatch, call_exc=RuntimeError("boom"))

    async with client:
        with pytest.raises(GitHubMCPConnectionError):
            await client.create_pull_request(OWNER, REPO, "t", "b", "h", "base")


# --- Tool errors (is_error=True) ----------------------------------------


async def test_tool_reported_error_raises_tool_error(monkeypatch):
    client, _ = _client(
        monkeypatch,
        call_result=_FakeCallToolResult(
            "failed to create pull request: missing required parameter: head", is_error=True
        ),
    )

    async with client:
        with pytest.raises(GitHubMCPToolError, match="missing required parameter"):
            await client.create_pull_request(OWNER, REPO, "t", "b", "h", "base")


async def test_unparseable_tool_response_raises_tool_error(monkeypatch):
    client, _ = _client(monkeypatch, call_result=_FakeCallToolResult("not json at all"))

    async with client:
        with pytest.raises(GitHubMCPToolError):
            await client.create_pull_request(OWNER, REPO, "t", "b", "h", "base")


# --- Timeout handling -----------------------------------------------------


async def test_timeout_during_call_raises_timeout_error(monkeypatch):
    client, _ = _client(monkeypatch, call_exc=TimeoutError("too slow"))

    async with client:
        with pytest.raises(GitHubMCPTimeoutError):
            await client.create_pull_request(OWNER, REPO, "t", "b", "h", "base")


async def test_timeout_during_connect_raises_timeout_error(monkeypatch):
    client, _ = _client(monkeypatch, enter_exc=TimeoutError("too slow"))

    with pytest.raises(GitHubMCPTimeoutError):
        async with client:
            pass
