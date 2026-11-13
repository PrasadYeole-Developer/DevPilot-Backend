"""
Offline unit tests for app.services.github_write_client.

All GitHub calls are mocked via httpx.MockTransport (no respx dependency
needed -- httpx supports this natively) so no real network calls are made.
"""

import json

import httpx
import pytest

from app.services.github_client import GitHubAPIError, GitHubNotFoundError, GitHubRateLimitError
from app.services.github_write_client import (
    GitHubBranchExistsError,
    GitHubWriteClient,
    GitHubWriteConfigurationError,
    InvalidBranchNameError,
    TreeFileEntry,
    validate_branch_name,
)

pytestmark = pytest.mark.anyio

OWNER = "PrasadYeole-Developer"
REPO = "MojitoCocktails"


def _client(handler, token="write-token-123") -> GitHubWriteClient:
    transport = httpx.MockTransport(handler)
    return GitHubWriteClient(
        token=token,
        timeout_seconds=5.0,
        bot_name="DevPilot Bot",
        bot_email="devpilot-bot@users.noreply.github.com",
        transport=transport,
    )


def _json_response(status_code: int, payload: dict) -> httpx.Response:
    return httpx.Response(status_code, json=payload)


# --- Configuration -----------------------------------------------------


def test_missing_write_token_raises_configuration_error():
    with pytest.raises(GitHubWriteConfigurationError):
        GitHubWriteClient(
            token=None, timeout_seconds=5.0, bot_name="Bot", bot_email="bot@example.com"
        )


def test_missing_write_token_never_mentions_github_token_as_fallback():
    with pytest.raises(GitHubWriteConfigurationError) as exc_info:
        GitHubWriteClient(token="", timeout_seconds=5.0, bot_name="Bot", bot_email="bot@example.com")
    assert "GITHUB_WRITE_TOKEN" in str(exc_info.value)


# --- Token is never logged/exposed --------------------------------------


async def test_token_is_sent_as_bearer_header_and_never_in_body():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _json_response(201, {"ref": "refs/heads/x", "object": {"sha": "abc123"}})

    client = _client(handler, token="super-secret-token")
    async with client:
        await client.create_branch(OWNER, REPO, "devpilot/fix-1", "basesha")

    assert captured[0].headers["Authorization"] == "Bearer super-secret-token"
    assert b"super-secret-token" not in captured[0].content


# --- 1. Branch creation ---------------------------------------------------


async def test_create_branch_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == f"/repos/{OWNER}/{REPO}/git/refs"
        payload = json.loads(request.content)
        assert payload == {"ref": "refs/heads/devpilot/fix-1", "sha": "base-sha-1"}
        return _json_response(201, {"ref": "refs/heads/devpilot/fix-1", "object": {"sha": "base-sha-1"}})

    client = _client(handler)
    async with client:
        result = await client.create_branch(OWNER, REPO, "devpilot/fix-1", "base-sha-1")

    assert result["ref"] == "refs/heads/devpilot/fix-1"


async def test_create_branch_already_exists_raises_specific_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(422, {"message": "Reference already exists"})

    client = _client(handler)
    async with client:
        with pytest.raises(GitHubBranchExistsError):
            await client.create_branch(OWNER, REPO, "devpilot/fix-1", "base-sha-1")


async def test_create_branch_invalid_name_never_calls_api():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("No HTTP call should be made for an invalid branch name.")

    client = _client(handler)
    async with client:
        with pytest.raises(InvalidBranchNameError):
            await client.create_branch(OWNER, REPO, "bad branch name", "base-sha-1")


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "a" * 300,
        "feature/../etc",
        "feature..broken",
        "/leading-slash",
        "trailing-slash/",
        "double//slash",
        "ends-with-dot.",
        "lockfile.lock",
        "has space",
        "has~tilde",
        "has^caret",
        "has:colon",
        "has?question",
        "has*star",
        "has[bracket",
        "has\\backslash",
        "weird@{sequence}",
        "@",
        "-leading-dash",
        ".hidden/segment",
    ],
)
def test_validate_branch_name_rejects_invalid_names(name):
    with pytest.raises(InvalidBranchNameError):
        validate_branch_name(name)


@pytest.mark.parametrize(
    "name",
    ["devpilot/fix-session-bug", "fix_123", "release-1.0", "a/b/c"],
)
def test_validate_branch_name_accepts_valid_names(name):
    assert validate_branch_name(name) == name


# --- 2. Blob creation -------------------------------------------------


async def test_create_blob_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/repos/{OWNER}/{REPO}/git/blobs"
        return _json_response(201, {"sha": "blob-sha-1", "url": "https://api.github.com/..."})

    client = _client(handler)
    async with client:
        sha = await client.create_blob(OWNER, REPO, "print('hello')\n")

    assert sha == "blob-sha-1"


# --- 3. Tree creation (multi-file) --------------------------------------


async def test_create_tree_with_multiple_files():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["base_tree"] == "base-tree-sha"
        assert payload["tree"] == [
            {"path": "app/a.py", "mode": "100644", "type": "blob", "sha": "sha-a"},
            {"path": "app/b.py", "mode": "100644", "type": "blob", "sha": "sha-b"},
        ]
        return _json_response(201, {"sha": "new-tree-sha"})

    client = _client(handler)
    async with client:
        tree_sha = await client.create_tree(
            OWNER,
            REPO,
            "base-tree-sha",
            [
                TreeFileEntry(path="app/a.py", blob_sha="sha-a"),
                TreeFileEntry(path="app/b.py", blob_sha="sha-b"),
            ],
        )

    assert tree_sha == "new-tree-sha"


async def test_create_tree_requires_at_least_one_file():
    client = _client(lambda request: _json_response(201, {"sha": "x"}))
    async with client:
        with pytest.raises(ValueError):
            await client.create_tree(OWNER, REPO, "base-tree-sha", [])


# --- 4. Commit creation -------------------------------------------------


async def test_create_commit_uses_configured_bot_identity():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["message"] == "Fix session revocation bug"
        assert payload["tree"] == "tree-sha-1"
        assert payload["parents"] == ["parent-sha-1"]
        assert payload["author"] == {"name": "DevPilot Bot", "email": "devpilot-bot@users.noreply.github.com"}
        assert payload["committer"] == payload["author"]
        return _json_response(201, {"sha": "commit-sha-1"})

    client = _client(handler)
    async with client:
        sha = await client.create_commit(
            OWNER, REPO, "Fix session revocation bug", "tree-sha-1", "parent-sha-1"
        )

    assert sha == "commit-sha-1"


# --- 5. Branch reference update -----------------------------------------


async def test_update_branch_reference_never_forces():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "PATCH"
        assert request.url.path == f"/repos/{OWNER}/{REPO}/git/refs/heads/devpilot/fix-1"
        payload = json.loads(request.content)
        assert payload == {"sha": "new-commit-sha", "force": False}
        return _json_response(200, {"ref": "refs/heads/devpilot/fix-1", "object": {"sha": "new-commit-sha"}})

    client = _client(handler)
    async with client:
        result = await client.update_branch_reference(OWNER, REPO, "devpilot/fix-1", "new-commit-sha")

    assert result["object"]["sha"] == "new-commit-sha"


async def test_update_branch_reference_invalid_name_never_calls_api():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("No HTTP call should be made for an invalid branch name.")

    client = _client(handler)
    async with client:
        with pytest.raises(InvalidBranchNameError):
            await client.update_branch_reference(OWNER, REPO, "bad name", "sha")


# --- 6. Pull request creation --------------------------------------------


async def test_create_pull_request_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/repos/{OWNER}/{REPO}/pulls"
        payload = json.loads(request.content)
        assert payload == {
            "title": "Fix session revocation bug",
            "body": "Closes the deactivation session leak.",
            "head": "devpilot/fix-1",
            "base": "main",
        }
        return _json_response(
            201,
            {
                "number": 42,
                "html_url": f"https://github.com/{OWNER}/{REPO}/pull/42",
                "url": f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/42",
                "state": "open",
            },
        )

    client = _client(handler)
    async with client:
        result = await client.create_pull_request(
            OWNER,
            REPO,
            "Fix session revocation bug",
            "Closes the deactivation session leak.",
            "devpilot/fix-1",
            "main",
        )

    assert result.number == 42
    assert result.url == f"https://github.com/{OWNER}/{REPO}/pull/42"
    assert result.state == "open"


# --- GitHub API error handling --------------------------------------------


async def test_not_found_raises_github_not_found_error():
    client = _client(lambda request: httpx.Response(404, json={"message": "Not Found"}))
    async with client:
        with pytest.raises(GitHubNotFoundError):
            await client.create_blob(OWNER, REPO, "content")


async def test_generic_client_error_raises_github_api_error():
    client = _client(lambda request: httpx.Response(401, json={"message": "Bad credentials"}))
    async with client:
        with pytest.raises(GitHubAPIError) as exc_info:
            await client.create_blob(OWNER, REPO, "content")
    assert exc_info.value.status_code == 401
    assert "Bad credentials" in str(exc_info.value)


async def test_conflict_raises_github_api_error():
    client = _client(lambda request: httpx.Response(409, json={"message": "Conflict"}))
    async with client:
        with pytest.raises(GitHubAPIError) as exc_info:
            await client.update_branch_reference(OWNER, REPO, "devpilot/fix-1", "sha")
    assert exc_info.value.status_code == 409


async def test_rate_limit_raises_github_rate_limit_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1700000000"},
            json={"message": "API rate limit exceeded"},
        )

    client = _client(handler)
    async with client:
        with pytest.raises(GitHubRateLimitError) as exc_info:
            await client.create_blob(OWNER, REPO, "content")
    assert exc_info.value.reset_at_epoch == 1700000000


async def test_timeout_raises_github_api_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out")

    client = _client(handler)
    async with client:
        with pytest.raises(GitHubAPIError):
            await client.create_blob(OWNER, REPO, "content")


async def test_network_error_raises_github_api_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = _client(handler)
    async with client:
        with pytest.raises(GitHubAPIError):
            await client.create_blob(OWNER, REPO, "content")
