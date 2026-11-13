"""
Unit tests for app.agents.nodes.apply_patch.apply_patch_node (Milestone 9:
real GitHub branch/commit/PR creation after human approval).

Exercises the node directly (not through the full proposal graph or API) so
every branch of its partial-failure handling is cheap and precise to test.
GitHubWriteClient is always either the FakeGitHubWriteClient test double
(tests/conftest.py) or, for the "missing token" case, the real class
constructed with no token (still no network call -- it raises before one
would ever be attempted).
"""

import app.agents.nodes.apply_patch as apply_patch_module
import pytest

from app.agents.nodes.apply_patch import apply_patch_node
from app.core.config import Settings
from app.ingestion.store import IngestedRepository, RepositoryStore, RetrievedFile
from app.schemas.proposal import ProposedFileChange
from app.services.patch_service import hash_content
from tests.conftest import FakeGitHubWriteClient

pytestmark = pytest.mark.anyio

OWNER = "PrasadYeole-Developer"
REPO = "MojitoCocktails"
REPOSITORY_ID = f"{OWNER}/{REPO}"


def _settings(write_token: str | None = "fake-write-token") -> Settings:
    return Settings(_env_file=None, github_write_token=write_token)


def _repository(**files: str) -> IngestedRepository:
    return IngestedRepository(
        owner=OWNER,
        repo=REPO,
        default_branch="main",
        files=[RetrievedFile(path=path, content=content, size=len(content), sha="x") for path, content in files.items()],
    )


def _change(path: str, original_content: str, proposed_content: str, summary: str = "fix it") -> ProposedFileChange:
    return ProposedFileChange(
        file_path=path,
        original_content_hash=hash_content(original_content),
        proposed_content=proposed_content,
        unified_diff="--- a\n+++ b\n",
        change_summary=summary,
    )


def _state(proposal_id: str, repository_id: str, changes: list[ProposedFileChange]) -> dict:
    return {
        "proposal_id": proposal_id,
        "repository_id": repository_id,
        "bug_report": "something is broken",
        "investigation_report": None,
        "messages": [],
        "proposal_draft": None,
        "proposed_changes": changes,
        "validation_result": None,
        "decision": "approved",
        "decision_reason": None,
        "status": "APPROVED",
        "error": None,
        "apply_result": None,
    }


def _store(repository: IngestedRepository) -> RepositoryStore:
    store = RepositoryStore()
    store.save(repository)
    return store


def _fake_client(monkeypatch, **kwargs) -> FakeGitHubWriteClient:
    client = FakeGitHubWriteClient(**kwargs)
    client.seed_default_branch("main", commit_sha="base-sha-1", tree_sha="base-tree-1")
    monkeypatch.setattr(apply_patch_module, "GitHubWriteClient", lambda **kw: client)
    return client


def _explode(*_args, **_kwargs):
    raise AssertionError("GitHubWriteClient must not be constructed for this case.")


# --- Success: single and multi-file ---------------------------------------


async def test_single_file_proposal_succeeds(monkeypatch):
    repo = _repository(**{"a.py": "x = 1\n"})
    client = _fake_client(monkeypatch)
    changes = [_change("a.py", "x = 1\n", "x = 2\n")]

    result = await apply_patch_node(_state("prop-1", REPOSITORY_ID, changes), store=_store(repo), settings=_settings())

    assert result["status"] == "APPLIED"
    apply_result = result["apply_result"]
    assert apply_result.branch_name == "devpilot/proposal-prop-1"
    assert apply_result.base_branch == "main"
    assert apply_result.commit_sha is not None
    assert apply_result.pull_request_number is not None
    assert apply_result.pull_request_url is not None
    assert apply_result.succeeded is True
    blob_calls = [c for c in client.calls if c[0] == "create_blob"]
    assert len(blob_calls) == 1


async def test_multi_file_proposal_succeeds(monkeypatch):
    repo = _repository(**{"a.py": "x = 1\n", "b.py": "y = 1\n", "c.py": "z = 1\n"})
    client = _fake_client(monkeypatch)
    changes = [
        _change("a.py", "x = 1\n", "x = 2\n"),
        _change("b.py", "y = 1\n", "y = 2\n"),
        _change("c.py", "z = 1\n", "z = 2\n"),
    ]

    result = await apply_patch_node(_state("prop-2", REPOSITORY_ID, changes), store=_store(repo), settings=_settings())

    assert result["status"] == "APPLIED"
    blob_calls = [c for c in client.calls if c[0] == "create_blob"]
    assert len(blob_calls) == 3
    tree_calls = [c for c in client.calls if c[0] == "create_tree"]
    assert len(tree_calls) == 1
    tree_files = tree_calls[0][1][3]  # (owner, repo, base_tree_sha, files)
    assert {f.path for f in tree_files} == {"a.py", "b.py", "c.py"}


# --- Pre-write validation: no GitHub call at all ---------------------------


async def test_repository_mismatch_fails_without_any_github_call(monkeypatch):
    monkeypatch.setattr(apply_patch_module, "GitHubWriteClient", _explode)
    repo = _repository(**{"a.py": "x = 1\n"})
    changes = [_change("a.py", "x = 1\n", "x = 2\n")]

    result = await apply_patch_node(
        _state("prop-3", "someone-else/other-repo", changes), store=_store(repo), settings=_settings()
    )

    assert result["status"] == "FAILED"
    assert "no longer matches" in result["error"]


async def test_stale_file_prevents_all_writes(monkeypatch):
    monkeypatch.setattr(apply_patch_module, "GitHubWriteClient", _explode)
    changes = [_change("a.py", "x = 1\n", "x = 2\n")]
    # The repository now has different content than when the proposal was drafted.
    repo_now = _repository(**{"a.py": "x = 999  # changed since the proposal was drafted\n"})

    result = await apply_patch_node(_state("prop-4", REPOSITORY_ID, changes), store=_store(repo_now), settings=_settings())

    assert result["status"] == "FAILED"
    assert "stale" in result["error"].lower()


async def test_missing_write_token_fails_without_any_github_call():
    # Deliberately NOT monkeypatched -- the real GitHubWriteClient class
    # raises in its own constructor before any network call is possible.
    repo = _repository(**{"a.py": "x = 1\n"})
    changes = [_change("a.py", "x = 1\n", "x = 2\n")]

    result = await apply_patch_node(
        _state("prop-5", REPOSITORY_ID, changes), store=_store(repo), settings=_settings(write_token=None)
    )

    assert result["status"] == "FAILED"
    assert "GITHUB_WRITE_TOKEN" in result["error"]
    assert "apply_result" not in result or result.get("apply_result") is None


# --- Failures at each write step: branch exists but nothing further was written ---


async def test_branch_creation_failure_is_plain_failed_not_partial(monkeypatch):
    repo = _repository(**{"a.py": "x = 1\n"})
    client = _fake_client(monkeypatch, fail_at="create_branch")
    changes = [_change("a.py", "x = 1\n", "x = 2\n")]

    result = await apply_patch_node(_state("prop-6", REPOSITORY_ID, changes), store=_store(repo), settings=_settings())

    # Nothing was ever created on GitHub this attempt -- plain FAILED.
    assert result["status"] == "FAILED"
    assert "Failed to create branch" in result["error"]


# --- Failures after the branch exists: APPLY_FAILED, partial state preserved ---


@pytest.mark.parametrize("fail_step", ["get_commit", "create_blob", "create_tree", "create_commit"])
async def test_failure_after_branch_created_is_apply_failed_with_no_commit_sha(monkeypatch, fail_step):
    repo = _repository(**{"a.py": "x = 1\n"})
    client = _fake_client(monkeypatch, fail_at=fail_step)
    changes = [_change("a.py", "x = 1\n", "x = 2\n")]

    result = await apply_patch_node(_state("prop-7", REPOSITORY_ID, changes), store=_store(repo), settings=_settings())

    assert result["status"] == "APPLY_FAILED"
    apply_result = result["apply_result"]
    assert apply_result.branch_name == "devpilot/proposal-prop-7"
    assert apply_result.commit_sha is None
    assert apply_result.succeeded is False
    assert any(c[0] == "create_branch" for c in client.calls)


async def test_ref_update_failure_preserves_commit_sha(monkeypatch):
    repo = _repository(**{"a.py": "x = 1\n"})
    client = _fake_client(monkeypatch, fail_at="update_branch_reference")
    changes = [_change("a.py", "x = 1\n", "x = 2\n")]

    result = await apply_patch_node(_state("prop-8", REPOSITORY_ID, changes), store=_store(repo), settings=_settings())

    assert result["status"] == "APPLY_FAILED"
    apply_result = result["apply_result"]
    assert apply_result.commit_sha is not None  # the commit itself was created
    assert apply_result.pull_request_number is None
    assert "moving branch" in result["error"]


async def test_pr_creation_failure_after_commit_preserves_branch_and_commit(monkeypatch):
    repo = _repository(**{"a.py": "x = 1\n"})
    client = _fake_client(monkeypatch, fail_at="create_pull_request")
    changes = [_change("a.py", "x = 1\n", "x = 2\n")]

    result = await apply_patch_node(_state("prop-9", REPOSITORY_ID, changes), store=_store(repo), settings=_settings())

    assert result["status"] == "APPLY_FAILED"
    apply_result = result["apply_result"]
    assert apply_result.branch_name == "devpilot/proposal-prop-9"
    assert apply_result.commit_sha is not None
    assert apply_result.pull_request_number is None
    assert apply_result.succeeded is False
    assert "pull request creation failed" in result["error"]


# --- Existing branch collision: safe reuse vs. refusal ---------------------


async def test_existing_branch_safe_to_reuse_continues_to_success(monkeypatch):
    repo = _repository(**{"a.py": "x = 1\n"})
    client = _fake_client(monkeypatch)
    branch_name = "devpilot/proposal-prop-10"
    # Pre-seed a branch whose tip commit genuinely belongs to this exact
    # proposal and is based on the current default-branch head.
    client.seed_existing_branch(
        branch_name,
        commit_sha="existing-commit-1",
        tree_sha="existing-tree-1",
        parent_sha="base-sha-1",
        message=f"DevPilot: previous attempt\n\nDevPilot-Proposal-Id: prop-10",
    )
    changes = [_change("a.py", "x = 1\n", "x = 2\n")]

    result = await apply_patch_node(_state("prop-10", REPOSITORY_ID, changes), store=_store(repo), settings=_settings())

    assert result["status"] == "APPLIED"
    # create_branch was attempted and collided, but the flow continued.
    assert any(c[0] == "create_branch" for c in client.calls)


async def test_existing_branch_unrelated_proposal_refuses(monkeypatch):
    repo = _repository(**{"a.py": "x = 1\n"})
    client = _fake_client(monkeypatch)
    branch_name = "devpilot/proposal-prop-11"
    client.seed_existing_branch(
        branch_name,
        commit_sha="existing-commit-2",
        tree_sha="existing-tree-2",
        parent_sha="base-sha-1",
        message="Some totally unrelated manual commit",
    )
    changes = [_change("a.py", "x = 1\n", "x = 2\n")]

    result = await apply_patch_node(_state("prop-11", REPOSITORY_ID, changes), store=_store(repo), settings=_settings())

    assert result["status"] == "FAILED"
    assert "does not reference" in result["error"]
    # Refused before touching any file content.
    assert not any(c[0] == "create_blob" for c in client.calls)


async def test_existing_branch_wrong_base_refuses(monkeypatch):
    repo = _repository(**{"a.py": "x = 1\n"})
    client = _fake_client(monkeypatch)
    branch_name = "devpilot/proposal-prop-12"
    # Same proposal id in the message, but based on a DIFFERENT (older) commit
    # than the current default-branch head -- default branch has moved on.
    client.seed_existing_branch(
        branch_name,
        commit_sha="existing-commit-3",
        tree_sha="existing-tree-3",
        parent_sha="some-older-sha",
        message="DevPilot: previous attempt\n\nDevPilot-Proposal-Id: prop-12",
    )
    changes = [_change("a.py", "x = 1\n", "x = 2\n")]

    result = await apply_patch_node(_state("prop-12", REPOSITORY_ID, changes), store=_store(repo), settings=_settings())

    assert result["status"] == "FAILED"
    assert "default branch has moved" in result["error"]
    assert not any(c[0] == "create_blob" for c in client.calls)


# --- Branch name construction -------------------------------------------


async def test_invalid_branch_name_fails_without_any_github_call(monkeypatch):
    monkeypatch.setattr(apply_patch_module, "GitHubWriteClient", _explode)
    repo = _repository(**{"a.py": "x = 1\n"})
    changes = [_change("a.py", "x = 1\n", "x = 2\n")]
    # A proposal_id containing characters that would make the derived branch
    # name invalid (uuid4().hex never does this in production; this proves
    # the defensive check actually runs).
    bad_proposal_id = "../bad id"

    result = await apply_patch_node(_state(bad_proposal_id, REPOSITORY_ID, changes), store=_store(repo), settings=_settings())

    assert result["status"] == "FAILED"
    assert "valid branch name" in result["error"]
