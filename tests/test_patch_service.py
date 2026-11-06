"""Offline tests for app.services.patch_service — diff generation and server-side validation."""

from app.ingestion.store import IngestedRepository, RetrievedFile
from app.schemas.proposal import ProposedFileChangeDraft
from app.services.patch_service import (
    MAX_FILES_PER_PROPOSAL,
    check_for_staleness,
    compute_unified_diff,
    hash_content,
    validate_proposed_changes,
)


def _repository(**files: str) -> IngestedRepository:
    return IngestedRepository(
        owner="acme",
        repo="webapp",
        default_branch="main",
        files=[RetrievedFile(path=path, content=content, size=len(content), sha="x") for path, content in files.items()],
    )


# --- Unified diff generation ---


def test_compute_unified_diff_reflects_real_changes():
    original = "def f():\n    return 1\n"
    proposed = "def f():\n    return 2\n"
    diff = compute_unified_diff("app/f.py", original, proposed)
    assert "-    return 1" in diff
    assert "+    return 2" in diff
    assert "a/app/f.py" in diff and "b/app/f.py" in diff


def test_compute_unified_diff_empty_for_identical_content():
    content = "unchanged\n"
    diff = compute_unified_diff("app/f.py", content, content)
    assert diff == ""


# --- Patch validation ---


def test_validate_proposed_changes_accepts_a_real_fix():
    repo = _repository(**{"app/auth/deactivate.py": "def deactivate_user(user):\n    user.is_active = False\n"})
    drafts = [
        ProposedFileChangeDraft(
            file_path="app/auth/deactivate.py",
            change_summary="Revoke session on deactivation.",
            proposed_content="def deactivate_user(user):\n    user.is_active = False\n    user.session_token = None\n",
        )
    ]
    changes, result = validate_proposed_changes(drafts, repo)
    assert result.valid is True
    assert len(changes) == 1
    assert changes[0].original_content_hash == hash_content(repo.files[0].content)
    assert "session_token" in changes[0].unified_diff


def test_validate_proposed_changes_rejects_no_changes():
    repo = _repository(**{"a.py": "x = 1\n"})
    changes, result = validate_proposed_changes([], repo)
    assert result.valid is False
    assert any(issue.code == "no_changes" for issue in result.issues)
    assert changes == []


def test_validate_proposed_changes_rejects_identical_content():
    repo = _repository(**{"a.py": "x = 1\n"})
    drafts = [ProposedFileChangeDraft(file_path="a.py", change_summary="no-op", proposed_content="x = 1\n")]
    _, result = validate_proposed_changes(drafts, repo)
    assert result.valid is False
    assert any(issue.code == "no_effective_change" for issue in result.issues)


def test_validate_proposed_changes_rejects_too_many_files():
    repo = _repository(**{f"f{i}.py": f"x{i} = {i}\n" for i in range(MAX_FILES_PER_PROPOSAL + 1)})
    drafts = [
        ProposedFileChangeDraft(file_path=f"f{i}.py", change_summary="c", proposed_content=f"x{i} = {i + 1}\n")
        for i in range(MAX_FILES_PER_PROPOSAL + 1)
    ]
    _, result = validate_proposed_changes(drafts, repo)
    assert result.valid is False
    assert any(issue.code == "too_many_files" for issue in result.issues)


def test_validate_proposed_changes_rejects_unsupported_extension():
    repo = _repository(**{"image.png": "binary-ish"})
    drafts = [ProposedFileChangeDraft(file_path="image.png", change_summary="c", proposed_content="changed")]
    _, result = validate_proposed_changes(drafts, repo)
    assert result.valid is False
    assert any(issue.code == "unsupported_file_type" for issue in result.issues)


# --- Path traversal / unknown files (never generation/deletion of arbitrary files) ---


def test_validate_proposed_changes_rejects_path_traversal():
    repo = _repository(**{"a.py": "x = 1\n"})
    drafts = [
        ProposedFileChangeDraft(file_path="../../etc/passwd", change_summary="c", proposed_content="evil"),
        ProposedFileChangeDraft(file_path="/etc/passwd", change_summary="c", proposed_content="evil"),
        ProposedFileChangeDraft(file_path="app/../../../secrets.py", change_summary="c", proposed_content="evil"),
    ]
    changes, result = validate_proposed_changes(drafts, repo)
    assert result.valid is False
    assert changes == []
    codes = {issue.code for issue in result.issues}
    assert "invalid_path" in codes


def test_validate_proposed_changes_rejects_file_creation():
    """A path not already present in the ingested repository is rejected -- no new-file creation."""
    repo = _repository(**{"a.py": "x = 1\n"})
    drafts = [ProposedFileChangeDraft(file_path="b.py", change_summary="new file", proposed_content="y = 2\n")]
    changes, result = validate_proposed_changes(drafts, repo)
    assert result.valid is False
    assert changes == []
    assert any(issue.code == "unknown_file" for issue in result.issues)


# --- Staleness detection ---


def test_check_for_staleness_detects_changed_content():
    repo_at_proposal_time = _repository(**{"a.py": "x = 1\n"})
    drafts = [ProposedFileChangeDraft(file_path="a.py", change_summary="c", proposed_content="x = 2\n")]
    changes, _ = validate_proposed_changes(drafts, repo_at_proposal_time)

    repo_now = _repository(**{"a.py": "x = 999  # someone re-ingested with different content\n"})
    issues = check_for_staleness(changes, repo_now)
    assert any(issue.code == "stale_proposal" for issue in issues)


def test_check_for_staleness_passes_when_unchanged():
    repo = _repository(**{"a.py": "x = 1\n"})
    drafts = [ProposedFileChangeDraft(file_path="a.py", change_summary="c", proposed_content="x = 2\n")]
    changes, _ = validate_proposed_changes(drafts, repo)

    assert check_for_staleness(changes, repo) == []


def test_check_for_staleness_detects_removed_file():
    repo_at_proposal_time = _repository(**{"a.py": "x = 1\n"})
    drafts = [ProposedFileChangeDraft(file_path="a.py", change_summary="c", proposed_content="x = 2\n")]
    changes, _ = validate_proposed_changes(drafts, repo_at_proposal_time)

    repo_now = _repository(**{"other.py": "y = 1\n"})  # a.py no longer exists
    issues = check_for_staleness(changes, repo_now)
    assert any(issue.code == "file_removed" for issue in issues)
