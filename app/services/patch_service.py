"""
Diff generation and server-side patch validation (Milestone 8).

Critical security property: unified diffs are ALWAYS computed here, from
(original, proposed) content, and NEVER trusted from the LLM. Every proposed
file path is checked against the exact, known set of files in the currently
ingested repository (the same ground-truth RepositoryStore the read-only
tools already use) — structurally identical to how
app/agents/tools/read_repository_file.py prevents traversal: there is no
filesystem path at all, only exact-match lookups against known content, so
"../../etc/passwd" or an absolute path simply never matches anything.
"""

import difflib
import hashlib

from app.ingestion.file_filter import ALLOWED_EXTENSIONS
from app.ingestion.store import IngestedRepository
from app.schemas.proposal import ProposedFileChange, ProposedFileChangeDraft, ValidationIssue, ValidationResult

MAX_FILES_PER_PROPOSAL = 5
MAX_DIFF_LINES_PER_FILE = 500
MAX_TOTAL_DIFF_LINES = 1500


def hash_content(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_unified_diff(file_path: str, original: str, proposed: str) -> str:
    lines = difflib.unified_diff(
        original.splitlines(keepends=True),
        proposed.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    return "".join(lines)


def _is_safe_relative_path(path: str) -> bool:
    if not path or path.startswith("/") or path.startswith("\\"):
        return False
    if ":" in path:  # e.g. "C:\\..." on Windows
        return False
    return ".." not in path.split("/")


def _has_allowed_extension(path: str) -> bool:
    filename = path.rsplit("/", 1)[-1]
    if "." not in filename:
        return False
    extension = "." + filename.rsplit(".", 1)[-1].lower()
    return extension in ALLOWED_EXTENSIONS


def validate_proposed_changes(
    drafts: list[ProposedFileChangeDraft],
    repository: IngestedRepository,
) -> tuple[list[ProposedFileChange], ValidationResult]:
    """
    Independently validates every proposed change against the real,
    currently-ingested repository content. Returns the finalized,
    server-computed changes (diff included) plus a report of any issues.
    `valid` is True only if there were zero issues AND at least one change
    survived — an empty-but-"valid" proposal is not a thing.
    """
    issues: list[ValidationIssue] = []
    finalized: list[ProposedFileChange] = []
    files_by_path = {f.path: f for f in repository.files}

    if not drafts:
        issues.append(ValidationIssue(code="no_changes", message="Proposal contains no file changes."))
        return finalized, ValidationResult(valid=False, issues=issues)

    if len(drafts) > MAX_FILES_PER_PROPOSAL:
        issues.append(
            ValidationIssue(
                code="too_many_files",
                message=f"Proposal touches {len(drafts)} files; the limit is {MAX_FILES_PER_PROPOSAL}.",
            )
        )

    total_diff_lines = 0
    seen_paths: set[str] = set()

    for draft in drafts:
        path = draft.file_path.strip()

        if path in seen_paths:
            issues.append(ValidationIssue(code="duplicate_file", file_path=path, message="File listed more than once."))
            continue
        seen_paths.add(path)

        if not _is_safe_relative_path(path):
            issues.append(
                ValidationIssue(
                    code="invalid_path", file_path=path,
                    message="Path traversal or absolute path rejected.",
                )
            )
            continue

        original_file = files_by_path.get(path)
        if original_file is None:
            # Also covers "file creation" — only files already present in the
            # ingested repository can be modified in this milestone.
            issues.append(
                ValidationIssue(
                    code="unknown_file", file_path=path,
                    message="File does not exist in the ingested repository; creating new files is not supported.",
                )
            )
            continue

        if not _has_allowed_extension(path):
            issues.append(
                ValidationIssue(code="unsupported_file_type", file_path=path, message="File type is not supported for modification.")
            )
            continue

        if draft.proposed_content == original_file.content:
            issues.append(
                ValidationIssue(code="no_effective_change", file_path=path, message="Proposed content is identical to the original.")
            )
            continue

        diff = compute_unified_diff(path, original_file.content, draft.proposed_content)
        diff_line_count = len(diff.splitlines())
        if diff_line_count > MAX_DIFF_LINES_PER_FILE:
            issues.append(
                ValidationIssue(
                    code="diff_too_large", file_path=path,
                    message=f"Diff is {diff_line_count} lines; the limit per file is {MAX_DIFF_LINES_PER_FILE}.",
                )
            )
            continue

        total_diff_lines += diff_line_count
        finalized.append(
            ProposedFileChange(
                file_path=path,
                original_content_hash=hash_content(original_file.content),
                proposed_content=draft.proposed_content,
                unified_diff=diff,
                change_summary=draft.change_summary,
            )
        )

    if total_diff_lines > MAX_TOTAL_DIFF_LINES:
        issues.append(
            ValidationIssue(
                code="total_diff_too_large",
                message=f"Combined diff is {total_diff_lines} lines; the limit is {MAX_TOTAL_DIFF_LINES}.",
            )
        )

    return finalized, ValidationResult(valid=not issues and bool(finalized), issues=issues)


def check_for_staleness(changes: list[ProposedFileChange], repository: IngestedRepository) -> list[ValidationIssue]:
    """
    Re-checked at approval/apply time (not just at generation time), since
    time can pass between drafting a proposal and it being approved — the
    repository could have been re-ingested in the meantime.
    """
    files_by_path = {f.path: f for f in repository.files}
    issues: list[ValidationIssue] = []
    for change in changes:
        current = files_by_path.get(change.file_path)
        if current is None:
            issues.append(
                ValidationIssue(
                    code="file_removed", file_path=change.file_path,
                    message="File no longer exists in the currently ingested repository.",
                )
            )
        elif hash_content(current.content) != change.original_content_hash:
            issues.append(
                ValidationIssue(
                    code="stale_proposal", file_path=change.file_path,
                    message="File content has changed since this proposal was generated.",
                )
            )
    return issues
