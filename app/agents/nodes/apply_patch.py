"""
LangGraph node: apply_patch.

Only reachable from request_human_approval's "approved" branch (see
app.agents.proposal_graph's conditional edges) -- the graph structure
itself, not a flag check, guarantees a rejected or still-pending proposal
can never reach here.

"Applying" now means: re-validate (repository identity + staleness, since
time may have passed since generation/validation), then actually create a
real GitHub branch, commit, and pull request via GitHubWriteClient --
*never* touching the repository's default branch directly, never force-
pushing, never merging. GitHubWriteClient is imported at module level (like
every other external service in this codebase, e.g.
app.api.routes.chat.EmbeddingService) specifically so tests can monkeypatch
`apply_patch` module's `GitHubWriteClient` name.

The original RepositoryStore snapshot is never mutated -- GitHub is the only
thing written to; the in-memory ingested content stays exactly what the
Repository page shows.

Failure granularity (see app.schemas.proposal.ProposalStatus):
- FAILED: nothing was written to GitHub in this attempt (pre-write
  validation failed, or write-token is missing, or the write that would have
  created the first new object -- the branch -- never actually created one,
  e.g. an unrelated pre-existing branch blocks it).
- APPLY_FAILED: at least one new GitHub object was created in this attempt
  (at minimum the branch) before something failed. apply_result always
  carries whatever was actually created, so a partial failure is never
  reported as if nothing happened, and never silently claimed as success.
- APPLIED: every step, including PR creation, succeeded.
"""

from __future__ import annotations

from app.agents.proposal_state import ProposalState
from app.core.config import Settings
from app.ingestion.store import RepositoryStore
from app.schemas.proposal import ApplyResult, ProposedFileChange
from app.services.github_client import GitHubAPIError
from app.services.github_write_client import (
    GitHubBranchExistsError,
    GitHubWriteClient,
    GitHubWriteConfigurationError,
    TreeFileEntry,
    validate_branch_name,
)
from app.services.patch_service import check_for_staleness


async def apply_patch_node(state: ProposalState, store: RepositoryStore, settings: Settings) -> dict:
    repository = store.get_current()
    if repository is None or f"{repository.owner}/{repository.repo}" != state["repository_id"]:
        return {
            "status": "FAILED",
            "error": "The currently ingested repository no longer matches this proposal's repository.",
        }

    changes = state["proposed_changes"]
    staleness_issues = check_for_staleness(changes, repository)
    if staleness_issues:
        return {
            "status": "FAILED",
            "error": "Proposal is stale: " + "; ".join(issue.message for issue in staleness_issues),
        }

    proposal_id = state["proposal_id"]
    owner, repo, default_branch = repository.owner, repository.repo, repository.default_branch

    # Cheapest, purely-local check first: never construct a client (even
    # though construction itself makes no network call) for input that's
    # already known to be invalid.
    try:
        branch_name = validate_branch_name(f"devpilot/proposal-{proposal_id}")
    except ValueError as exc:
        return {"status": "FAILED", "error": f"Could not construct a valid branch name: {exc}"}

    try:
        client = GitHubWriteClient(
            token=settings.github_write_token,
            timeout_seconds=settings.github_write_timeout_seconds,
            bot_name=settings.github_bot_name,
            bot_email=settings.github_bot_email,
        )
    except GitHubWriteConfigurationError as exc:
        return {"status": "FAILED", "error": str(exc)}

    commit_message = _build_commit_message(proposal_id, state, changes)
    pr_title, pr_body = _build_pull_request_content(proposal_id, state, changes)

    async with client:
        try:
            base_commit_sha = await client.get_branch_head_commit_sha(owner, repo, default_branch)
        except GitHubAPIError as exc:
            return {
                "status": "FAILED",
                "error": f"Could not resolve the default branch's current commit: {exc}",
            }

        try:
            await client.create_branch(owner, repo, branch_name, base_commit_sha)
        except GitHubBranchExistsError:
            reusable, reuse_error = await _verify_existing_branch_reusable(
                client, owner, repo, branch_name, proposal_id, base_commit_sha
            )
            if not reusable:
                return {
                    "status": "FAILED",
                    "error": reuse_error,
                    "apply_result": ApplyResult(
                        branch_name=branch_name,
                        base_branch=default_branch,
                        base_commit_sha=base_commit_sha,
                        error=reuse_error,
                    ),
                }
            # Verified: the existing branch's tip commit already references
            # this exact proposal and is based on the current default-branch
            # head -- safe to treat as already created and continue.
        except GitHubAPIError as exc:
            return {
                "status": "FAILED",
                "error": f"Failed to create branch '{branch_name}': {exc}",
                "apply_result": ApplyResult(
                    branch_name=branch_name,
                    base_branch=default_branch,
                    base_commit_sha=base_commit_sha,
                    error=str(exc),
                ),
            }

        # Past this point the branch demonstrably exists on GitHub -- any
        # failure from here on is a partial application (APPLY_FAILED), never
        # a plain FAILED, and apply_result must say exactly how far we got.

        try:
            base_commit = await client.get_commit(owner, repo, base_commit_sha)
            base_tree_sha = base_commit["tree"]["sha"]
        except GitHubAPIError as exc:
            return _partial_failure(
                branch_name, default_branch, base_commit_sha, None,
                f"Branch '{branch_name}' exists, but the base tree could not be resolved: {exc}",
            )

        try:
            blob_entries = []
            for change in changes:
                blob_sha = await client.create_blob(owner, repo, change.proposed_content)
                blob_entries.append(TreeFileEntry(path=change.file_path, blob_sha=blob_sha))
            tree_sha = await client.create_tree(owner, repo, base_tree_sha, blob_entries)
        except GitHubAPIError as exc:
            return _partial_failure(
                branch_name, default_branch, base_commit_sha, None,
                f"Branch '{branch_name}' exists, but writing file content failed: {exc}",
            )

        try:
            commit_sha = await client.create_commit(owner, repo, commit_message, tree_sha, base_commit_sha)
        except GitHubAPIError as exc:
            return _partial_failure(
                branch_name, default_branch, base_commit_sha, None,
                f"Branch '{branch_name}' exists, but creating the commit failed: {exc}",
            )

        try:
            await client.update_branch_reference(owner, repo, branch_name, commit_sha)
        except GitHubAPIError as exc:
            return _partial_failure(
                branch_name, default_branch, base_commit_sha, commit_sha,
                f"Commit {commit_sha} was created, but moving branch '{branch_name}' to it failed: {exc}",
            )

        try:
            pr = await client.create_pull_request(owner, repo, pr_title, pr_body, branch_name, default_branch)
        except GitHubAPIError as exc:
            return _partial_failure(
                branch_name, default_branch, base_commit_sha, commit_sha,
                f"Branch '{branch_name}' was pushed (commit {commit_sha}), but pull request creation "
                f"failed: {exc}",
            )

    return {
        "status": "APPLIED",
        "apply_result": ApplyResult(
            branch_name=branch_name,
            base_branch=default_branch,
            base_commit_sha=base_commit_sha,
            commit_sha=commit_sha,
            pull_request_number=pr.number,
            pull_request_url=pr.url,
            pull_request_state=pr.state,
            succeeded=True,
        ),
    }


def _partial_failure(
    branch_name: str, base_branch: str, base_commit_sha: str, commit_sha: str | None, error: str
) -> dict:
    return {
        "status": "APPLY_FAILED",
        "error": error,
        "apply_result": ApplyResult(
            branch_name=branch_name,
            base_branch=base_branch,
            base_commit_sha=base_commit_sha,
            commit_sha=commit_sha,
            error=error,
        ),
    }


async def _verify_existing_branch_reusable(
    client: GitHubWriteClient,
    owner: str,
    repo: str,
    branch_name: str,
    proposal_id: str,
    expected_base_commit_sha: str,
) -> tuple[bool, str | None]:
    """
    A pre-existing branch with this exact (deterministic) name is only safe
    to treat as "already done" if its tip commit (a) demonstrably belongs to
    this proposal and (b) is still based on the default branch's current
    head. Anything else is refused -- never blindly reused, never silently
    rewritten.
    """
    try:
        existing_head_sha = await client.get_branch_head_commit_sha(owner, repo, branch_name)
        commit = await client.get_commit(owner, repo, existing_head_sha)
    except GitHubAPIError as exc:
        return False, f"Branch '{branch_name}' already exists but could not be inspected: {exc}"

    message = commit.get("message") or ""
    parent_shas = {parent.get("sha") for parent in commit.get("parents", [])}

    if proposal_id not in message:
        return False, (
            f"Branch '{branch_name}' already exists but its tip commit does not reference "
            f"proposal {proposal_id}. Refusing to reuse an unrelated branch."
        )
    if expected_base_commit_sha not in parent_shas:
        return False, (
            f"Branch '{branch_name}' already exists but is not based on the default branch's "
            "current head -- the default branch has moved since the branch was created. "
            "Refusing to reuse it automatically; manual cleanup is required."
        )
    return True, None


def _build_commit_message(proposal_id: str, state: ProposalState, changes: list[ProposedFileChange]) -> str:
    draft = state.get("proposal_draft")
    summary = draft.explanation if draft else "Automated fix generated by DevPilot."
    files_line = "\n".join(f"- {change.file_path}: {change.change_summary}" for change in changes)
    return f"DevPilot: {summary}\n\n{files_line}\n\nDevPilot-Proposal-Id: {proposal_id}"


def _build_pull_request_content(
    proposal_id: str, state: ProposalState, changes: list[ProposedFileChange]
) -> tuple[str, str]:
    draft = state.get("proposal_draft")
    report = state.get("investigation_report")

    explanation = draft.explanation if draft else "Automated fix generated by DevPilot."
    risk_level = draft.risk_level if draft else "unknown"
    assumptions = draft.assumptions if draft else []
    root_cause = report.root_cause if report else "(not available)"

    title = f"DevPilot: {explanation}".strip()
    if len(title) > 120:
        title = title[:117] + "..."

    files_section = "\n".join(f"- `{change.file_path}` — {change.change_summary}" for change in changes)
    assumptions_section = "\n".join(f"- {a}" for a in assumptions) or "(none stated)"

    body = (
        f"**Root cause:** {root_cause}\n\n"
        f"**Fix:** {explanation}\n\n"
        f"**Risk level:** {risk_level}\n\n"
        f"**Assumptions:**\n{assumptions_section}\n\n"
        f"**Files changed:**\n{files_section}\n\n"
        "---\n"
        f"Generated by DevPilot from proposal `{proposal_id}` after human approval. "
        "This pull request was not merged automatically -- it requires normal repository review."
    )
    return title, body
