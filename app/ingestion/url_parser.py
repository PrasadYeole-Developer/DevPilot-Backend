"""Parsing and validation for GitHub repository URLs."""

import re

_HTTPS_PATTERN = re.compile(
    r"^https?://(www\.)?github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(\.git)?/?$"
)
_SSH_PATTERN = re.compile(
    r"^git@github\.com:(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(\.git)?$"
)


class InvalidRepositoryURLError(ValueError):
    """Raised when a repository_url is not a well-formed, supported GitHub URL."""


def parse_github_url(repository_url: str) -> tuple[str, str]:
    """
    Extract (owner, repo) from a GitHub repository URL.

    Supports https://github.com/<owner>/<repo> (with or without .git / trailing slash)
    and the git@github.com:<owner>/<repo>.git SSH form. Anything else — other hosts,
    malformed URLs, URLs pointing at a sub-path like /blob/main/file.py — is rejected.
    """
    if not repository_url or not repository_url.strip():
        raise InvalidRepositoryURLError("repository_url must not be empty")

    candidate = repository_url.strip()

    match = _HTTPS_PATTERN.match(candidate) or _SSH_PATTERN.match(candidate)
    if not match:
        raise InvalidRepositoryURLError(
            "Unsupported or malformed GitHub repository URL. "
            "Expected format: https://github.com/<owner>/<repository>"
        )

    return match.group("owner"), match.group("repo")
