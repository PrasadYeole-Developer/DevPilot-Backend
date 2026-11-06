"""
Decides which files discovered in a repository tree are worth retrieving.

Filtering happens entirely from tree metadata (path, size) — no content is
fetched yet at this stage — so excluded files never cost an extra API call.
"""

from dataclasses import dataclass

ALLOWED_EXTENSIONS: set[str] = {
    ".py",
    ".java",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".markdown",
    ".sql",
    ".properties",
    ".xml",  # common for Java/Maven (pom.xml) and Android projects
}

EXCLUDED_DIR_NAMES: set[str] = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "dist",
    "build",
    "coverage",
    "__pycache__",
}

EXCLUDED_LOCKFILE_NAMES: set[str] = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Pipfile.lock",
    "composer.lock",
    "Gemfile.lock",
}


@dataclass(frozen=True)
class TreeEntry:
    path: str
    sha: str
    size: int


@dataclass(frozen=True)
class SkippedEntry:
    path: str
    reason: str


def _is_in_excluded_dir(path: str) -> bool:
    segments = path.split("/")
    return any(segment in EXCLUDED_DIR_NAMES for segment in segments[:-1])


def _extension_of(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def classify_entry(entry: TreeEntry, max_file_size_bytes: int) -> str | None:
    """
    Return a skip reason if the entry should be excluded, or None if it's eligible.
    """
    filename = entry.path.rsplit("/", 1)[-1]

    if _is_in_excluded_dir(entry.path):
        return "excluded_directory"

    if filename in EXCLUDED_LOCKFILE_NAMES:
        return "lockfile"

    if ".min." in filename.lower():
        return "generated_or_minified_file"

    extension = _extension_of(filename)
    if extension not in ALLOWED_EXTENSIONS:
        return "unsupported_file_type"

    if entry.size > max_file_size_bytes:
        return "file_too_large"

    return None


def filter_tree_entries(
    entries: list[TreeEntry],
    max_file_size_bytes: int,
    max_files: int,
) -> tuple[list[TreeEntry], list[SkippedEntry]]:
    """
    Split tree entries into (eligible, skipped). Eligible entries are capped at
    max_files — any overflow is reported as skipped rather than silently dropped.
    """
    eligible: list[TreeEntry] = []
    skipped: list[SkippedEntry] = []

    for entry in entries:
        reason = classify_entry(entry, max_file_size_bytes)
        if reason is None:
            eligible.append(entry)
        else:
            skipped.append(SkippedEntry(path=entry.path, reason=reason))

    if len(eligible) > max_files:
        overflow = eligible[max_files:]
        eligible = eligible[:max_files]
        skipped.extend(
            SkippedEntry(path=entry.path, reason="max_file_count_exceeded") for entry in overflow
        )

    return eligible, skipped
