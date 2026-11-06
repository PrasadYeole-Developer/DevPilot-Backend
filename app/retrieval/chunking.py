"""
Splits ingested source files into overlapping, line-aligned chunks.

This is deliberately a character-based strategy (not a code-aware AST parser) —
per Milestone 3 scope, that's acceptable for now. Chunks always break on line
boundaries (never mid-line), and when a chunk has to be cut short, the cut
point prefers the nearest blank line so logical blocks aren't split exactly in
half. Original file content is untouched — it stays in the repository store
as-is for future code analysis; chunking only reads it.
"""

from dataclasses import dataclass

from app.ingestion.store import RetrievedFile


@dataclass(frozen=True)
class CodeChunk:
    path: str
    chunk_index: int
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    text: str


def chunk_file(path: str, content: str, chunk_size: int, chunk_overlap: int) -> list[CodeChunk]:
    """Chunk a single file's content. chunk_size/chunk_overlap are in characters."""
    lines = content.splitlines()
    if not lines:
        return []

    chunks: list[CodeChunk] = []
    start_idx = 0
    line_count = len(lines)
    chunk_index = 0

    while start_idx < line_count:
        end_idx, blank_boundary = _find_chunk_end(lines, start_idx, chunk_size)

        # Prefer ending at the last blank line in range, so we don't cut a
        # function/class body in half — but only if it still leaves a
        # reasonably sized chunk (and we actually had more content to cut).
        if (
            end_idx < line_count
            and blank_boundary is not None
            and start_idx < blank_boundary < end_idx - 1
        ):
            end_idx = blank_boundary + 1

        chunk_lines = lines[start_idx:end_idx]
        chunks.append(
            CodeChunk(
                path=path,
                chunk_index=chunk_index,
                start_line=start_idx + 1,
                end_line=end_idx,
                text="\n".join(chunk_lines),
            )
        )
        chunk_index += 1

        if end_idx >= line_count:
            break

        start_idx = _next_start(lines, start_idx, end_idx, chunk_overlap)

    return chunks


def _find_chunk_end(lines: list[str], start_idx: int, chunk_size: int) -> tuple[int, int | None]:
    """Walk forward from start_idx accumulating whole lines up to ~chunk_size chars."""
    char_count = 0
    end_idx = start_idx
    last_blank_idx: int | None = None
    line_count = len(lines)

    while end_idx < line_count:
        line_len = len(lines[end_idx]) + 1  # +1 for the newline
        if char_count + line_len > chunk_size and end_idx > start_idx:
            break
        char_count += line_len
        if lines[end_idx].strip() == "":
            last_blank_idx = end_idx
        end_idx += 1

    return end_idx, last_blank_idx


def _next_start(lines: list[str], start_idx: int, end_idx: int, chunk_overlap: int) -> int:
    """Step back from end_idx by ~chunk_overlap chars worth of whole lines, guaranteeing progress."""
    overlap_chars = 0
    overlap_start = end_idx
    while overlap_start > start_idx and overlap_chars < chunk_overlap:
        overlap_start -= 1
        overlap_chars += len(lines[overlap_start]) + 1
    return max(overlap_start, start_idx + 1)


def chunk_files(files: list[RetrievedFile], chunk_size: int, chunk_overlap: int) -> list[CodeChunk]:
    """Chunk every retrieved file, preserving each file's relative path."""
    all_chunks: list[CodeChunk] = []
    for file in files:
        all_chunks.extend(chunk_file(file.path, file.content, chunk_size, chunk_overlap))
    return all_chunks
