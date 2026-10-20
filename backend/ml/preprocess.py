"""Text preprocessing, chunking, and anti-leakage guards for CUAD and contracts."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


class LeakageError(Exception):
    """Raised when training/fitting operations are attempted on non-training splits."""
    pass


@dataclass
class TextChunk:
    text: str
    char_start: int
    char_end: int
    contract_id: str
    clause_type: str | None = None
    span_text: str | None = None
    span_start: int | None = None
    span_end: int | None = None


# Patterns for headers, footers, page numbers
_HEADER_FOOTER_PATTERNS = [
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*confidential\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*[-—–]+\s*\d+\s*[-—–]+\s*$", re.MULTILINE),
]


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters using NFKC decomposition/composition."""
    return unicodedata.normalize("NFKC", text)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of horizontal spaces/tabs, preserving newlines."""
    # Replace runs of horizontal whitespace with single space
    lines = text.splitlines()
    cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    # Remove consecutive empty lines (more than 2)
    result: list[str] = []
    empty_count = 0
    for line in cleaned_lines:
        if not line:
            empty_count += 1
            if empty_count <= 1:
                result.append(line)
        else:
            empty_count = 0
            result.append(line)
    return "\n".join(result)


def remove_headers_footers(text: str) -> str:
    """Heuristic removal of common header and footer artifacts."""
    cleaned = text
    for pattern in _HEADER_FOOTER_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


def clean_contract_text(text: str) -> str:
    """Full preprocessing pipeline: unicode normalization, header removal, whitespace collapse."""
    text = normalize_unicode(text)
    text = remove_headers_footers(text)
    text = collapse_whitespace(text)
    return text


def chunk_text(
    text: str,
    contract_id: str = "",
    window_chars: int = 1500,
    stride_chars: int = 300,
) -> list[TextChunk]:
    """Slide a window across *text* with *stride_chars* overlap.

    Returns TextChunk objects carrying exact character offsets relative to original *text*.
    """
    if not text:
        return []

    chunks: list[TextChunk] = []
    text_len = len(text)
    start = 0

    while start < text_len:
        end = min(start + window_chars, text_len)

        # Snap to word boundary if not at end of text
        if end < text_len:
            boundary = text.rfind(" ", start + stride_chars, end)
            if boundary != -1 and boundary > start:
                end = boundary

        chunk_str = text[start:end]
        chunks.append(
            TextChunk(
                text=chunk_str,
                char_start=start,
                char_end=end,
                contract_id=contract_id,
            )
        )

        if end >= text_len:
            break
        start += stride_chars

    return chunks


class GuardedSplitContext:
    """Context guard to prevent vectorizers or models from fitting on validation or test sets."""

    def __init__(self, split: str) -> None:
        self.split = split

    def __enter__(self) -> GuardedSplitContext:
        if self.split != "train":
            raise LeakageError(
                f"Model training / fitting is only permitted on 'train' split! Attempted on: {self.split!r}"
            )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
