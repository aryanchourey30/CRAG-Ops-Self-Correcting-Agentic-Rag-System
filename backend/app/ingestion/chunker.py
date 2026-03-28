"""Semantic chunking utilities."""

from __future__ import annotations

from typing import Iterable


def _window_text(tokens: list[str], size: int, overlap: int) -> Iterable[str]:
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + size)
        yield " ".join(tokens[start:end]).strip()
        if end >= len(tokens):
            break
        start = max(end - overlap, start + 1)


def semantic_chunk(text: str, *, chunk_size: int = 220, overlap: int = 40) -> list[str]:
    """Split text into semantically reasonable chunks using paragraph first, then windows."""

    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []

    for paragraph in paragraphs:
        tokens = paragraph.split()
        if len(tokens) <= chunk_size:
            chunks.append(paragraph)
            continue
        chunks.extend(chunk for chunk in _window_text(tokens, chunk_size, overlap) if chunk)

    return chunks
