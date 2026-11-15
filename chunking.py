"""Page-aware text chunking.

Chunks never cross page boundaries, so every retrieved passage maps to exactly one page
of the source PDF. That is what makes page-level citations reliable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .pdf import Page

MIN_CHUNK_CHARS = 20

_SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?…])\s+(?=[A-ZÀ-Ý0-9“"(\[])')


@dataclass(frozen=True)
class Chunk:
    index: int  # position within the document
    page: int
    text: str


def _length(parts: list[str]) -> int:
    return sum(len(p) for p in parts) + max(len(parts) - 1, 0)


def _hard_split(unit: str, size: int) -> list[str]:
    """Split a unit that is longer than ``size`` on word boundaries."""
    pieces: list[str] = []
    current = ""
    for word in unit.split(" "):
        if len(word) > size:  # pathological token (URL, base64...)
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(word[i : i + size] for i in range(0, len(word), size))
        elif current and len(current) + 1 + len(word) > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        pieces.append(current)
    return pieces


def split_units(text: str, size: int) -> list[str]:
    """Break text into sentence-like units, none longer than ``size``."""
    units: list[str] = []
    for paragraph in text.split("\n\n"):
        for sentence in _SENTENCE_BOUNDARY.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            units.extend(_hard_split(sentence, size) if len(sentence) > size else [sentence])
    return units


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Greedily pack sentences into chunks of at most ``size`` characters.

    Consecutive chunks share up to ``overlap`` characters of trailing sentences, so an
    answer that straddles a boundary is still fully contained in one of them.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    chunks: list[str] = []
    current: list[str] = []
    for unit in split_units(text, size):
        if current and _length(current) + 1 + len(unit) > size:
            chunks.append(" ".join(current))
            tail: list[str] = []
            for previous in reversed(current):
                if _length([previous, *tail]) > overlap:
                    break
                tail.insert(0, previous)
            current = tail if _length(tail) + 1 + len(unit) <= size else []
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages: Iterable[Page], size: int = 900, overlap: int = 150) -> list[Chunk]:
    """Chunk every page, tagging each chunk with its page number."""
    chunks: list[Chunk] = []
    for page in pages:
        for text in chunk_text(page.text, size, overlap):
            if len(text) >= MIN_CHUNK_CHARS:  # drop page numbers and stray fragments
                chunks.append(Chunk(index=len(chunks), page=page.number, text=text))
    return chunks
