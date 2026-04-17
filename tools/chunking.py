"""
tools/chunking.py -- Paragraph-Level Chunking
===============================================
Content-aware paragraph-window chunking with configurable size and overlap.
Stop words are excluded from the character budget so that the chunk size
limit reflects meaningful content rather than filler.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict


PARA_CHUNK_SIZE = 2_500
PARA_CHUNK_OVERLAP = 400

_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "that", "this",
    "these", "those", "it", "its", "their", "they", "he", "she", "we",
    "i", "you", "not", "no", "nor", "so", "yet", "both", "either",
    "neither", "whether", "if", "while", "when", "where", "which", "who",
    "whom", "whose", "what", "how", "all", "any", "each", "every", "few",
    "more", "most", "other", "some", "such", "than", "then", "there",
    "thus", "under", "upon", "after", "before", "between", "into",
    "through", "during", "within", "about", "against", "above", "below",
    "also", "per", "ref", "ibid", "para", "pp", "et", "al",
})


def _content_length(text: str) -> int:
    """Character length of text excluding stop words (for chunk boundary decisions)."""
    tokens = re.findall(r"\S+", text.lower())
    return sum(len(t) for t in tokens if t.strip(".,;:!?()[]{}\"'") not in _STOP_WORDS)


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    chunk_index: int = 0
    char_start: int = 0
    char_end: int = 0
    ai_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = PARA_CHUNK_SIZE,
    overlap: int = PARA_CHUNK_OVERLAP,
) -> list[ChunkRecord]:
    """
    Split text into overlapping paragraph-window chunks.
    Uses content-aware length (excluding stop words) for boundary decisions,
    so chunks contain ~chunk_size characters of meaningful content.
    Returns a list of ChunkRecord objects.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[ChunkRecord] = []
    current_parts: list[str] = []
    current_content_len = 0

    def _emit(parts: list[str]) -> None:
        chunk_text_joined = "\n\n".join(parts)
        search_from = max(0, chunks[-1].char_end - 50) if chunks else 0
        start = text.find(parts[0], search_from)
        if start == -1:
            start = chunks[-1].char_end if chunks else 0
        chunks.append(
            ChunkRecord(
                chunk_id=f"{doc_id}_{len(chunks)}",
                text=chunk_text_joined,
                chunk_index=len(chunks),
                char_start=start,
                char_end=start + len(chunk_text_joined),
            )
        )

    for para in paragraphs:
        para_content_len = _content_length(para)
        if current_content_len + para_content_len > chunk_size and current_parts:
            _emit(current_parts)
            overlap_parts: list[str] = []
            overlap_content_len = 0
            for part in reversed(current_parts):
                part_cl = _content_length(part)
                if overlap_content_len + part_cl <= overlap:
                    overlap_parts.insert(0, part)
                    overlap_content_len += part_cl
                else:
                    break
            current_parts = overlap_parts
            current_content_len = overlap_content_len
        current_parts.append(para)
        current_content_len += para_content_len

    if current_parts:
        _emit(current_parts)

    return chunks
