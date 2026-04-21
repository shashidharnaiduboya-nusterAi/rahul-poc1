"""
tools/embeddings.py -- Embedding Utilities
===========================================
Wrapper around SentenceTransformers (Qwen3-Embedding-0.6B) for encoding text
and Qdrant for vector search operations.

Qwen3-Embedding-0.6B produces 1024-dim vectors and supports instruction-aware
encoding: query texts benefit from the built-in "query" prompt while documents
are encoded without a prompt.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBED_DIM = 1024

_model: Optional[SentenceTransformer] = None
_qdrant: Optional[QdrantClient] = None


def get_embed_model() -> SentenceTransformer:
    global _model
    if _model is None:
        hf_token = os.getenv("HF_TOKEN", None)
        _model = SentenceTransformer(
            EMBED_MODEL_NAME,
            trust_remote_code=True,
            token=hf_token,
        )
    return _model


def get_qdrant(path: Optional[str] = None) -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        qdrant_path = path or os.getenv("QDRANT_PATH", "data/qdrant")
        Path(qdrant_path).mkdir(parents=True, exist_ok=True)
        _qdrant = QdrantClient(path=qdrant_path)
    return _qdrant


# ---------------------------------------------------------------------------
# Document encoding (no query prompt -- used for PG docs, case content, etc.)
# ---------------------------------------------------------------------------
def encode_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode a list of document texts. Returns numpy array of shape (N, 1024)."""
    model = get_embed_model()
    return model.encode(texts, show_progress_bar=False, batch_size=batch_size)


def encode_single(text: str) -> list[float]:
    """Encode a single document text string. Returns a list of floats."""
    model = get_embed_model()
    return model.encode(text[:16_000]).tolist()


# ---------------------------------------------------------------------------
# Query encoding (uses Qwen3 "query" prompt for better retrieval accuracy)
# ---------------------------------------------------------------------------
def encode_texts_as_query(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode a list of query texts with the 'query' prompt. Returns (N, 1024)."""
    model = get_embed_model()
    return model.encode(
        texts, prompt_name="query", show_progress_bar=False, batch_size=batch_size
    )


def encode_single_as_query(text: str) -> list[float]:
    """Encode a single query text with the 'query' prompt. Returns list of floats."""
    model = get_embed_model()
    return model.encode(text[:16_000], prompt_name="query").tolist()


def encode_long_text_as_query(
    text: str,
    window_chars: int = 8_000,
    overlap_chars: int = 1_000,
    max_windows: int = 12,
) -> list[float]:
    """
    Encode a long document as a single pooled query vector.

    Strategy:
      1. Split ``text`` into overlapping windows of ``window_chars`` characters
         with ``overlap_chars`` overlap, cap at ``max_windows`` to bound cost.
      2. Encode each window with the 'query' prompt via
         :func:`encode_texts_as_query`.
      3. L2-normalise each window vector, mean-pool, then L2-normalise again
         so cosine search behaves well.

    Returns a list of ``EMBED_DIM`` floats.  Falls back to
    :func:`encode_single_as_query` when the text fits in a single window.
    """
    text = text or ""
    if not text:
        return [0.0] * EMBED_DIM

    if len(text) <= window_chars:
        return encode_single_as_query(text)

    stride = max(window_chars - overlap_chars, 1)
    windows: list[str] = []
    i = 0
    while i < len(text) and len(windows) < max_windows:
        windows.append(text[i: i + window_chars])
        i += stride

    vecs = encode_texts_as_query(windows)  # shape (N, D)
    # L2 normalise each row
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = vecs / norms
    pooled = normed.mean(axis=0)
    pooled_norm = np.linalg.norm(pooled)
    if pooled_norm == 0:
        return pooled.tolist()
    return (pooled / pooled_norm).tolist()
