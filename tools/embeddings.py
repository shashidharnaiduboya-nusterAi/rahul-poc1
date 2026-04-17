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
