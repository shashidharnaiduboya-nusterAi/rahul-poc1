"""
tools/retrieval.py -- Hybrid Retrieval with Reciprocal Rank Fusion (RRF)
=========================================================================
Three-level hybrid retrieval against two Qdrant collections:

    pg_doc_index   -- one vector per PG document      (L0, L1 target)
    pg_chunks      -- paragraph/section vectors        (L2 target)

Signals fused with **Reciprocal Rank Fusion** (weighted):

    Level 0  : case summary embedding  vs  pg_doc_index       (broad sweep)
    Level 1  : full-doc + retrieval-profile embeddings vs pg_doc_index
    Level 2  : each case paragraph embedding vs pg_chunks
               (per-case-para hits are grouped to parent PG doc; this is the
                only level that actually compares paragraph-to-paragraph)
    BM25     : single global corpus over all PG docs
               (title + summary + practice_area + body sample)
    Citation : alert cite_defs + cite_refs + case cite_ref
               intersected with PG doc cite_ids (both sides normalised)

Each signal contributes ``weight / (k + rank)`` per doc; contributions are
summed to give the final score.  All cut-offs are expressed as cosine
thresholds so individual levels can drop junk before they pollute the ranks
-- but L2 is no longer *stricter* than L1 (was the main recall regression).

Evidence chain: ``matched_paragraphs`` carries the specific case-para / PG-chunk
pairs that triggered each hit, so the downstream MatchingAgent can stay
grounded in real text rather than the whole-doc summary.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from collections import defaultdict
from typing import Optional

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

from tools.logging_setup import get_logger

log = get_logger("retrieval")

# ---------------------------------------------------------------------------
# Collection names
# ---------------------------------------------------------------------------
PG_DOC_COLL = "pg_doc_index"
PG_CHUNK_COLL = "pg_chunks"

# ---------------------------------------------------------------------------
# Default config (overridden by env)
# ---------------------------------------------------------------------------
_DEFAULT_SIM_THRESHOLD = 0.30
_DEFAULT_TOP_K = 25
_DEFAULT_SCORE_GAP = 0.0          # 0 disables gap-cutoff

_DEFAULT_L0_LIMIT = 100
_DEFAULT_L1_LIMIT = 50
_DEFAULT_L2_PER_PARA_LIMIT = 10
_MAX_PARA_QUERIES = 20

# RRF defaults
_DEFAULT_RRF_K = 60
_DEFAULT_W_L0 = 1.0
_DEFAULT_W_L1 = 1.0
_DEFAULT_W_L2 = 1.2
_DEFAULT_W_BM25 = 0.5
_DEFAULT_W_CITE = 1.5

# ---------------------------------------------------------------------------
# Text helpers (shared with chunking / keyword extraction)
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset({
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

_LEGAL_BOOST = frozenset({
    "appellant", "respondent", "claimant", "defendant", "plaintiff",
    "judgment", "judgement", "appeal", "tribunal", "court", "section",
    "act", "regulation", "clause", "contract", "liability", "damages",
    "negligence", "breach", "statute", "statutory", "precedent",
    "jurisdiction", "ruling", "order", "injunction", "remedy",
    "submission", "evidence", "witness", "hearing", "counsel",
    "principle", "doctrine", "authority", "ratio", "obiter",
    "legislation", "provision", "agreement", "obligation", "rights",
    "duty", "standard", "care", "loss", "causation", "quantum",
    "assessment", "award", "relief", "declaration", "enforcement",
})


def kw_tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", (text or "").lower())
    return [t for t in tokens if t not in _STOPWORDS]


def extract_keywords(
    texts: list[str],
    ai_keywords: Optional[list[list[str]]] = None,
    top_n: int = 150,
) -> list[str]:
    """Frequency-based keyword extraction with a small legal-term boost."""
    from collections import Counter

    all_tokens: list[str] = []
    for text in texts:
        all_tokens.extend(kw_tokenize(text))
    if ai_keywords:
        for kw_list in ai_keywords:
            for kw in kw_list:
                all_tokens.extend(kw_tokenize(kw))
    freq = Counter(all_tokens)
    boosted = {k: v * 2 if k in _LEGAL_BOOST else v for k, v in freq.items()}
    return [k for k, _ in sorted(boosted.items(), key=lambda x: -x[1])[:top_n]]


# ---------------------------------------------------------------------------
# Citation normalisation
# ---------------------------------------------------------------------------
_CITATION_STRIP_RE = re.compile(r"[\[\]\(\)\.,;:]")
_CITATION_WS_RE = re.compile(r"\s+")


def normalize_citation(cite: str) -> str:
    """
    Aggressive citation normaliser so formatting drift between alert and PG
    payload doesn't prevent overlap detection.

    Rules:
      - lower-case
      - strip ``#`` (Lexis anchor markers)
      - drop square/round brackets, dots, commas, semicolons, colons
      - collapse whitespace
    """
    if not cite:
        return ""
    s = cite.strip().lower().replace("#", " ")
    s = _CITATION_STRIP_RE.sub(" ", s)
    s = _CITATION_WS_RE.sub(" ", s).strip()
    return s


def _normalise_cite_set(cites) -> set[str]:
    """Return a set of normalised citation strings, tolerating JSON/string input."""
    if isinstance(cites, str):
        try:
            cites = json.loads(cites)
        except (json.JSONDecodeError, TypeError):
            cites = [cites]
    out: set[str] = set()
    for c in cites or []:
        n = normalize_citation(c)
        if n:
            out.add(n)
    return out


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------
def _collection_exists(qdrant: QdrantClient, name: str) -> bool:
    try:
        return name in [c.name for c in qdrant.get_collections().collections]
    except Exception as exc:
        log.error("get_collections failed: %s", exc, extra={"step": "qdrant"})
        return False


def _qdrant_search(
    qdrant: QdrantClient,
    collection: str,
    vector: list[float],
    limit: int,
    score_threshold: float,
) -> list[dict]:
    try:
        results_obj = qdrant.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            score_threshold=score_threshold if score_threshold > 0 else None,
            with_payload=True,
        )
    except Exception as exc:
        log.error("Qdrant error on '%s': %s", collection, exc,
                  extra={"step": "qdrant_search"})
        return []

    results: list[dict] = []
    for hit in results_obj.points:
        p = hit.payload or {}
        results.append({
            "point_id": str(hit.id),
            "doc_id": p.get("doc_id", str(hit.id)),
            "doc_title": p.get("doc_title", ""),
            "cite_ids": p.get("cite_ids", []),
            "practice_area": p.get("practice_area", ""),
            "doc_summary": p.get("doc_summary", ""),
            "section_id": p.get("section_id", ""),
            "heading": p.get("heading", ""),
            "chunk_index": p.get("chunk_index", -1),
            "text_preview": p.get("text_preview", ""),
            "cosine_score": round(float(hit.score), 4),
        })
    return results


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion helper
# ---------------------------------------------------------------------------
def _rrf_contribution(
    ranked_doc_ids: list[str],
    k: int,
    weight: float,
) -> dict[str, float]:
    """
    Return {doc_id -> weight / (k + rank)} for the given ranking.
    If a doc_id appears more than once, the best (smallest) rank wins.
    """
    out: dict[str, float] = {}
    for rank, did in enumerate(ranked_doc_ids, start=1):
        if did in out:
            continue
        out[did] = weight / (k + rank)
    return out


# ---------------------------------------------------------------------------
# Global BM25 cache over the entire PG corpus
# ---------------------------------------------------------------------------
_BM25_CACHE: Optional[dict] = None

_BM25_CACHE_PATH = os.path.join("data", "bm25_pg.pkl")


def _bm25_corpus_fingerprint(doc_count: int) -> str:
    return f"count={doc_count}"


def _load_all_pg_docs_from_qdrant(qdrant: QdrantClient) -> list[dict]:
    """Scroll through ``pg_doc_index`` and collect payloads for BM25 corpus."""
    if not _collection_exists(qdrant, PG_DOC_COLL):
        return []

    docs: list[dict] = []
    offset = None
    PAGE = 256
    while True:
        try:
            points, next_offset = qdrant.scroll(
                collection_name=PG_DOC_COLL,
                limit=PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            log.error("BM25 scroll failed: %s", exc, extra={"step": "bm25_build"})
            break
        for p in points:
            payload = p.payload or {}
            docs.append({
                "doc_id": payload.get("doc_id", str(p.id)),
                "doc_title": payload.get("doc_title", ""),
                "doc_summary": payload.get("doc_summary", ""),
                "practice_area": payload.get("practice_area", ""),
                "jurisdiction": payload.get("jurisdiction", ""),
            })
        offset = next_offset
        if offset is None:
            break
    return docs


def _load_chunk_samples_for_bm25(
    qdrant: QdrantClient,
    per_doc_chars: int = 2_000,
) -> dict[str, str]:
    """
    Concatenate a short sample of each doc's pg_chunks ``text_preview`` fields
    (up to ``per_doc_chars`` per doc) so the global BM25 corpus contains *body*
    text, not just summaries.  Returns {doc_id -> body_sample}.
    """
    if not _collection_exists(qdrant, PG_CHUNK_COLL):
        return {}

    acc: dict[str, list[str]] = defaultdict(list)
    acc_len: dict[str, int] = defaultdict(int)
    offset = None
    PAGE = 512
    while True:
        try:
            points, next_offset = qdrant.scroll(
                collection_name=PG_CHUNK_COLL,
                limit=PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            log.error("BM25 chunk scroll failed: %s", exc,
                      extra={"step": "bm25_build"})
            break
        for p in points:
            payload = p.payload or {}
            did = payload.get("doc_id")
            if not did:
                continue
            if acc_len[did] >= per_doc_chars:
                continue
            snippet = (payload.get("text_preview") or "")[: per_doc_chars - acc_len[did]]
            if snippet:
                acc[did].append(snippet)
                acc_len[did] += len(snippet)
        offset = next_offset
        if offset is None:
            break

    return {did: " ".join(parts) for did, parts in acc.items()}


def _bm25_text_for(doc: dict) -> str:
    return " ".join([
        doc.get("doc_title", ""),
        doc.get("doc_summary", ""),
        doc.get("practice_area", ""),
        doc.get("jurisdiction", ""),
        doc.get("body_sample", ""),
    ])


def _build_global_bm25(qdrant: QdrantClient) -> Optional[dict]:
    """
    Build (or load from disk) a BM25 index over the entire PG corpus.
    Cached in-process; also serialised to ``data/bm25_pg.pkl`` keyed on the
    number of PG docs, so re-ingests transparently invalidate it.
    """
    global _BM25_CACHE
    if _BM25_CACHE is not None:
        return _BM25_CACHE

    t0 = time.perf_counter()
    docs = _load_all_pg_docs_from_qdrant(qdrant)
    if not docs:
        log.warning("BM25 build: no docs in pg_doc_index", extra={"step": "bm25_build"})
        return None

    fingerprint = _bm25_corpus_fingerprint(len(docs))

    # Try disk cache
    try:
        if os.path.exists(_BM25_CACHE_PATH):
            with open(_BM25_CACHE_PATH, "rb") as f:
                disk = pickle.load(f)
            if disk.get("fingerprint") == fingerprint:
                _BM25_CACHE = disk
                log.info(
                    "BM25 cache hit doc_count=%d path=%s elapsed_ms=%.1f",
                    len(docs), _BM25_CACHE_PATH,
                    (time.perf_counter() - t0) * 1000.0,
                    extra={"step": "bm25_build"},
                )
                return _BM25_CACHE
    except Exception as exc:
        log.warning("BM25 cache load failed: %s", exc, extra={"step": "bm25_build"})

    # Enrich with body samples from pg_chunks
    body_samples = _load_chunk_samples_for_bm25(qdrant)
    for d in docs:
        d["body_sample"] = body_samples.get(d["doc_id"], "")

    corpus = [kw_tokenize(_bm25_text_for(d)) for d in docs]

    # Filter docs with empty tokenisations (shouldn't happen, but guard).
    valid = [(d, toks) for d, toks in zip(docs, corpus) if toks]
    if not valid:
        log.warning("BM25 build: no valid tokens", extra={"step": "bm25_build"})
        return None

    valid_docs = [d for d, _ in valid]
    valid_tokens = [toks for _, toks in valid]

    bm25 = BM25Okapi(valid_tokens)

    cache = {
        "fingerprint": fingerprint,
        "docs": valid_docs,
        "bm25": bm25,
        "built_at": time.time(),
    }
    _BM25_CACHE = cache

    try:
        os.makedirs(os.path.dirname(_BM25_CACHE_PATH), exist_ok=True)
        with open(_BM25_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
    except Exception as exc:
        log.warning("BM25 cache write failed: %s", exc, extra={"step": "bm25_build"})

    log.info(
        "BM25 built doc_count=%d with_body=%d elapsed_ms=%.1f",
        len(valid_docs),
        sum(1 for d in valid_docs if d.get("body_sample")),
        (time.perf_counter() - t0) * 1000.0,
        extra={"step": "bm25_build"},
    )
    return cache


def _bm25_rank(keywords: list[str], qdrant: QdrantClient, limit: int = 100) -> list[tuple[str, float]]:
    """
    Return up to ``limit`` (doc_id, score) pairs ranked by BM25 over the global
    PG corpus.  Scores are returned normalised to the max of the current query
    so they live on a [0, 1] scale for logging purposes.
    """
    if not keywords:
        return []
    cache = _build_global_bm25(qdrant)
    if not cache:
        return []

    bm25: BM25Okapi = cache["bm25"]
    docs: list[dict] = cache["docs"]

    raw = bm25.get_scores(keywords)
    max_s = max(raw) if len(raw) > 0 and max(raw) > 0 else 1.0

    pairs = [(d["doc_id"], float(s) / max_s) for d, s in zip(docs, raw) if s > 0]
    pairs.sort(key=lambda x: -x[1])
    return pairs[:limit]


# ---------------------------------------------------------------------------
# Citation overlap scoring
# ---------------------------------------------------------------------------
def _citation_rank(
    candidates: dict[str, dict],
    cite_refs: list[str],
    cite_defs: list[str],
    alert_cite_refs: list[str],
    alert_cite_defs: list[str],
) -> list[tuple[str, int]]:
    """
    Return [(doc_id, overlap_count)] sorted by overlap desc for docs with >=1
    overlap.  Both sides use :func:`normalize_citation`.
    """
    case_cites = (
        _normalise_cite_set(cite_refs)
        | _normalise_cite_set(cite_defs)
        | _normalise_cite_set(alert_cite_refs)
        | _normalise_cite_set(alert_cite_defs)
    )
    if not case_cites:
        return []

    out: list[tuple[str, int]] = []
    for did, doc in candidates.items():
        pg_cites = _normalise_cite_set(doc.get("cite_ids", []))
        overlap = len(pg_cites & case_cites)
        if overlap > 0:
            out.append((did, overlap))
    out.sort(key=lambda x: -x[1])
    return out


# ===========================================================================
# Main entry point
# ===========================================================================
def three_level_retrieve(
    qdrant: QdrantClient,
    case_summary_embedding: Optional[list[float]],
    case_full_doc_embedding: Optional[list[float]],
    doc_embeddings: dict[str, list[float]],
    chunk_embeddings: list[list[float]],
    chunk_metadata: list[dict],
    keywords: list[str],
    case_cite_refs: list[str] = None,
    case_cite_defs: list[str] = None,
    alert_cite_refs: list[str] = None,
    alert_cite_defs: list[str] = None,
    sim_threshold: float = None,
) -> list[dict]:
    """
    Run three-level hybrid retrieval with reciprocal rank fusion.

    Inputs come from :mod:`agents.case_processing`; outputs are consumed by
    :mod:`agents.retrieval` / :mod:`agents.matching`.  Each result entry:

        {
            doc_id, doc_title, score,
            query_hits, para_match_count,
            matched_paragraphs: [
                {chunk_index, cosine_score, text_preview,
                 pg_section_id, pg_heading, pg_text_preview}
            ],
            component_scores: {l0, l1, l2, bm25, citation}  -- RRF component sums
        }
    """
    case_cite_refs = case_cite_refs or []
    case_cite_defs = case_cite_defs or []
    alert_cite_refs = alert_cite_refs or []
    alert_cite_defs = alert_cite_defs or []

    if sim_threshold is None:
        sim_threshold = float(os.getenv("RETRIEVAL_SIM_THRESHOLD",
                                        str(_DEFAULT_SIM_THRESHOLD)))
    top_k = int(os.getenv("RETRIEVAL_TOP_K", str(_DEFAULT_TOP_K)))
    l0_limit = int(os.getenv("RETRIEVAL_L0_LIMIT", str(_DEFAULT_L0_LIMIT)))
    l1_limit = int(os.getenv("RETRIEVAL_L1_LIMIT", str(_DEFAULT_L1_LIMIT)))
    l2_per_para = int(os.getenv("RETRIEVAL_L2_PER_PARA_LIMIT",
                                str(_DEFAULT_L2_PER_PARA_LIMIT)))
    score_gap = float(os.getenv("RETRIEVAL_SCORE_GAP", str(_DEFAULT_SCORE_GAP)))

    rrf_k = int(os.getenv("RRF_K", str(_DEFAULT_RRF_K)))
    w_l0 = float(os.getenv("RRF_W_L0", str(_DEFAULT_W_L0)))
    w_l1 = float(os.getenv("RRF_W_L1", str(_DEFAULT_W_L1)))
    w_l2 = float(os.getenv("RRF_W_L2", str(_DEFAULT_W_L2)))
    w_bm25 = float(os.getenv("RRF_W_BM25", str(_DEFAULT_W_BM25)))
    w_cite = float(os.getenv("RRF_W_CITE", str(_DEFAULT_W_CITE)))

    # L0 relaxed, L2 no longer stricter than L1 (was the recall regression).
    l0_threshold = sim_threshold * 0.85
    l2_threshold = sim_threshold * 0.95

    log.info(
        "config sim=%.3f top_k=%d l0=%d l1=%d l2_per_para=%d "
        "rrf_k=%d weights=(L0=%.2f,L1=%.2f,L2=%.2f,BM25=%.2f,CITE=%.2f) "
        "thresholds=(l0=%.3f,l1=%.3f,l2=%.3f)",
        sim_threshold, top_k, l0_limit, l1_limit, l2_per_para,
        rrf_k, w_l0, w_l1, w_l2, w_bm25, w_cite,
        l0_threshold, sim_threshold, l2_threshold,
        extra={"step": "retrieve_config"},
    )

    if not _collection_exists(qdrant, PG_DOC_COLL):
        log.error("Collection '%s' not found -- run ingest.py first.",
                  PG_DOC_COLL, extra={"step": "retrieve_config"})
        return []

    have_chunk_coll = _collection_exists(qdrant, PG_CHUNK_COLL)
    if not have_chunk_coll:
        log.warning(
            "Collection '%s' not found -- L2 will fall back to '%s'. "
            "Re-run `python3 ingest.py <pg.xml>` (or with --rebuild-chunks) "
            "to enable paragraph-level PG matching.",
            PG_CHUNK_COLL, PG_DOC_COLL,
            extra={"step": "retrieve_config"},
        )

    all_docs: dict[str, dict] = {}
    component_scores: dict[str, dict[str, float]] = defaultdict(
        lambda: {"l0": 0.0, "l1": 0.0, "l2": 0.0, "bm25": 0.0, "citation": 0.0}
    )
    query_hits: dict[str, int] = defaultdict(int)
    max_cosine: dict[str, float] = defaultdict(float)

    # Paragraph evidence: doc_id -> list of evidence dicts
    para_evidence: dict[str, list[dict]] = defaultdict(list)

    # ----------------------------------------------------------------- L0
    t0 = time.perf_counter()
    log.info("=== LEVEL 0: case summary broad sweep ===",
             extra={"step": "retrieve_l0"})
    l0_doc_ids: list[str] = []
    if case_summary_embedding:
        hits = _qdrant_search(qdrant, PG_DOC_COLL, case_summary_embedding,
                              limit=l0_limit, score_threshold=l0_threshold)
        log.info("L0 summary hits=%d threshold=%.3f",
                 len(hits), l0_threshold, extra={"step": "retrieve_l0"})
        for h in hits:
            did = h["doc_id"]
            all_docs.setdefault(did, h)
            query_hits[did] += 1
            max_cosine[did] = max(max_cosine[did], h["cosine_score"])
            l0_doc_ids.append(did)
        _log_top_hits("L0/summary", hits)
    else:
        log.warning("no case_summary_embedding -- skipping L0",
                    extra={"step": "retrieve_l0"})
    rrf_l0 = _rrf_contribution(l0_doc_ids, rrf_k, w_l0)
    for did, s in rrf_l0.items():
        component_scores[did]["l0"] += s

    log.info("L0 done docs=%d elapsed_ms=%.1f",
             len(l0_doc_ids), (time.perf_counter() - t0) * 1000.0,
             extra={"step": "retrieve_l0"})

    # ----------------------------------------------------------------- L1
    t1 = time.perf_counter()
    log.info("=== LEVEL 1: full-doc + retrieval profile refinement ===",
             extra={"step": "retrieve_l1"})

    l1_queries: list[tuple[str, list[float]]] = []
    if case_full_doc_embedding:
        l1_queries.append(("case_full_doc", case_full_doc_embedding))
    for name, emb in (doc_embeddings or {}).items():
        if emb:
            l1_queries.append((name, emb))

    for qname, emb in l1_queries:
        hits = _qdrant_search(qdrant, PG_DOC_COLL, emb,
                              limit=l1_limit, score_threshold=sim_threshold)
        log.info("L1 %s hits=%d threshold=%.3f",
                 qname, len(hits), sim_threshold, extra={"step": "retrieve_l1"})
        doc_ids_this = []
        for h in hits:
            did = h["doc_id"]
            all_docs.setdefault(did, h)
            query_hits[did] += 1
            max_cosine[did] = max(max_cosine[did], h["cosine_score"])
            doc_ids_this.append(did)
        _log_top_hits(f"L1/{qname}", hits)
        rrf_this = _rrf_contribution(doc_ids_this, rrf_k, w_l1)
        for did, s in rrf_this.items():
            component_scores[did]["l1"] += s

    log.info("L1 done total_candidates=%d elapsed_ms=%.1f",
             len(all_docs), (time.perf_counter() - t1) * 1000.0,
             extra={"step": "retrieve_l1"})

    # ----------------------------------------------------------------- L2
    t2 = time.perf_counter()
    log.info("=== LEVEL 2: per-case-paragraph grounding vs %s ===",
             PG_CHUNK_COLL if have_chunk_coll else PG_DOC_COLL,
             extra={"step": "retrieve_l2"})

    l2_collection = PG_CHUNK_COLL if have_chunk_coll else PG_DOC_COLL

    if chunk_embeddings:
        query_indices: list[int] = []
        for i, cm in enumerate(chunk_metadata):
            preview = cm.get("text_preview", "")
            if len(preview) >= 150:
                query_indices.append(i)
        if len(query_indices) > _MAX_PARA_QUERIES:
            step = len(query_indices) / _MAX_PARA_QUERIES
            query_indices = [query_indices[int(i * step)]
                             for i in range(_MAX_PARA_QUERIES)]

        log.info("L2 searching with %d paragraphs (of %d total)",
                 len(query_indices), len(chunk_embeddings),
                 extra={"step": "retrieve_l2"})

        para_hit_count = 0
        for idx in query_indices:
            if idx >= len(chunk_embeddings):
                continue
            emb = chunk_embeddings[idx]
            hits = _qdrant_search(
                qdrant, l2_collection, emb,
                limit=l2_per_para, score_threshold=l2_threshold,
            )

            # Collapse hits to per-doc best cosine for ranking
            best_per_doc: dict[str, dict] = {}
            for h in hits:
                did = h["doc_id"]
                prev = best_per_doc.get(did)
                if prev is None or h["cosine_score"] > prev["cosine_score"]:
                    best_per_doc[did] = h

            ranked_doc_ids = [
                h["doc_id"]
                for h in sorted(best_per_doc.values(),
                                key=lambda x: -x["cosine_score"])
            ]
            rrf_this = _rrf_contribution(ranked_doc_ids, rrf_k, w_l2)
            for did, s in rrf_this.items():
                component_scores[did]["l2"] += s

            cm = chunk_metadata[idx] if idx < len(chunk_metadata) else {}
            for h in best_per_doc.values():
                did = h["doc_id"]
                # Preserve doc payload from whatever collection first revealed
                # it -- don't overwrite pg_doc_index payload with a chunk one.
                if did not in all_docs:
                    all_docs[did] = {
                        "doc_id": did,
                        "doc_title": h.get("doc_title", ""),
                        "cite_ids": h.get("cite_ids", []),
                        "practice_area": h.get("practice_area", ""),
                        "doc_summary": h.get("doc_summary", ""),
                        "cosine_score": h["cosine_score"],
                    }
                query_hits[did] += 1
                max_cosine[did] = max(max_cosine[did], h["cosine_score"])

                para_evidence[did].append({
                    "chunk_index": idx,
                    "cosine_score": h["cosine_score"],
                    "text_preview": cm.get("text_preview", "")[:400],
                    "pg_section_id": h.get("section_id", ""),
                    "pg_heading": h.get("heading", ""),
                    "pg_text_preview": h.get("text_preview", "")[:400],
                })

            para_hit_count += len(best_per_doc)

            log.debug(
                "L2 case_chunk=%d hits=%d kept=%d sample=%r",
                idx, len(hits), len(best_per_doc),
                (cm.get("text_preview", "") or "")[:120],
                extra={"step": "retrieve_l2"},
            )

        log.info("L2 total_doc_hits=%d docs_with_evidence=%d elapsed_ms=%.1f",
                 para_hit_count, len(para_evidence),
                 (time.perf_counter() - t2) * 1000.0,
                 extra={"step": "retrieve_l2"})
    else:
        log.warning("no chunk_embeddings -- skipping L2",
                    extra={"step": "retrieve_l2"})

    log.info("total candidates after semantic levels=%d",
             len(all_docs), extra={"step": "retrieve_semantic_done"})

    if not all_docs:
        log.warning("no documents above threshold -- lower RETRIEVAL_SIM_THRESHOLD",
                    extra={"step": "retrieve_done"})
        return []

    # ---------------------------------------------------------------- BM25
    t_bm = time.perf_counter()
    log.info("=== BM25 global keyword boost ===", extra={"step": "retrieve_bm25"})
    bm25_pairs = _bm25_rank(keywords[:80], qdrant, limit=200)
    bm25_doc_ids = [did for did, _ in bm25_pairs]
    rrf_bm25 = _rrf_contribution(bm25_doc_ids, rrf_k, w_bm25)
    for did, s in rrf_bm25.items():
        component_scores[did]["bm25"] += s
        # BM25 can surface docs not found semantically: make sure they still
        # have a home in ``all_docs`` for final formatting.
        if did not in all_docs:
            cache = _build_global_bm25(qdrant)
            doc_map = {d["doc_id"]: d for d in (cache["docs"] if cache else [])}
            meta = doc_map.get(did, {})
            all_docs[did] = {
                "doc_id": did,
                "doc_title": meta.get("doc_title", ""),
                "cite_ids": [],
                "practice_area": meta.get("practice_area", ""),
                "doc_summary": meta.get("doc_summary", ""),
                "cosine_score": 0.0,
            }

    log.info("BM25 contributed to %d docs keywords_used=%d elapsed_ms=%.1f",
             len(rrf_bm25), min(len(keywords), 80),
             (time.perf_counter() - t_bm) * 1000.0,
             extra={"step": "retrieve_bm25"})
    if bm25_pairs[:5]:
        log.debug("BM25 top5=%s", bm25_pairs[:5], extra={"step": "retrieve_bm25"})

    # ------------------------------------------------------------- Citation
    t_c = time.perf_counter()
    cite_ranks = _citation_rank(
        all_docs, case_cite_refs, case_cite_defs,
        alert_cite_refs, alert_cite_defs,
    )
    cite_doc_ids = [did for did, _ in cite_ranks]
    rrf_cite = _rrf_contribution(cite_doc_ids, rrf_k, w_cite)
    for did, s in rrf_cite.items():
        component_scores[did]["citation"] += s
    log.info(
        "citation overlap docs=%d alert_cites=(refs=%d,defs=%d) "
        "case_cites=(refs=%d,defs=%d) elapsed_ms=%.1f",
        len(cite_ranks), len(alert_cite_refs), len(alert_cite_defs),
        len(case_cite_refs), len(case_cite_defs),
        (time.perf_counter() - t_c) * 1000.0,
        extra={"step": "retrieve_citation"},
    )
    if cite_ranks[:5]:
        log.debug("citation top5=%s", cite_ranks[:5],
                  extra={"step": "retrieve_citation"})

    # ----------------------------------------------------------------- Final
    min_query_hits = int(os.getenv("RETRIEVAL_MIN_HITS", "1"))

    results_list: list[dict] = []
    for doc_id, doc in all_docs.items():
        cs = component_scores[doc_id]
        total = cs["l0"] + cs["l1"] + cs["l2"] + cs["bm25"] + cs["citation"]
        if total <= 0:
            continue

        q_hits = query_hits[doc_id]
        # min_hits still honoured, but citation+BM25 alone can produce 0 q_hits
        # (no semantic match); we allow citation-only hits through because
        # strong citation overlap is a powerful recall signal.
        if q_hits < min_query_hits and cs["citation"] <= 0 and cs["bm25"] <= 0:
            continue

        raw_evidence = para_evidence.get(doc_id, [])
        best_per_chunk: dict[int, dict] = {}
        for ev in raw_evidence:
            ci = ev["chunk_index"]
            if (ci not in best_per_chunk
                    or ev["cosine_score"] > best_per_chunk[ci]["cosine_score"]):
                best_per_chunk[ci] = ev

        matched_paras = sorted(best_per_chunk.values(),
                               key=lambda x: -x["cosine_score"])

        results_list.append({
            "doc_id": doc_id,
            "doc_title": doc.get("doc_title", ""),
            "score": round(total, 6),
            "max_cosine": round(max_cosine[doc_id], 4),
            "query_hits": q_hits,
            "matched_paragraphs": matched_paras,
            "para_match_count": len(matched_paras),
            "component_scores": {k: round(v, 6) for k, v in cs.items()},
            # Carry a little PG payload through so MatchingAgent doesn't have
            # to re-query Qdrant just for the doc_summary / title.
            "pg_doc_summary": doc.get("doc_summary", ""),
            "practice_area": doc.get("practice_area", ""),
        })

    results_list.sort(key=lambda d: -d["score"])
    results_list = _apply_score_gap_cutoff(results_list, score_gap)

    if len(results_list) > top_k:
        log.info("trimming %d -> top_k=%d", len(results_list), top_k,
                 extra={"step": "retrieve_done"})
        results_list = results_list[:top_k]

    log.info("final results=%d", len(results_list),
             extra={"step": "retrieve_done"})
    if results_list:
        log.info(
            "score range=%.4f -- %.4f  docs_with_para_evidence=%d",
            results_list[-1]["score"], results_list[0]["score"],
            sum(1 for r in results_list if r["para_match_count"] > 0),
            extra={"step": "retrieve_done"},
        )
        for r in results_list[:10]:
            cs = r["component_scores"]
            log.info(
                "[rank] %s score=%.4f hits=%d paras=%d "
                "L0=%.3f L1=%.3f L2=%.3f BM25=%.3f CITE=%.3f | %s",
                r["doc_id"], r["score"], r["query_hits"], r["para_match_count"],
                cs["l0"], cs["l1"], cs["l2"], cs["bm25"], cs["citation"],
                (r["doc_title"] or "")[:70],
                extra={"step": "retrieve_done"},
            )

    return results_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _apply_score_gap_cutoff(
    results: list[dict], gap_threshold: float, min_keep: int = 5
) -> list[dict]:
    if gap_threshold <= 0 or len(results) <= min_keep:
        return results

    for i in range(min_keep, len(results)):
        gap = results[i - 1]["score"] - results[i]["score"]
        if gap >= gap_threshold:
            log.info("score gap=%.4f at rank=%d -- cutting off",
                     gap, i, extra={"step": "retrieve_done"})
            return results[:i]
    return results


def _log_top_hits(label: str, hits: list[dict], n: int = 5) -> None:
    if not hits:
        return
    for i, h in enumerate(hits[:n]):
        log.debug(
            "%s top%d doc=%s cos=%.4f section=%s title=%r",
            label, i + 1, h["doc_id"], h["cosine_score"],
            h.get("section_id") or "-",
            (h.get("doc_title") or "")[:60],
            extra={"step": "retrieve_search"},
        )
