"""
tools/retrieval.py -- Three-Level Hybrid Retrieval (Grounded)
==============================================================
Level 0: Case document summary embedding (broad sweep)
Level 1: Full case doc embedding (comprehensive refinement against L0 candidates)
Level 2: Individual paragraph embeddings (each paragraph searches independently,
         preserving ground truth -- NO pooling)

Each PG doc result includes `matched_paragraphs` -- the specific case paragraphs
that matched it, providing an evidence chain for downstream matching/reasoning.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

PG_DOC_COLL = "pg_doc_index"

_DEFAULT_SIM_THRESHOLD = 0.30
_DEFAULT_TOP_K = 25
_DEFAULT_SCORE_GAP = 0.08

_DEFAULT_L0_LIMIT = 100
_DEFAULT_L1_LIMIT = 50
_DEFAULT_L2_PER_PARA_LIMIT = 10
_MAX_PARA_QUERIES = 20

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
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def extract_keywords(
    texts: list[str],
    ai_keywords: Optional[list[list[str]]] = None,
    top_n: int = 150,
) -> list[str]:
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
# Qdrant search
# ---------------------------------------------------------------------------
def _collection_exists(qdrant: QdrantClient, name: str) -> bool:
    return name in [c.name for c in qdrant.get_collections().collections]


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
        )
    except Exception as exc:
        print(f"  [Retriever] Qdrant error: {exc}")
        return []

    results: list[dict] = []
    for hit in results_obj.points:
        p = hit.payload or {}
        results.append({
            "doc_id": p.get("doc_id", str(hit.id)),
            "doc_title": p.get("doc_title", ""),
            "cite_ids": p.get("cite_ids", []),
            "practice_area": p.get("practice_area", ""),
            "doc_summary": p.get("doc_summary", ""),
            "cosine_score": round(float(hit.score), 4),
        })
    return results


# ---------------------------------------------------------------------------
# BM25 scoring
# ---------------------------------------------------------------------------
def _bm25_score(keywords: list[str], docs: list[dict]) -> dict[str, float]:
    if not docs or not keywords:
        return {}

    corpus = [
        kw_tokenize(
            (d.get("doc_summary") or "")
            + " " + (d.get("doc_title") or "")
            + " " + (d.get("practice_area") or "")
        )
        for d in docs
    ]

    valid_indices = [i for i, c in enumerate(corpus) if c]
    if not valid_indices:
        return {}

    valid_corpus = [corpus[i] for i in valid_indices]
    valid_docs = [docs[i] for i in valid_indices]

    bm25 = BM25Okapi(valid_corpus)
    raw_scores = bm25.get_scores(keywords)

    max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
    return {
        d["doc_id"]: round(float(s / max_score), 4)
        for d, s in zip(valid_docs, raw_scores)
        if s > 0
    }


# ---------------------------------------------------------------------------
# Citation matching
# ---------------------------------------------------------------------------
def _citation_boost(
    candidates: dict[str, dict],
    cite_refs: list[str],
    cite_defs: list[str],
) -> dict[str, float]:
    if not cite_refs and not cite_defs:
        return {}

    case_cites = set()
    for c in cite_refs + cite_defs:
        normalized = c.strip().lower().replace("#", "")
        if normalized:
            case_cites.add(normalized)

    if not case_cites:
        return {}

    boosts: dict[str, float] = {}
    for doc_id, doc in candidates.items():
        pg_cites = doc.get("cite_ids", [])
        if isinstance(pg_cites, str):
            try:
                pg_cites = json.loads(pg_cites)
            except (json.JSONDecodeError, TypeError):
                pg_cites = []

        overlap = sum(1 for pc in pg_cites if pc.strip().lower() in case_cites)
        if overlap > 0:
            boosts[doc_id] = min(overlap * 0.05, 0.15)

    if boosts:
        print(f"  [Retriever]   Citation overlap: {len(boosts)} PG docs boosted")
    return boosts


# ---------------------------------------------------------------------------
# Score-gap cutoff
# ---------------------------------------------------------------------------
def _apply_score_gap_cutoff(
    results: list[dict], gap_threshold: float, min_keep: int = 3
) -> list[dict]:
    if len(results) <= min_keep:
        return results

    for i in range(min_keep, len(results)):
        gap = results[i - 1]["score"] - results[i]["score"]
        if gap >= gap_threshold:
            print(f"  [Retriever]   Score gap {gap:.3f} at position {i} -- cutting off")
            return results[:i]

    return results


# ===========================================================================
# Three-Level Grounded Retrieval
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
    sim_threshold: float = None,
) -> list[dict]:
    """
    Three-level grounded retrieval with paragraph-level evidence tracking.

    Level 0 -- Broad sweep using case document summary
    Level 1 -- Full case doc embedding + retrieval profile against L0 candidates
    Level 2 -- Individual paragraph embeddings (NO pooling -- each paragraph
               searches independently, preserving ground truth)

    Each result includes `matched_paragraphs` listing the specific case
    paragraphs that matched, providing grounding for downstream agents.
    """
    case_cite_refs = case_cite_refs or []
    case_cite_defs = case_cite_defs or []

    if sim_threshold is None:
        sim_threshold = float(os.getenv(
            "RETRIEVAL_SIM_THRESHOLD", str(_DEFAULT_SIM_THRESHOLD)))
    top_k = int(os.getenv("RETRIEVAL_TOP_K", str(_DEFAULT_TOP_K)))
    l0_limit = int(os.getenv("RETRIEVAL_L0_LIMIT", str(_DEFAULT_L0_LIMIT)))
    l1_limit = int(os.getenv("RETRIEVAL_L1_LIMIT", str(_DEFAULT_L1_LIMIT)))
    l2_per_para = int(os.getenv("RETRIEVAL_L2_PER_PARA_LIMIT",
                                str(_DEFAULT_L2_PER_PARA_LIMIT)))
    score_gap = float(os.getenv("RETRIEVAL_SCORE_GAP", str(_DEFAULT_SCORE_GAP)))

    l0_threshold = sim_threshold * 0.90
    l2_threshold = sim_threshold * 1.15

    print(f"  [Retriever] Config: sim_threshold={sim_threshold}, top_k={top_k}, "
          f"l0_limit={l0_limit}, l1_limit={l1_limit}, "
          f"l2_per_para={l2_per_para}, score_gap={score_gap}")

    if not _collection_exists(qdrant, PG_DOC_COLL):
        print(f"  [Retriever] Collection '{PG_DOC_COLL}' not found. Run ingest.py first.")
        return []

    doc_scores: dict[str, dict] = defaultdict(lambda: {
        "weighted_sum": 0.0,
        "weight_total": 0.0,
        "max_cosine": 0.0,
        "query_hits": 0,
    })
    all_docs: dict[str, dict] = {}

    # Per-PG-doc paragraph evidence: doc_id -> list of {chunk_index, score, preview}
    para_evidence: dict[str, list[dict]] = defaultdict(list)

    def _accumulate(results: list[dict], weight: float, label: str):
        for r in results:
            did = r["doc_id"]
            cos = r["cosine_score"]
            all_docs.setdefault(did, r)
            entry = doc_scores[did]
            entry["weighted_sum"] += cos * weight
            entry["weight_total"] += weight
            entry["max_cosine"] = max(entry["max_cosine"], cos)
            entry["query_hits"] += 1

    # ======================================================================
    # LEVEL 0: Case document summary -- broad sweep (weight = 1.2)
    # ======================================================================
    print("  [Retriever] === LEVEL 0: Case summary broad sweep ===")

    if case_summary_embedding:
        results = _qdrant_search(
            qdrant, PG_DOC_COLL, case_summary_embedding,
            limit=l0_limit,
            score_threshold=l0_threshold,
        )
        _accumulate(results, weight=1.2, label="case_summary")
        print(f"  [Retriever]   Semantic (case_summary): "
              f"{len(results)} docs above {l0_threshold:.3f}")
    else:
        print("  [Retriever]   No case summary embedding -- skipping Level 0")

    level0_count = len(all_docs)
    print(f"  [Retriever]   Level 0 candidates: {level0_count}")

    # ======================================================================
    # LEVEL 1: Full case doc embedding + retrieval profile (weight = 1.0)
    # Checks the entire case content against the L0 candidate pool
    # ======================================================================
    print("  [Retriever] === LEVEL 1: Full case doc refinement ===")

    if case_full_doc_embedding:
        results = _qdrant_search(
            qdrant, PG_DOC_COLL, case_full_doc_embedding,
            limit=l1_limit,
            score_threshold=sim_threshold,
        )
        _accumulate(results, weight=1.0, label="case_full_doc")
        print(f"  [Retriever]   Semantic (case_full_doc): "
              f"{len(results)} docs above {sim_threshold}")

    for rep_name, embedding in doc_embeddings.items():
        if not embedding:
            continue
        results = _qdrant_search(
            qdrant, PG_DOC_COLL, embedding,
            limit=l1_limit,
            score_threshold=sim_threshold,
        )
        _accumulate(results, weight=1.0, label=rep_name)
        print(f"  [Retriever]   Semantic ({rep_name}): "
              f"{len(results)} docs above {sim_threshold}")

    level1_count = len(all_docs)
    print(f"  [Retriever]   Level 1 candidates: {level1_count} "
          f"(added {level1_count - level0_count})")

    # ======================================================================
    # LEVEL 2: Individual paragraph embeddings (NO pooling)
    # Each paragraph searches independently, preserving its actual content.
    # Tracks which paragraphs matched which PG docs for grounding.
    # ======================================================================
    print("  [Retriever] === LEVEL 2: Individual paragraph grounding ===")

    if chunk_embeddings:
        # Select paragraphs to query: skip very short ones, cap at MAX
        query_indices = []
        for i, cm in enumerate(chunk_metadata):
            preview = cm.get("text_preview", "")
            if len(preview) >= 150:
                query_indices.append(i)
        if len(query_indices) > _MAX_PARA_QUERIES:
            step = len(query_indices) / _MAX_PARA_QUERIES
            query_indices = [query_indices[int(i * step)]
                            for i in range(_MAX_PARA_QUERIES)]

        print(f"  [Retriever]   Searching with {len(query_indices)} "
              f"individual paragraphs (of {len(chunk_embeddings)} total)")

        para_hit_count = 0
        for idx in query_indices:
            if idx >= len(chunk_embeddings):
                continue
            emb = chunk_embeddings[idx]
            results = _qdrant_search(
                qdrant, PG_DOC_COLL, emb,
                limit=l2_per_para,
                score_threshold=l2_threshold,
            )

            for r in results:
                did = r["doc_id"]
                cos = r["cosine_score"]
                all_docs.setdefault(did, r)

                entry = doc_scores[did]
                entry["weighted_sum"] += cos * 0.6
                entry["weight_total"] += 0.6
                entry["max_cosine"] = max(entry["max_cosine"], cos)
                entry["query_hits"] += 1

                cm = chunk_metadata[idx] if idx < len(chunk_metadata) else {}
                para_evidence[did].append({
                    "chunk_index": idx,
                    "cosine_score": cos,
                    "text_preview": cm.get("text_preview", "")[:400],
                })

            para_hit_count += len(results)

        print(f"  [Retriever]   Paragraph queries produced {para_hit_count} "
              f"total hits across {len(para_evidence)} PG docs")
    else:
        print("  [Retriever]   No chunk embeddings -- skipping Level 2")

    total_after_semantic = len(all_docs)
    print(f"  [Retriever]   Total after all semantic levels: {total_after_semantic}")

    if not all_docs:
        print("  [Retriever]   No documents above threshold. "
              "Try lowering RETRIEVAL_SIM_THRESHOLD.")
        return []

    # ======================================================================
    # BM25 keyword boost
    # ======================================================================
    print("  [Retriever] === BM25 keyword boost ===")

    bm25_doc = _bm25_score(keywords[:80], list(all_docs.values()))
    for did, score in bm25_doc.items():
        doc_scores[did]["weighted_sum"] += score * 0.08
        doc_scores[did]["weight_total"] += 0.08

    para_kw = list(keywords)
    for cm in chunk_metadata:
        for field_name in ("keywords", "entities", "key_topics"):
            para_kw.extend(cm.get(field_name, []))
    para_kw_unique = list(set(kw_tokenize(" ".join(para_kw))))[:25]
    bm25_para = _bm25_score(para_kw_unique, list(all_docs.values()))
    for did, score in bm25_para.items():
        doc_scores[did]["weighted_sum"] += score * 0.03
        doc_scores[did]["weight_total"] += 0.03

    bm25_hit = len([d for d in bm25_doc if bm25_doc[d] > 0])
    print(f"  [Retriever]   BM25 doc-keywords: {bm25_hit} docs scored")

    # ======================================================================
    # Citation boost
    # ======================================================================
    cite_boosts = _citation_boost(all_docs, case_cite_refs, case_cite_defs)
    for did, boost in cite_boosts.items():
        doc_scores[did]["weighted_sum"] += boost
        doc_scores[did]["weight_total"] += 0.15

    # ======================================================================
    # Final scoring + attach paragraph evidence
    # ======================================================================
    print("  [Retriever] === Final scoring ===")

    min_query_hits = int(os.getenv("RETRIEVAL_MIN_HITS", "1"))

    results_list: list[dict] = []
    for doc_id, scores in doc_scores.items():
        doc = all_docs.get(doc_id)
        if not doc:
            continue

        wt = scores["weight_total"]
        if wt <= 0:
            continue

        if scores["query_hits"] < min_query_hits:
            continue

        avg_score = scores["weighted_sum"] / wt
        hit_bonus = min(scores["query_hits"] * 0.015, 0.08)
        final_score = avg_score + hit_bonus

        # Deduplicate paragraph evidence by chunk_index, keep highest score
        raw_evidence = para_evidence.get(doc_id, [])
        best_per_chunk: dict[int, dict] = {}
        for ev in raw_evidence:
            ci = ev["chunk_index"]
            if ci not in best_per_chunk or ev["cosine_score"] > best_per_chunk[ci]["cosine_score"]:
                best_per_chunk[ci] = ev

        # Overlap dedup: if consecutive chunks share char ranges,
        # keep only the higher-scoring one to avoid double-counting
        if len(best_per_chunk) > 1:
            sorted_indices = sorted(best_per_chunk.keys())
            to_remove: set[int] = set()
            for j in range(len(sorted_indices) - 1):
                ci_a = sorted_indices[j]
                ci_b = sorted_indices[j + 1]
                if ci_a in to_remove:
                    continue
                cm_a = chunk_metadata[ci_a] if ci_a < len(chunk_metadata) else {}
                cm_b = chunk_metadata[ci_b] if ci_b < len(chunk_metadata) else {}
                end_a = cm_a.get("char_end", 0)
                start_b = cm_b.get("char_start", 0)
                if end_a > start_b and end_a > 0 and start_b > 0:
                    score_a = best_per_chunk[ci_a]["cosine_score"]
                    score_b = best_per_chunk[ci_b]["cosine_score"]
                    to_remove.add(ci_a if score_b >= score_a else ci_b)
            for ci in to_remove:
                del best_per_chunk[ci]

        matched_paras = sorted(best_per_chunk.values(),
                               key=lambda x: -x["cosine_score"])

        results_list.append({
            "doc_id": doc_id,
            "doc_title": doc.get("doc_title", ""),
            "score": round(final_score, 4),
            "max_cosine": scores["max_cosine"],
            "query_hits": scores["query_hits"],
            "matched_paragraphs": matched_paras,
            "para_match_count": len(matched_paras),
        })

    results_list.sort(key=lambda d: -d["score"])

    results_list = _apply_score_gap_cutoff(results_list, score_gap)

    if len(results_list) > top_k:
        print(f"  [Retriever]   Trimming from {len(results_list)} to top-{top_k}")
        results_list = results_list[:top_k]

    print(f"  [Retriever]   Final: {len(results_list)} PG docs returned")
    if results_list:
        print(f"  [Retriever]   Score range: "
              f"{results_list[-1]['score']:.4f} -- {results_list[0]['score']:.4f}")
        with_paras = sum(1 for r in results_list if r["para_match_count"] > 0)
        print(f"  [Retriever]   Docs with paragraph evidence: {with_paras}")

    return results_list
