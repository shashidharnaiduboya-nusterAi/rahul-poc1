"""
agents/retrieval.py -- Retrieval Agent
========================================
Three-level hybrid retrieval with **reciprocal rank fusion** against the
``pg_doc_index`` and ``pg_chunks`` Qdrant collections.  Uses both case and
alert citations for the citation boost; see :mod:`tools.retrieval` for the
underlying algorithm.

Tunable env vars (defaults in parentheses):

    RETRIEVAL_SIM_THRESHOLD   min cosine similarity                (0.30)
    RETRIEVAL_TOP_K           max results returned                 (25)
    RETRIEVAL_L0_LIMIT        L0 broad sweep limit                 (100)
    RETRIEVAL_L1_LIMIT        L1 refinement limit                  (50)
    RETRIEVAL_L2_PER_PARA_LIMIT  L2 per-paragraph limit            (10)
    RETRIEVAL_MIN_HITS        min semantic hits per doc            (1)
    RETRIEVAL_SCORE_GAP       adaptive score-gap cutoff (0 = off)  (0.0)
    RRF_K                     RRF constant                         (60)
    RRF_W_L0 / RRF_W_L1 / RRF_W_L2 / RRF_W_BM25 / RRF_W_CITE  signal weights
"""

from __future__ import annotations

import os
import time
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from tools.embeddings import get_qdrant
from tools.retrieval import three_level_retrieve
from tools.metadata_db import get_pg_source_file
from tools.logging_setup import get_logger, bind_alert

_log = get_logger("agents.retrieval")


class RetrievalAgent(BaseAgent):
    """
    Three-level hybrid retrieval with RRF.  Uses both alert (cite_refs /
    cite_defs) and case (cite_ref) citations for the citation boost.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        alert_meta = state.get("alert_metadata", {}) or {}
        alert_id = alert_meta.get("lni_id") or state.get("case_id") or "-"
        log = bind_alert(_log, alert_id, step="retrieval")

        case_summary_embedding = state.get("case_summary_embedding")
        case_full_doc_embedding = state.get("case_full_doc_embedding")
        doc_embeddings = state.get("case_doc_embeddings", {}) or {}
        chunk_embeddings = state.get("case_chunk_embeddings", []) or []
        keywords = state.get("case_keywords", []) or []
        chunk_metadata = state.get("case_chunk_metadata", []) or []

        case_cite_ref = state.get("case_cite_ref", "") or ""
        case_cite_refs = [case_cite_ref] if case_cite_ref else []

        alert_cite_defs = alert_meta.get("cite_defs", []) or []
        alert_cite_refs = alert_meta.get("cite_refs", []) or []

        log.info(
            "inputs: chunks=%d keywords=%d case_cite=%r alert_defs=%d alert_refs=%d "
            "l0_emb=%s l1_emb=%s extra_doc_embs=%d",
            len(chunk_embeddings), len(keywords), case_cite_ref or "-",
            len(alert_cite_defs), len(alert_cite_refs),
            bool(case_summary_embedding), bool(case_full_doc_embedding),
            len(doc_embeddings),
        )

        if not doc_embeddings and not chunk_embeddings and not case_summary_embedding:
            log.error("no case embeddings in state -- cannot retrieve")
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text="ERROR: No case embeddings in state. Cannot retrieve.")]
                ),
            )
            return

        qdrant_path = os.getenv("QDRANT_PATH", "data/qdrant")
        qdrant = get_qdrant(qdrant_path)

        t0 = time.perf_counter()
        results = three_level_retrieve(
            qdrant=qdrant,
            case_summary_embedding=case_summary_embedding,
            case_full_doc_embedding=case_full_doc_embedding,
            doc_embeddings=doc_embeddings,
            chunk_embeddings=chunk_embeddings,
            chunk_metadata=chunk_metadata,
            keywords=keywords,
            case_cite_refs=case_cite_refs,
            case_cite_defs=[],            # case_cite_defs not currently tracked
            alert_cite_refs=alert_cite_refs,
            alert_cite_defs=alert_cite_defs,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        log.info("retrieval finished candidates=%d elapsed_ms=%.1f",
                 len(results), dt_ms)

        for doc in results:
            doc["source_file"] = get_pg_source_file(doc["doc_id"])

        state["candidate_pg_docs"] = results

        summary_lines = [
            f"Three-level retrieval (RRF): {len(results)} candidate PG documents",
            f"  case_cite_ref : {case_cite_ref or 'N/A'}",
            f"  alert_cite_refs: {len(alert_cite_refs)}  "
            f"alert_cite_defs: {len(alert_cite_defs)}",
        ]
        if results:
            summary_lines.append(
                f"  Score range    : {results[-1]['score']:.4f} -- {results[0]['score']:.4f}"
            )
        summary_lines.append("")

        for i, doc in enumerate(results):
            cs = doc.get("component_scores", {}) or {}
            summary_lines.append(
                f"  [{i + 1}] {doc['doc_id']} | "
                f"score={doc['score']:.4f} "
                f"hits={doc.get('query_hits', '?')} "
                f"paras={doc.get('para_match_count', 0)} "
                f"| L0={cs.get('l0', 0):.3f} L1={cs.get('l1', 0):.3f} "
                f"L2={cs.get('l2', 0):.3f} BM25={cs.get('bm25', 0):.3f} "
                f"CITE={cs.get('citation', 0):.3f} | "
                f"{(doc.get('doc_title', '') or '')[:55]}"
            )

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text="\n".join(summary_lines))]),
            actions=EventActions(state_delta={"candidate_pg_docs": results}),
        )


retrieval_agent = RetrievalAgent(
    name="RetrievalAgent",
    description=(
        "Hybrid retrieval with reciprocal rank fusion across L0/L1 doc-level "
        "vectors, L2 paragraph-level vectors, global BM25, and citation "
        "overlap."
    ),
)
