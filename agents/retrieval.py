"""
agents/retrieval.py -- Retrieval Agent
========================================
Three-level hybrid retrieval against pg_doc_index (doc-level embeddings only).
Uses case metadata (not alert metadata) for citation matching and embeddings.
Score-based aggregation with configurable thresholds for precision control.

Env vars for tuning:
  RETRIEVAL_SIM_THRESHOLD  -- min cosine similarity (default 0.35)
  RETRIEVAL_TOP_K          -- max results returned (default 25)
  RETRIEVAL_L0_LIMIT       -- Level 0 broad sweep limit (default 200)
  RETRIEVAL_L1_LIMIT       -- Level 1 page-doc limit (default 100)
  RETRIEVAL_L2_LIMIT       -- Level 2 paragraph limit (default 50)
  RETRIEVAL_SCORE_GAP      -- drop results after this gap (default 0.08)
"""

from __future__ import annotations

import os
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from tools.embeddings import get_qdrant
from tools.retrieval import three_level_retrieve
from tools.metadata_db import get_pg_source_file


class RetrievalAgent(BaseAgent):
    """
    Runs three-level hybrid retrieval to find candidate PG documents.
    Uses case metadata (not alert metadata) for citations and embeddings.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        case_summary_embedding = state.get("case_summary_embedding")
        case_full_doc_embedding = state.get("case_full_doc_embedding")
        doc_embeddings = state.get("case_doc_embeddings", {})
        chunk_embeddings = state.get("case_chunk_embeddings", [])
        keywords = state.get("case_keywords", [])
        chunk_metadata = state.get("case_chunk_metadata", [])

        case_cite_ref = state.get("case_cite_ref", "")
        case_cite_refs = [case_cite_ref] if case_cite_ref else []

        if not doc_embeddings and not chunk_embeddings and not case_summary_embedding:
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text="ERROR: No case embeddings in state. Cannot retrieve.")]
                ),
            )
            return

        qdrant_path = os.getenv("QDRANT_PATH", "data/qdrant")
        qdrant = get_qdrant(qdrant_path)

        print("  [RetrievalAgent] Starting three-level grounded retrieval...")
        results = three_level_retrieve(
            qdrant=qdrant,
            case_summary_embedding=case_summary_embedding,
            case_full_doc_embedding=case_full_doc_embedding,
            doc_embeddings=doc_embeddings,
            chunk_embeddings=chunk_embeddings,
            chunk_metadata=chunk_metadata,
            keywords=keywords,
            case_cite_refs=case_cite_refs,
            case_cite_defs=[],
        )

        for doc in results:
            doc["source_file"] = get_pg_source_file(doc["doc_id"])

        state["candidate_pg_docs"] = results

        summary_lines = [
            f"Three-level retrieval: {len(results)} candidate PG documents",
            f"  case_cite_ref: {case_cite_ref or 'N/A'}",
        ]
        if results:
            summary_lines.append(
                f"  Score range: {results[-1]['score']:.4f} -- {results[0]['score']:.4f}"
            )
        summary_lines.append("")

        for i, doc in enumerate(results):
            summary_lines.append(
                f"  [{i + 1}] {doc['doc_id']} | score={doc['score']:.4f} "
                f"| hits={doc.get('query_hits', '?')} | {doc.get('doc_title', '')[:55]}"
            )

        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text="\n".join(summary_lines))]
            ),
            actions=EventActions(state_delta={"candidate_pg_docs": results}),
        )


retrieval_agent = RetrievalAgent(
    name="RetrievalAgent",
    description="Three-level hybrid retrieval with score-based aggregation and precision controls.",
)
