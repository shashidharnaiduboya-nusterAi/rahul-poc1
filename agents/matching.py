"""
agents/matching.py -- Matching Agent (Grounded)
=================================================
For each candidate PG document, parses it into sections/subsections and uses
an LLM to determine which sections are impacted.

Changes vs the prior implementation:
  * PG docs retrieved via citation/BM25 only (no L2 paragraph evidence) are
    no longer hard-skipped.  We fall back to the top-N case chunks (scored
    against the PG doc summary) so every retrieved doc gets a fair matching
    attempt.
  * Structured logging (tools/logging_setup) replaces scattered prints.  Every
    log line carries ``alert_id`` so operators can trace one alert end-to-end.
"""

from __future__ import annotations

import json
import time
from typing import AsyncGenerator

import numpy as np

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from tools.xml_parsers import parse_pgdoc_sections
from tools.llm_helper import call_llm_json
from tools.embeddings import encode_texts_as_query
from tools.logging_setup import get_logger, bind_alert
from prompts.matching import MATCHING_SYSTEM, MATCHING_USER_TEMPLATE

_log = get_logger("agents.matching")

_FALLBACK_TOP_CHUNKS = 3


def _format_matched_paragraphs(
    matched: list[dict], case_chunks: list[dict], label: str = "Case Para",
) -> str:
    """Render matched case paragraphs into the prompt block."""
    if not matched:
        return "(No matched paragraphs available)"

    lines: list[str] = []
    for mp in matched[:10]:
        chunk_idx = mp.get("chunk_index", -1)
        score = mp.get("cosine_score", 0)
        preview = mp.get("text_preview", "")

        full_text = preview
        if 0 <= chunk_idx < len(case_chunks):
            chunk = case_chunks[chunk_idx]
            full_text = chunk.get("text", preview) if isinstance(chunk, dict) else preview

        lines.append(f"[{label} {chunk_idx + 1}] (relevance: {score:.2f})\n{full_text}")

    return "\n\n".join(lines)


def _fallback_chunks_for_pg_doc(
    pg_doc: dict,
    case_chunks: list[dict],
    case_chunk_embeddings: list[list[float]],
) -> list[dict]:
    """
    When retrieval produced no paragraph-level evidence for a PG doc, pick the
    top-N case chunks by cosine similarity against the PG doc summary / title
    so MatchingAgent still has something concrete to reason over.

    Returns a list of dicts in the same shape as ``matched_paragraphs``.
    """
    if not case_chunks or not case_chunk_embeddings:
        return []

    summary = (pg_doc.get("pg_doc_summary") or "").strip()
    title = (pg_doc.get("doc_title") or "").strip()
    seed = " ".join(x for x in (title, summary) if x)[:4_000]
    if not seed:
        return []

    try:
        q_vec = encode_texts_as_query([seed])[0]
    except Exception:
        return []

    case_arr = np.asarray(case_chunk_embeddings, dtype=np.float32)
    if case_arr.size == 0:
        return []

    q = np.asarray(q_vec, dtype=np.float32)
    # Cosine: both sides are (near) unit-norm from Qwen3; still normalise to be safe
    q_n = np.linalg.norm(q) or 1.0
    case_norms = np.linalg.norm(case_arr, axis=1)
    case_norms[case_norms == 0] = 1.0
    sims = (case_arr @ q) / (case_norms * q_n)

    top_idx = np.argsort(-sims)[:_FALLBACK_TOP_CHUNKS]

    evidence: list[dict] = []
    for idx in top_idx:
        i = int(idx)
        if sims[i] <= 0:
            continue
        chunk = case_chunks[i] if i < len(case_chunks) else {}
        preview = chunk.get("text", "") if isinstance(chunk, dict) else ""
        evidence.append({
            "chunk_index": i,
            "cosine_score": float(sims[i]),
            "text_preview": preview[:400],
            "fallback": True,
        })
    return evidence


class MatchingAgent(BaseAgent):
    """Grounded section-level impact matching with fallback evidence."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        candidates = state.get("candidate_pg_docs", []) or []
        case_chunks = state.get("case_chunks", []) or []
        case_chunk_embeddings = state.get("case_chunk_embeddings", []) or []
        case_keywords = state.get("case_keywords", []) or []
        case_citation = (state.get("case_citation", "")
                         or state.get("case_cite_ref", ""))
        alert_meta = state.get("alert_metadata", {}) or {}
        alert_id = alert_meta.get("lni_id") or state.get("case_id") or "-"

        log = bind_alert(_log, alert_id, step="matching")

        if not candidates:
            log.warning("no candidate PG docs -- nothing to match")
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text="No candidate PG documents to match against.")]
                ),
            )
            return

        log.info("starting matching over %d candidates", len(candidates))

        match_reports: list[dict] = []
        total_matches = 0
        fallback_used = 0

        case_kw_str = ", ".join(case_keywords[:80]) if case_keywords else "N/A"

        for doc_idx, pg_doc in enumerate(candidates):
            pg_doc_id = pg_doc["doc_id"]
            pg_doc_title = pg_doc.get("doc_title", "")
            source_file = pg_doc.get("source_file", "")
            para_count = pg_doc.get("para_match_count", 0)

            log.info(
                "[%d/%d] matching doc=%s paras=%d score=%.4f title=%r",
                doc_idx + 1, len(candidates), pg_doc_id, para_count,
                pg_doc.get("score", 0.0), pg_doc_title[:80],
            )

            matched = pg_doc.get("matched_paragraphs") or []
            if not matched:
                fb = _fallback_chunks_for_pg_doc(
                    pg_doc, case_chunks, case_chunk_embeddings,
                )
                if fb:
                    fallback_used += 1
                    matched = fb
                    log.info("  fallback evidence built: %d case chunks (no L2 hits)",
                             len(fb))
                else:
                    log.warning("  no matched paragraphs AND fallback failed -- skipping")
                    continue

            matched_para_text = _format_matched_paragraphs(matched, case_chunks)

            sections: list[dict] = []
            if source_file:
                try:
                    sections = parse_pgdoc_sections(source_file)
                except Exception as exc:
                    log.error("  parse_pgdoc_sections error: %s", exc)

            if not sections:
                chunk_text_preview = pg_doc.get("pg_doc_summary", "")
                if chunk_text_preview:
                    sections = [{
                        "section_id": "full_doc",
                        "heading": pg_doc_title,
                        "text": chunk_text_preview,
                        "subsections": [],
                    }]
                else:
                    log.warning("  no sections parseable from PG doc -- skipping")
                    continue

            log.info("  parsed PG doc into %d sections", len(sections))

            matched_sections: list[dict] = []
            t_llm_total = 0.0
            llm_calls = 0
            for section in sections:
                section_text = section["text"][:5_000]
                if len(section_text) < 30:
                    continue

                user_msg = MATCHING_USER_TEMPLATE.format(
                    matched_case_paragraphs=matched_para_text,
                    case_keywords=case_kw_str,
                    case_citation=case_citation or "N/A",
                    pg_doc_title=pg_doc_title,
                    pg_doc_id=pg_doc_id,
                    section_id=section["section_id"],
                    section_heading=section.get("heading", "(no heading)"),
                    section_text=section_text,
                )

                try:
                    t0 = time.perf_counter()
                    raw = call_llm_json(
                        system=MATCHING_SYSTEM,
                        user=user_msg,
                        model_type="strong",
                    )
                    dt = (time.perf_counter() - t0) * 1000.0
                    t_llm_total += dt
                    llm_calls += 1
                    result = json.loads(raw)

                    strength = result.get("match_strength", "NONE")
                    if (result.get("is_impacted")
                            and strength in ("HIGH", "MEDIUM")):
                        matched_sections.append({
                            "section_id": section["section_id"],
                            "section_heading": section.get("heading", ""),
                            "section_text": section_text,
                            "match_strength": strength,
                            "match_reason": result.get("match_reason", ""),
                            "relevant_case_aspects": result.get("relevant_case_aspects", []),
                            "affected_concepts": result.get("affected_concepts", []),
                        })
                        log.debug("    section=%s strength=%s elapsed_ms=%.1f",
                                  section["section_id"], strength, dt)
                except (json.JSONDecodeError, Exception) as exc:
                    log.error("    LLM error on section=%s: %s",
                              section["section_id"], exc)

            if matched_sections:
                total_matches += len(matched_sections)
                match_reports.append({
                    "pg_doc_id": pg_doc_id,
                    "pg_doc_title": pg_doc_title,
                    "source_file": source_file,
                    "score": pg_doc.get("score", 0),
                    "matched_paragraphs": matched,
                    "matched_sections": matched_sections,
                    "fallback_evidence": bool(pg_doc.get("matched_paragraphs") is None
                                              or not pg_doc.get("matched_paragraphs")),
                })
                log.info(
                    "  matched %d/%d sections llm_calls=%d llm_total_ms=%.1f",
                    len(matched_sections), len(sections), llm_calls, t_llm_total,
                )
            else:
                log.info(
                    "  no sections matched (llm_calls=%d total_ms=%.1f)",
                    llm_calls, t_llm_total,
                )

        state["match_reports"] = match_reports

        log.info(
            "matching done docs_with_matches=%d total_sections=%d fallback_used=%d",
            len(match_reports), total_matches, fallback_used,
        )

        summary = (
            f"Matching complete: {len(match_reports)} PG docs with matches, "
            f"{total_matches} total matched sections "
            f"({fallback_used} used fallback evidence)"
        )

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=summary)]),
            actions=EventActions(state_delta={"match_reports": match_reports}),
        )


matching_agent = MatchingAgent(
    name="MatchingAgent",
    description=(
        "Grounded section-level impact matching. Uses matched case paragraphs "
        "from retrieval when available, and falls back to top-N case chunks "
        "scored against the PG doc summary when no paragraph evidence exists."
    ),
)
