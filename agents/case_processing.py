"""
agents/case_processing.py -- Case Processing Agent
====================================================
When an alert arrives and the case is identified:
  1. Find the case using cite_ref (normcite from cite_defs) -- primary lookup
     Fallback: try lni_id if cite_ref lookup fails
  2. Chunk the case text at paragraph level, generate per-chunk AI summaries
  3. Embed every chunk (case-side paragraph vectors)
  4. Generate doc-level detailed summary + retrieval profile
  5. Build a sliding-window pooled full-document embedding so long judgments
     are not silently truncated at 16 kB (the previous behaviour)
"""

from __future__ import annotations

import json
import time
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from tools.chunking import chunk_text, ChunkRecord
from tools.embeddings import (
    encode_texts_as_query,
    encode_single_as_query,
    encode_long_text_as_query,
)
from tools.retrieval import extract_keywords
from tools.metadata_db import (
    get_case_by_lni,
    get_case_text_by_lni,
    find_case_by_cite_refs,
    get_case_text,
)
from tools.llm_helper import call_llm, call_llm_json
from tools.logging_setup import get_logger, bind_alert, StepTimer
from prompts.chunk_summary import CHUNK_SUMMARY_SYSTEM, CHUNK_BATCH_SIZE
from prompts.case_summary import CASE_SUMMARY_SYSTEM, RETRIEVAL_PROFILE_SYSTEM

_log = get_logger("agents.case_processing")


def _generate_chunk_summaries(chunks: list[ChunkRecord], log) -> None:
    """Generate structured AI summaries per chunk in-place."""
    total = len(chunks)
    log.info("generating chunk summaries total=%d batch=%d",
             total, CHUNK_BATCH_SIZE, extra={"step": "chunk_summaries"})

    t0 = time.perf_counter()
    for start in range(0, total, CHUNK_BATCH_SIZE):
        batch = chunks[start: start + CHUNK_BATCH_SIZE]
        texts = [c.text for c in batch]
        numbered = "\n\n".join(f"[{i + 1}] {t[:1_200]}" for i, t in enumerate(texts))

        try:
            raw = call_llm_json(
                system=CHUNK_SUMMARY_SYSTEM,
                user=numbered,
                model_type="fast",
            )
            parsed = json.loads(raw)
            results = parsed if isinstance(parsed, list) else []
            if not results and isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        results = v
                        break
        except (json.JSONDecodeError, Exception) as exc:
            log.warning("chunk summary LLM error: %s", exc,
                        extra={"step": "chunk_summaries"})
            results = []

        aligned: list[dict] = [{} for _ in batch]
        for item in results:
            if isinstance(item, dict):
                idx = item.get("chunk_index")
                if isinstance(idx, int) and 1 <= idx <= len(batch):
                    aligned[idx - 1] = item

        for chunk, summary in zip(batch, aligned):
            chunk.ai_summary = summary

        done = min(start + CHUNK_BATCH_SIZE, total)
        log.info("chunk summaries progress=%d/%d", done, total,
                 extra={"step": "chunk_summaries"})
    log.info("chunk summaries done elapsed_ms=%.1f",
             (time.perf_counter() - t0) * 1000.0,
             extra={"step": "chunk_summaries"})


class CaseProcessingAgent(BaseAgent):
    """
    Retrieves the case from the stored DB, then at runtime:
      1. Finds case via cite_ref (primary) or LNI (fallback).
      2. Paragraph-level chunking + per-chunk AI summaries.
      3. Per-chunk embeddings (query mode).
      4. Doc-level summary + retrieval profile via LLM.
      5. Sliding-window pooled full-doc embedding (replaces 16k-char truncation).
      6. Keyword extraction (doc text + summary + AI keyword/entity/topic lists).
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        alert_meta = state.get("alert_metadata", {}) or {}

        lni_id = alert_meta.get("lni_id", "") or ""
        cite_defs = alert_meta.get("cite_defs", []) or []
        cite_refs = alert_meta.get("cite_refs", []) or []

        log = bind_alert(_log, lni_id or "-", step="case_processing")

        case_meta: dict = {}
        case_text = ""
        lookup_method = ""

        with StepTimer(log, "case_lookup"):
            if cite_defs:
                log.info("lookup by cite_defs=%s", cite_defs)
                case_meta = find_case_by_cite_refs(cite_defs)
                if case_meta:
                    case_text = get_case_text(case_meta)
                    lookup_method = f"cite_ref match: {case_meta.get('cite_ref', '')}"
                    log.info("found via cite_defs lni=%s", case_meta.get("lni_id"))

            if not case_text and cite_refs:
                log.info("lookup by cite_refs=%s", cite_refs)
                case_meta = find_case_by_cite_refs(cite_refs)
                if case_meta:
                    case_text = get_case_text(case_meta)
                    lookup_method = f"cite_ref match (from cite_refs): {case_meta.get('cite_ref', '')}"
                    log.info("found via cite_refs lni=%s", case_meta.get("lni_id"))

            if not case_text and lni_id:
                log.info("lookup by LNI id=%s", lni_id)
                case_meta = get_case_by_lni(lni_id) or {}
                case_text = get_case_text_by_lni(lni_id)
                if case_text:
                    lookup_method = f"LNI ID: {lni_id}"
                    log.info("found via LNI id")

        if not case_text:
            log.error(
                "no case found; tried cite_defs=%s cite_refs=%s lni=%s",
                cite_defs, cite_refs, lni_id,
            )
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(
                        text=f"ERROR: No case found.\n"
                             f"  Tried cite_defs: {cite_defs}\n"
                             f"  Tried cite_refs: {cite_refs}\n"
                             f"  Tried LNI: {lni_id}\n"
                             f"  Make sure the case was pre-indexed with: "
                             f"python3 ingest.py <case.xml>"
                    )]
                ),
            )
            return

        case_id = case_meta.get("lni_id", lni_id or "unknown")
        case_title = case_meta.get("case_title", "")
        case_cite = case_meta.get("cite_ref", "")
        log.info(
            "case retrieved lookup=%s lni=%s cite=%s chars=%d title=%r",
            lookup_method, case_id, case_cite, len(case_text), case_title[:80],
        )

        # ---------------------- chunking ------------------------------
        with StepTimer(log, "chunking"):
            chunks = chunk_text(case_text, case_id)
            log.info("chunked into %d chunks", len(chunks))

        if not chunks:
            log.warning("no chunks produced -- aborting")
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text="WARN: No chunks produced from case text.")]
                ),
            )
            return

        # ---------------------- chunk summaries -----------------------
        try:
            with StepTimer(log, "chunk_summaries"):
                _generate_chunk_summaries(chunks, log)
        except Exception as exc:
            log.warning("chunk summary error (continuing): %s", exc)

        # ---------------------- chunk embeddings ----------------------
        chunk_texts = [c.text for c in chunks]
        with StepTimer(log, "chunk_embeddings"):
            chunk_embeddings_np = encode_texts_as_query(chunk_texts)
            chunk_embeddings = [emb.tolist() for emb in chunk_embeddings_np]
            log.info("chunk embeddings dim=%d count=%d",
                     len(chunk_embeddings[0]) if chunk_embeddings else 0,
                     len(chunk_embeddings))

        # ---------------------- doc summary + profile -----------------
        case_doc_summary = ""
        case_retrieval_profile = ""
        try:
            truncated = case_text[:180_000]
            if len(case_text) > 180_000:
                truncated += "\n\n[Document truncated at 180,000 chars]"

            with StepTimer(log, "doc_summary_llm"):
                case_doc_summary = call_llm(
                    system=CASE_SUMMARY_SYSTEM,
                    user=(f"Generate a comprehensive human-readable summary of the "
                          f"following case:\n\n{truncated}"),
                    model_type="strong",
                )
                log.info("doc_summary words=%d",
                         len(case_doc_summary.split()))

            with StepTimer(log, "retrieval_profile_llm"):
                case_retrieval_profile = call_llm(
                    system=RETRIEVAL_PROFILE_SYSTEM,
                    user=(f"Generate a retrieval profile for the following case:"
                          f"\n\n{truncated}"),
                    model_type="strong",
                )
                log.info("retrieval_profile words=%d",
                         len(case_retrieval_profile.split()))
        except Exception as exc:
            log.warning("doc summary/profile error: %s", exc)

        # ---------------------- doc-level embeddings ------------------
        case_summary_embedding = None
        if case_doc_summary:
            with StepTimer(log, "summary_embedding"):
                case_summary_embedding = encode_single_as_query(case_doc_summary)

        with StepTimer(log, "full_doc_embedding_pooled"):
            case_full_doc_embedding = encode_long_text_as_query(case_text)
            log.info("full-doc pooled embedding built over ~%d chars",
                     len(case_text))

        doc_embeddings: dict[str, list[float]] = {}
        if case_retrieval_profile:
            with StepTimer(log, "profile_embedding"):
                doc_embeddings["retrieval_profile"] = encode_single_as_query(
                    case_retrieval_profile
                )

        # ---------------------- keywords ------------------------------
        all_keyword_texts = list(chunk_texts)
        if case_doc_summary:
            all_keyword_texts.append(case_doc_summary)
        if case_retrieval_profile:
            all_keyword_texts.append(case_retrieval_profile)
        all_keyword_texts.append(case_text[:80_000])

        ai_kw_lists = [c.ai_summary.get("keywords", []) for c in chunks]
        ai_entity_lists = [c.ai_summary.get("entities", []) for c in chunks]
        ai_topic_lists = [c.ai_summary.get("key_topics", []) for c in chunks]
        keywords = extract_keywords(
            all_keyword_texts,
            ai_keywords=ai_kw_lists + ai_entity_lists + ai_topic_lists,
        )
        log.info("keywords extracted count=%d sample=%s",
                 len(keywords), keywords[:15])

        # ---------------------- chunk metadata ------------------------
        chunk_metadata: list[dict] = []
        for c in chunks:
            chunk_metadata.append({
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "text_preview": c.text[:300],
                "keywords": c.ai_summary.get("keywords", []),
                "entities": c.ai_summary.get("entities", []),
                "key_topics": c.ai_summary.get("key_topics", []),
                "citations": c.ai_summary.get("citations", []),
            })

        chunk_dicts = [c.to_dict() for c in chunks]
        state_delta = {
            "case_id": case_id,
            "case_title": case_title,
            "case_cite_ref": case_cite,
            "case_citation": case_cite,
            "case_text": case_text[:80_000],
            "case_chunks": chunk_dicts,
            "case_chunk_metadata": chunk_metadata,
            "case_doc_summary": case_doc_summary,
            "case_retrieval_profile": case_retrieval_profile,
            "case_summary_embedding": case_summary_embedding,
            "case_full_doc_embedding": case_full_doc_embedding,
            "case_chunk_embeddings": chunk_embeddings,
            "case_doc_embeddings": doc_embeddings,
            "case_keywords": keywords,
        }
        state.update(state_delta)

        summary = (
            f"Case '{case_id}' processed:\n"
            f"  Lookup: {lookup_method}\n"
            f"  cite_ref: {case_cite}\n"
            f"  Chunks: {len(chunks)} (paragraph-level)\n"
            f"  Chunk embeddings: {len(chunk_embeddings)}\n"
            f"  Doc summary: {len(case_doc_summary.split())} words\n"
            f"  Retrieval profile: {len(case_retrieval_profile.split())} words\n"
            f"  L0 summary embedding: {'ready' if case_summary_embedding else 'none'}\n"
            f"  L1 pooled full-doc embedding: ready (sliding-window)\n"
            f"  L1 profile embeddings: {len(doc_embeddings)}\n"
            f"  Keywords: {len(keywords)}"
        )

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=summary)]),
            actions=EventActions(state_delta=state_delta),
        )


case_processing_agent = CaseProcessingAgent(
    name="CaseProcessingAgent",
    description=(
        "Finds case by cite_ref (normcite), chunks it, generates summaries "
        "and embeddings (including a sliding-window pooled full-doc vector). "
        "All runtime data is temporary / in-memory."
    ),
)
