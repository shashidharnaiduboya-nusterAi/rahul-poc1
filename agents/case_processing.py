"""
agents/case_processing.py -- Case Processing Agent
====================================================
When an alert arrives and the case is identified:
  1. Find the case using cite_ref (normcite from cite_defs) -- primary lookup
     Fallback: try lni_id if cite_ref lookup fails
  2. Chunk it at paragraph level, get metadata per paragraph, summarize each chunk
  3. Store embedding of each chunked paragraph (in session state -- temporary)
  4. Get document-level detailed summary (15-20 pages), metadata, and embeddings
"""

from __future__ import annotations

import json
import os
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from tools.chunking import chunk_text, ChunkRecord
from tools.embeddings import encode_texts_as_query, encode_single_as_query
from tools.retrieval import extract_keywords
from tools.metadata_db import (
    get_case_by_lni,
    get_case_text_by_lni,
    find_case_by_cite_refs,
    get_case_text,
)
from tools.llm_helper import call_llm, call_llm_json
from prompts.chunk_summary import CHUNK_SUMMARY_SYSTEM, CHUNK_BATCH_SIZE
from prompts.case_summary import CASE_SUMMARY_SYSTEM, RETRIEVAL_PROFILE_SYSTEM


def _generate_chunk_summaries(chunks: list[ChunkRecord]) -> None:
    """Generate structured AI summaries per chunk in-place."""
    total = len(chunks)
    print(f"  [CaseAgent] Generating chunk summaries ({total} chunks, batch={CHUNK_BATCH_SIZE})...")

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
        except (json.JSONDecodeError, Exception):
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
        print(f"  [CaseAgent]   Chunk summaries: {done}/{total}")


class CaseProcessingAgent(BaseAgent):
    """
    Retrieves the case from the stored database, then at runtime:
      1. Finds case using cite_ref (from alert cite_defs/normcite) -- primary
         Fallback: LNI ID lookup
      2. Chunks at paragraph level
      3. Generates per-chunk metadata + summary (LLM fast model)
      4. Stores chunk embeddings in session state (temporary)
      5. Generates document-level detailed summary (15-20 pages)
      6. Generates retrieval profile
      7. Embeds doc-level summary + profile
      8. Extracts keywords for BM25

    All results stored in session state (temporary, in-memory).
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        alert_meta = state.get("alert_metadata", {})

        lni_id = alert_meta.get("lni_id", "")
        cite_defs = alert_meta.get("cite_defs", [])
        cite_refs = alert_meta.get("cite_refs", [])

        # PRIMARY: Find case using cite_ref (normcite from cite_defs)
        case_meta = {}
        case_text = ""
        lookup_method = ""

        if cite_defs:
            print(f"  [CaseAgent] Looking up case by cite_ref: {cite_defs}")
            case_meta = find_case_by_cite_refs(cite_defs)
            if case_meta:
                case_text = get_case_text(case_meta)
                lookup_method = f"cite_ref match: {case_meta.get('cite_ref', '')}"
                print(f"  [CaseAgent] Found via cite_ref -> LNI={case_meta.get('lni_id', '')}")

        # FALLBACK 1: Try cite_refs too
        if not case_text and cite_refs:
            print(f"  [CaseAgent] cite_defs didn't match, trying cite_refs: {cite_refs}")
            case_meta = find_case_by_cite_refs(cite_refs)
            if case_meta:
                case_text = get_case_text(case_meta)
                lookup_method = f"cite_ref match (from cite_refs): {case_meta.get('cite_ref', '')}"
                print(f"  [CaseAgent] Found via cite_refs -> LNI={case_meta.get('lni_id', '')}")

        # FALLBACK 2: Try LNI ID
        if not case_text and lni_id:
            print(f"  [CaseAgent] cite_ref lookup failed, falling back to LNI: {lni_id}")
            case_meta = get_case_by_lni(lni_id)
            case_text = get_case_text_by_lni(lni_id)
            if case_text:
                lookup_method = f"LNI ID: {lni_id}"
                print(f"  [CaseAgent] Found via LNI ID")

        if not case_text:
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(
                        text=f"ERROR: No case found.\n"
                             f"  Tried cite_defs: {cite_defs}\n"
                             f"  Tried cite_refs: {cite_refs}\n"
                             f"  Tried LNI: {lni_id}\n"
                             f"  Make sure the case was pre-indexed with: python3 ingest.py <case.xml>"
                    )]
                ),
            )
            return

        case_id = case_meta.get("lni_id", lni_id or "unknown")
        case_title = case_meta.get("case_title", "")
        case_cite = case_meta.get("cite_ref", "")
        print(f"  [CaseAgent] Case retrieved: {lookup_method}")
        print(f"  [CaseAgent]   LNI: {case_id} | cite_ref: {case_cite}")
        print(f"  [CaseAgent]   Title: {case_title}")
        print(f"  [CaseAgent]   Text: {len(case_text)} chars")

        # Step 2: Paragraph-level chunking
        chunks = chunk_text(case_text, case_id)
        print(f"  [CaseAgent] Chunked into {len(chunks)} chunks")

        if not chunks:
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text="WARN: No chunks produced from case text.")]
                ),
            )
            return

        # Step 3: Per-chunk structured summaries + metadata (LLM fast model)
        try:
            _generate_chunk_summaries(chunks)
        except Exception as exc:
            print(f"  [CaseAgent] Chunk summary error (continuing): {exc}")

        # Step 4: Embed each chunked paragraph (stored in session state = temporary)
        chunk_texts = [c.text for c in chunks]
        chunk_embeddings_np = encode_texts_as_query(chunk_texts)
        chunk_embeddings = [emb.tolist() for emb in chunk_embeddings_np]
        print(f"  [CaseAgent] Embedded {len(chunk_embeddings)} paragraph chunks")

        # Step 5: Document-level detailed summary (15-20 pages) + metadata
        case_doc_summary = ""
        case_retrieval_profile = ""
        try:
            truncated = case_text[:180_000]
            if len(case_text) > 180_000:
                truncated += "\n\n[Document truncated at 180,000 chars]"

            print("  [CaseAgent] Generating comprehensive case summary (15-20 pages)...")
            case_doc_summary = call_llm(
                system=CASE_SUMMARY_SYSTEM,
                user=f"Generate a comprehensive human-readable summary of the following case:\n\n{truncated}",
                model_type="strong",
            )
            print(f"  [CaseAgent]   Summary: {len(case_doc_summary.split())} words")

            print("  [CaseAgent] Generating retrieval profile...")
            case_retrieval_profile = call_llm(
                system=RETRIEVAL_PROFILE_SYSTEM,
                user=f"Generate a retrieval profile for the following case:\n\n{truncated}",
                model_type="strong",
            )
            print(f"  [CaseAgent]   Profile: {len(case_retrieval_profile.split())} words")
        except Exception as exc:
            print(f"  [CaseAgent] Doc summary error: {exc}")

        # Step 6a: Embed case summary for Level 0 broad retrieval
        case_summary_embedding = None
        if case_doc_summary:
            case_summary_embedding = encode_single_as_query(case_doc_summary)

        # Step 6b: Embed raw case text for Level 1 full-document comparison
        case_full_doc_embedding = encode_single_as_query(case_text[:16_000])
        print(f"  [CaseAgent] Embedded full case document ({min(len(case_text), 16_000)} chars)")

        # Step 6c: Embed retrieval profile for Level 1 targeted refinement
        doc_embeddings: dict[str, list[float]] = {}
        if case_retrieval_profile:
            doc_embeddings["retrieval_profile"] = encode_single_as_query(case_retrieval_profile)

        # Step 7: Extract keywords from WHOLE document text + summaries + AI metadata
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

        # Step 8: Collect per-chunk metadata (for paragraph-level retrieval)
        chunk_metadata: list[dict] = []
        for c in chunks:
            meta = {
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "text_preview": c.text[:300],
                "keywords": c.ai_summary.get("keywords", []),
                "entities": c.ai_summary.get("entities", []),
                "key_topics": c.ai_summary.get("key_topics", []),
                "citations": c.ai_summary.get("citations", []),
            }
            chunk_metadata.append(meta)

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
            f"  Level 0 embedding: {'ready' if case_summary_embedding else 'none'}\n"
            f"  Level 1 full-doc embedding: ready\n"
            f"  Level 1 profile embeddings: {len(doc_embeddings)} representations\n"
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
        "and embeddings. All runtime data is temporary / in-memory."
    ),
)
