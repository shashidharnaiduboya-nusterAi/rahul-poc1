"""
agents/matching.py -- Matching Agent (Grounded)
=================================================
For each candidate PG document, parses it into sections/subsections and uses
LLM to determine which sections are impacted. Uses matched case paragraphs
(from retrieval) as evidence instead of the full case summary, keeping the
LLM grounded in actual case content.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from tools.xml_parsers import parse_pgdoc_sections
from tools.llm_helper import call_llm_json
from prompts.matching import MATCHING_SYSTEM, MATCHING_USER_TEMPLATE


def _format_matched_paragraphs(pg_doc: dict, case_chunks: list[dict]) -> str:
    """Build a text block of the actual case paragraphs that matched this PG doc."""
    matched = pg_doc.get("matched_paragraphs", [])
    if not matched and case_chunks:
        return "(No specific paragraph matches -- using case summary context)"

    lines: list[str] = []
    for i, mp in enumerate(matched[:10]):
        chunk_idx = mp.get("chunk_index", -1)
        score = mp.get("cosine_score", 0)
        preview = mp.get("text_preview", "")

        full_text = preview
        if chunk_idx >= 0 and chunk_idx < len(case_chunks):
            chunk = case_chunks[chunk_idx]
            full_text = chunk.get("text", preview) if isinstance(chunk, dict) else preview

        lines.append(f"[Case Para {chunk_idx + 1}] (relevance: {score:.2f})\n{full_text}")

    return "\n\n".join(lines) if lines else "(No matched paragraphs available)"


class MatchingAgent(BaseAgent):
    """
    Processes each candidate PG document:
      1. Parses PG XML into sections/subsections
      2. Uses matched case paragraphs (not full summary) as evidence
      3. LLM evaluates section impact grounded in actual case excerpts
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        candidates = state.get("candidate_pg_docs", [])
        case_chunks = state.get("case_chunks", [])
        case_summary = state.get("case_doc_summary", "")
        case_keywords = state.get("case_keywords", [])
        case_citation = (state.get("case_citation", "")
                         or state.get("case_cite_ref", ""))

        if not candidates:
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text="No candidate PG documents to match against.")]
                ),
            )
            return

        match_reports: list[dict] = []
        total_matches = 0

        case_kw_str = ", ".join(case_keywords[:80]) if case_keywords else "N/A"

        for doc_idx, pg_doc in enumerate(candidates):
            pg_doc_id = pg_doc["doc_id"]
            pg_doc_title = pg_doc.get("doc_title", "")
            source_file = pg_doc.get("source_file", "")
            para_count = pg_doc.get("para_match_count", 0)

            print(f"  [MatchingAgent] [{doc_idx + 1}/{len(candidates)}] "
                  f"Matching: {pg_doc_id} ({para_count} matched paras)")

            matched_para_text = _format_matched_paragraphs(pg_doc, case_chunks)

            if not pg_doc.get("matched_paragraphs"):
                print(f"  [MatchingAgent]   No paragraph evidence -- skipping "
                      f"(guardrail should have caught this)")
                continue

            sections: list[dict] = []
            if source_file:
                try:
                    sections = parse_pgdoc_sections(source_file)
                except Exception as exc:
                    print(f"  [MatchingAgent]   Parse error: {exc}")

            if not sections:
                chunk_text = pg_doc.get("chunk_text", "")
                if chunk_text:
                    sections = [{
                        "section_id": "full_doc",
                        "heading": pg_doc_title,
                        "text": chunk_text,
                        "subsections": [],
                    }]

            matched_sections: list[dict] = []

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
                    raw = call_llm_json(
                        system=MATCHING_SYSTEM,
                        user=user_msg,
                        model_type="strong",
                    )
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
                except (json.JSONDecodeError, Exception) as exc:
                    print(f"  [MatchingAgent]   LLM error on "
                          f"section {section['section_id']}: {exc}")

            if matched_sections:
                total_matches += len(matched_sections)
                match_reports.append({
                    "pg_doc_id": pg_doc_id,
                    "pg_doc_title": pg_doc_title,
                    "source_file": source_file,
                    "score": pg_doc.get("score", 0),
                    "matched_paragraphs": pg_doc.get("matched_paragraphs", []),
                    "matched_sections": matched_sections,
                })
                print(f"  [MatchingAgent]   {len(matched_sections)} sections matched")
            else:
                print(f"  [MatchingAgent]   No sections matched")

        state["match_reports"] = match_reports

        summary = (
            f"Matching complete: {len(match_reports)} PG docs with matches, "
            f"{total_matches} total matched sections"
        )

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=summary)]),
            actions=EventActions(state_delta={"match_reports": match_reports}),
        )


matching_agent = MatchingAgent(
    name="MatchingAgent",
    description="Grounded section-level impact matching using matched case paragraphs.",
)
