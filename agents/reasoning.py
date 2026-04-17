"""
agents/reasoning.py -- Reasoning Agent (Grounded)
===================================================
For each matched PG document section, generates specific change suggestions
grounded in the actual case paragraphs that matched, not a broad summary.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from tools.llm_helper import call_llm_json
from prompts.reasoning import REASONING_SYSTEM, REASONING_USER_TEMPLATE


def _format_evidence_for_reasoning(
    report: dict, case_chunks: list[dict]
) -> str:
    """Build case excerpt text from the matched paragraphs for this PG doc."""
    matched = report.get("matched_paragraphs", [])
    if not matched:
        return "(No specific paragraph evidence available)"

    lines: list[str] = []
    for mp in matched[:10]:
        chunk_idx = mp.get("chunk_index", -1)
        score = mp.get("cosine_score", 0)
        preview = mp.get("text_preview", "")

        full_text = preview
        if 0 <= chunk_idx < len(case_chunks):
            chunk = case_chunks[chunk_idx]
            full_text = chunk.get("text", preview) if isinstance(chunk, dict) else preview

        lines.append(
            f"[Case Para {chunk_idx + 1}] (relevance: {score:.2f})\n{full_text}"
        )

    return "\n\n".join(lines) if lines else "(No matched paragraphs available)"


class ReasoningAgent(BaseAgent):
    """
    Generates grounded WHERE/WHAT/WHY change suggestions using matched case
    paragraphs as evidence instead of the broad case summary.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        match_reports = state.get("match_reports", [])
        case_chunks = state.get("case_chunks", [])
        case_summary = state.get("case_doc_summary", "")
        case_citation = (state.get("case_citation", "")
                         or state.get("case_cite_ref", ""))

        if not match_reports:
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text="No matched sections to generate suggestions for.")]
                ),
            )
            return

        all_suggestions: list[dict] = []
        total_suggestions = 0

        for report_idx, report in enumerate(match_reports):
            pg_doc_id = report["pg_doc_id"]
            pg_doc_title = report.get("pg_doc_title", "")
            matched_sections = report.get("matched_sections", [])

            print(f"  [ReasoningAgent] [{report_idx + 1}/{len(match_reports)}] "
                  f"{pg_doc_id}: {len(matched_sections)} sections")

            evidence_text = _format_evidence_for_reasoning(report, case_chunks)

            if evidence_text.startswith("(No specific") or evidence_text.startswith("(No matched"):
                print(f"  [ReasoningAgent]   No paragraph evidence -- skipping")
                continue

            doc_suggestions: list[dict] = []

            for section in matched_sections:
                sec_heading = section.get("section_heading", "(no heading)")
                user_msg = REASONING_USER_TEMPLATE.format(
                    matched_case_paragraphs=evidence_text,
                    case_citation=case_citation or "N/A",
                    match_reason=section.get("match_reason", ""),
                    pg_doc_title=pg_doc_title,
                    pg_doc_id=pg_doc_id,
                    section_id=section["section_id"],
                    section_heading=sec_heading,
                    section_text=section.get("section_text", "")[:5_000],
                )

                try:
                    raw = call_llm_json(
                        system=REASONING_SYSTEM,
                        user=user_msg,
                        model_type="strong",
                    )
                    result = json.loads(raw)

                    suggestion = result.get("suggestion")
                    if isinstance(suggestion, list):
                        suggestion = suggestion[0] if suggestion else None

                    if (suggestion
                            and isinstance(suggestion, dict)
                            and suggestion.get("what_to_change", "").strip()
                            and suggestion.get("change_type", "").upper() != "NOTE"):
                        doc_suggestions.append({
                            "section_id": section["section_id"],
                            "section_heading": sec_heading,
                            "match_strength": section.get("match_strength", ""),
                            "suggestion": suggestion,
                            "priority": result.get("priority", "MEDIUM"),
                            "summary": result.get("summary", ""),
                        })
                        total_suggestions += 1
                    else:
                        print(f"  [ReasoningAgent]   {section['section_id']}: "
                              f"no substantive suggestion")

                except (json.JSONDecodeError, Exception) as exc:
                    print(f"  [ReasoningAgent]   LLM error on "
                          f"{section['section_id']}: {exc}")

            if doc_suggestions:
                all_suggestions.append({
                    "pg_doc_id": pg_doc_id,
                    "pg_doc_title": pg_doc_title,
                    "source_file": report.get("source_file", ""),
                    "section_suggestions": doc_suggestions,
                })
                print(f"  [ReasoningAgent]   {len(doc_suggestions)} sections "
                      f"with substantive suggestions kept")
            else:
                print(f"  [ReasoningAgent]   No substantive suggestions -- "
                      f"PG doc filtered out")

        state["suggestions"] = all_suggestions

        summary = (
            f"Reasoning complete: {total_suggestions} raw suggestions, "
            f"{sum(len(s['section_suggestions']) for s in all_suggestions)} sections "
            f"with substantive changes across {len(all_suggestions)} PG documents"
        )

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=summary)]),
            actions=EventActions(state_delta={"suggestions": all_suggestions}),
        )


reasoning_agent = ReasoningAgent(
    name="ReasoningAgent",
    description="Generates grounded WHERE/WHAT/WHY suggestions using matched case paragraphs.",
)
