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
from tools.logging_setup import get_logger, bind_alert
from prompts.reasoning import REASONING_SYSTEM, REASONING_USER_TEMPLATE

_log = get_logger("agents.reasoning")


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

        match_reports = state.get("match_reports", []) or []
        case_chunks = state.get("case_chunks", []) or []
        case_summary = state.get("case_doc_summary", "")
        case_citation = (state.get("case_citation", "")
                         or state.get("case_cite_ref", ""))
        alert_meta = state.get("alert_metadata", {}) or {}
        alert_id = alert_meta.get("lni_id") or state.get("case_id") or "-"
        log = bind_alert(_log, alert_id, step="reasoning")

        if not match_reports:
            log.warning("no matched sections to reason over")
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text="No matched sections to generate suggestions for.")]
                ),
            )
            return

        all_suggestions: list[dict] = []
        total_suggestions = 0

        log.info("starting reasoning over %d match reports", len(match_reports))

        for report_idx, report in enumerate(match_reports):
            pg_doc_id = report["pg_doc_id"]
            pg_doc_title = report.get("pg_doc_title", "")
            matched_sections = report.get("matched_sections", [])

            log.info("[%d/%d] doc=%s sections=%d",
                     report_idx + 1, len(match_reports),
                     pg_doc_id, len(matched_sections))

            evidence_text = _format_evidence_for_reasoning(report, case_chunks)

            if evidence_text.startswith("(No specific") or evidence_text.startswith("(No matched"):
                log.warning("  no paragraph evidence -- skipping doc=%s", pg_doc_id)
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

                    if suggestion and isinstance(suggestion, dict):
                        ct = (suggestion.get("change_type", "") or "").strip().upper()
                        if ct not in ("UPDATE", "NEW", "REMOVE"):
                            ct = "UPDATE"
                        suggestion["change_type"] = ct

                    if (suggestion
                            and isinstance(suggestion, dict)
                            and suggestion.get("what_to_change", "").strip()
                            and suggestion.get("change_type", "").upper() != "NOTE"):

                        where_quote = (suggestion.get("where", "") or "").strip()
                        if where_quote and ct == "UPDATE":
                            sec_text_lower = section.get("section_text", "").lower()
                            if where_quote.lower() not in sec_text_lower:
                                words = where_quote.lower().split()
                                overlap = sum(1 for w in words if w in sec_text_lower)
                                if len(words) > 3 and overlap / len(words) < 0.5:
                                    log.warning(
                                        "  grounding failed section=%s -- quoted text not in PG section",
                                        section["section_id"],
                                    )
                                    continue
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
                        log.info("  section=%s no substantive suggestion",
                                 section["section_id"])

                except (json.JSONDecodeError, Exception) as exc:
                    log.error("  LLM error on section=%s: %s",
                              section["section_id"], exc)

            if doc_suggestions:
                all_suggestions.append({
                    "pg_doc_id": pg_doc_id,
                    "pg_doc_title": pg_doc_title,
                    "source_file": report.get("source_file", ""),
                    "section_suggestions": doc_suggestions,
                })
                log.info("  kept %d sections with substantive suggestions",
                         len(doc_suggestions))
            else:
                log.info("  no substantive suggestions -- doc filtered out")

        state["suggestions"] = all_suggestions
        log.info(
            "reasoning done total_raw=%d docs_with_suggestions=%d",
            total_suggestions, len(all_suggestions),
        )

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
