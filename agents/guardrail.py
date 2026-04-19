"""
agents/guardrail.py -- Relevance Guardrail Agent
==================================================
Sits between Retrieval and Matching. Uses a fast LLM call per candidate PG doc
to determine genuine relevance before expensive section-level analysis.

Eliminates the "high recall, low precision" problem by applying domain-specific
Banking & Finance relevance criteria as a pre-filter.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from tools.llm_helper import call_llm_json
from prompts.guardrail import GUARDRAIL_SYSTEM, GUARDRAIL_USER_TEMPLATE


def _build_case_excerpts(candidates: list[dict], case_chunks: list[dict]) -> str:
    """Build a combined case excerpt string from the top matched paragraphs."""
    seen_chunks: set[int] = set()
    lines: list[str] = []

    for doc in candidates[:5]:
        for mp in doc.get("matched_paragraphs", [])[:3]:
            ci = mp.get("chunk_index", -1)
            if ci in seen_chunks or ci < 0:
                continue
            seen_chunks.add(ci)

            text = mp.get("text_preview", "")
            if 0 <= ci < len(case_chunks):
                chunk = case_chunks[ci]
                text = chunk.get("text", text) if isinstance(chunk, dict) else text
            if text:
                lines.append(f"[Para {ci + 1}] {text[:600]}")

        if len(lines) >= 8:
            break

    return "\n\n".join(lines) if lines else "(No case excerpts available)"


class GuardrailAgent(BaseAgent):
    """
    Pre-filters candidate PG documents using domain-specific relevance criteria.
    Only documents passing the guardrail proceed to section-level matching.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        candidates = state.get("candidate_pg_docs", [])
        case_chunks = state.get("case_chunks", [])
        case_summary = state.get("case_doc_summary", "")
        case_citation = (state.get("case_citation", "")
                         or state.get("case_cite_ref", ""))

        if not candidates:
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text="No candidates to filter.")]
                ),
            )
            return

        # Guardrail disabled — pass all candidates through to matching
        filtered = list(candidates)
        print(f"  [Guardrail] Passing all {len(filtered)} candidates through (guardrail disabled)")

        state["candidate_pg_docs"] = filtered

        summary = f"Guardrail: {len(filtered)} candidates passed through (guardrail disabled)"

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=summary)]),
            actions=EventActions(state_delta={"candidate_pg_docs": filtered}),
        )


guardrail_agent = GuardrailAgent(
    name="GuardrailAgent",
    description="Pre-filters candidate PG docs using Banking & Finance relevance criteria.",
)
