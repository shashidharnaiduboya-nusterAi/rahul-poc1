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

        case_excerpts = _build_case_excerpts(candidates, case_chunks)
        case_summary_brief = case_summary[:2_000] if case_summary else "(No summary)"

        filtered: list[dict] = []
        rejected = 0

        print(f"  [Guardrail] Evaluating {len(candidates)} candidate PG docs...")

        for i, doc in enumerate(candidates):
            pg_title = doc.get("doc_title", "")
            pg_summary = doc.get("doc_summary", "") or pg_title
            pg_practice = doc.get("practice_area", "")

            user_msg = GUARDRAIL_USER_TEMPLATE.format(
                case_excerpts=case_excerpts,
                case_summary_brief=case_summary_brief,
                case_citation=case_citation or "N/A",
                pg_doc_title=pg_title,
                pg_practice_area=pg_practice or "N/A",
                pg_doc_summary=pg_summary[:1_500],
            )

            try:
                raw = call_llm_json(
                    system=GUARDRAIL_SYSTEM,
                    user=user_msg,
                    model_type="fast",
                )
                result = json.loads(raw)

                is_relevant = result.get("is_relevant", False)
                confidence = result.get("confidence", "LOW")
                reason = result.get("reason", "")

                if is_relevant and confidence in ("HIGH", "MEDIUM"):
                    filtered.append(doc)
                    print(f"  [Guardrail]   [{i+1}] PASS ({confidence}): "
                          f"{pg_title[:50]} -- {reason[:80]}")
                else:
                    rejected += 1
                    print(f"  [Guardrail]   [{i+1}] REJECT ({confidence}): "
                          f"{pg_title[:50]} -- {reason[:80]}")

            except (json.JSONDecodeError, Exception) as exc:
                filtered.append(doc)
                print(f"  [Guardrail]   [{i+1}] ERROR (passing through): {exc}")

        state["candidate_pg_docs"] = filtered

        summary = (
            f"Guardrail: {len(filtered)} passed, {rejected} rejected "
            f"(from {len(candidates)} candidates)"
        )
        print(f"  [Guardrail] {summary}")

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=summary)]),
            actions=EventActions(state_delta={"candidate_pg_docs": filtered}),
        )


guardrail_agent = GuardrailAgent(
    name="GuardrailAgent",
    description="Pre-filters candidate PG docs using Banking & Finance relevance criteria.",
)
