"""
agents/alert_processing.py -- Alert Processing Agent
======================================================
Parses alert XML to extract metadata including:
  - cite_defs (normcite from CITE-DEF spans) and cite_refs
  - practice area (Banking and Financial)
  - key holdings (from XML structure + LLM fallback)
"""

from __future__ import annotations

import json
import re
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from tools.xml_parsers import parse_alert
from tools.logging_setup import get_logger, bind_alert

_log = get_logger("agents.alert_processing")

_KEY_HOLDINGS_SYSTEM = """\
You are a legal analyst. Given the text of a case news alert, extract:
1. key_holdings: A list of the key legal holdings or decisions mentioned.
2. summary: A concise summary (3-5 sentences) of what the case decided and why it matters.
3. key_phrases: Important legal phrases or terms from the alert.

Return ONLY a JSON object with these three fields. No markdown, no code fences.\
"""


def _try_parse_json(raw: str) -> dict:
    """Parse JSON from LLM response, handling code fences and partial JSON."""
    cleaned = raw.strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip().strip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


class AlertProcessingAgent(BaseAgent):
    """
    Parses alert XML and extracts structured metadata.
    Extracts key holdings from XML structure first, then uses LLM to enrich.
    cite_def = normcite attribute on CITE-DEF spans (the actual citation string).
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        xml_path = state.get("alert_xml_path", "")
        log = bind_alert(_log, "-", step="alert_processing")

        if not xml_path:
            log.error("no alert_xml_path in state")
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text="ERROR: No alert_xml_path in state.")]
                ),
            )
            return

        alert_data = parse_alert(xml_path)

        alert_metadata = {
            "lni_id": alert_data.get("lni_id"),
            "cite_defs": alert_data.get("cite_defs", []),
            "cite_refs": alert_data.get("cite_refs", []),
            "date_of_decision": alert_data.get("date_of_decision"),
            "court_name": alert_data.get("court_name"),
            "jurisdiction": alert_data.get("jurisdiction"),
            "practice_area": alert_data.get("practice_area"),
            "news_summary": alert_data.get("news_summary", ""),
            "source_file": alert_data.get("source_file", ""),
            "key_holdings": alert_data.get("xml_holdings", []),
            "key_phrases": [],
        }

        # Rebind logger with alert_id now that we know it
        alert_id = alert_metadata["lni_id"] or "-"
        log = bind_alert(_log, alert_id, step="alert_processing")
        log.info(
            "alert parsed lni=%s cite_defs=%d cite_refs=%d practice_area=%r "
            "news_summary_chars=%d xml_holdings=%d",
            alert_metadata["lni_id"],
            len(alert_metadata["cite_defs"]),
            len(alert_metadata["cite_refs"]),
            alert_metadata["practice_area"],
            len(alert_metadata["news_summary"]),
            len(alert_metadata["key_holdings"]),
        )
        log.debug("cite_defs=%s cite_refs=%s",
                  alert_metadata["cite_defs"], alert_metadata["cite_refs"])

        # Enrich with LLM-based key holdings extraction
        news_summary = alert_metadata["news_summary"]
        if news_summary and len(news_summary) > 50:
            try:
                from tools.llm_helper import call_llm
                result = call_llm(
                    system=_KEY_HOLDINGS_SYSTEM,
                    user=f"ALERT TEXT:\n{news_summary[:4_000]}",
                    model_type="fast",
                )
                parsed = _try_parse_json(result)
                if parsed:
                    llm_holdings = parsed.get("key_holdings", [])
                    if llm_holdings:
                        existing = set(alert_metadata["key_holdings"])
                        for h in llm_holdings:
                            if h and h not in existing:
                                alert_metadata["key_holdings"].append(h)
                                existing.add(h)

                    alert_metadata["key_phrases"] = parsed.get("key_phrases", [])
                    if parsed.get("summary"):
                        alert_metadata["llm_summary"] = parsed["summary"]
                    log.info(
                        "LLM extracted holdings=%d phrases=%d",
                        len(llm_holdings),
                        len(alert_metadata["key_phrases"]),
                    )
            except Exception as exc:
                log.warning(
                    "LLM extraction failed: %s (xml_holdings still available: %d)",
                    exc, len(alert_metadata["key_holdings"]),
                )
        else:
            log.info("news summary too short (%d chars) -- xml holdings only",
                     len(news_summary))

        state["alert_metadata"] = alert_metadata

        summary_lines = [
            f"LNI: {alert_metadata['lni_id']}",
            f"Court: {alert_metadata['court_name']}",
            f"Jurisdiction: {alert_metadata['jurisdiction']}",
            f"Practice Area: {alert_metadata['practice_area']}",
            f"cite_defs (normcite): {alert_metadata['cite_defs']}",
            f"cite_refs: {alert_metadata['cite_refs']}",
            f"News summary: {len(news_summary)} chars",
            f"Key holdings: {len(alert_metadata['key_holdings'])} extracted",
        ]

        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text="Alert metadata extracted:\n" + "\n".join(summary_lines))]
            ),
            actions=EventActions(state_delta={"alert_metadata": alert_metadata}),
        )


alert_processing_agent = AlertProcessingAgent(
    name="AlertProcessingAgent",
    description="Parses alert XML, extracts cite_defs (normcite), practice area, and key holdings.",
)
