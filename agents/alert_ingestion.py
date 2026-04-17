"""
agents/alert_ingestion.py -- Alert Ingestion Agent
====================================================
Reads an alert XML file path from session state and loads its content.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types


class AlertIngestionAgent(BaseAgent):
    """
    Reads the alert XML file specified in state['alert_xml_path'].
    Stores the raw XML content and file path in session state.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        xml_path_str = state.get("alert_xml_path", "")

        if not xml_path_str:
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text="ERROR: No alert_xml_path in session state.")]
                ),
            )
            return

        xml_path = Path(xml_path_str)
        if not xml_path.is_file():
            yield Event(
                author=self.name,
                content=types.Content(
                    parts=[types.Part(text=f"ERROR: Alert file not found: {xml_path}")]
                ),
            )
            return

        raw_xml = xml_path.read_text(encoding="utf-8", errors="ignore")
        state["alert_raw_xml"] = raw_xml
        state["alert_xml_path"] = str(xml_path.resolve())

        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=f"Alert file loaded: {xml_path.name} ({len(raw_xml)} chars)")]
            ),
            actions=EventActions(state_delta={
                "alert_raw_xml": raw_xml,
                "alert_xml_path": str(xml_path.resolve()),
            }),
        )


alert_ingestion_agent = AlertIngestionAgent(
    name="AlertIngestionAgent",
    description="Reads alert XML file and stores content in session state.",
)
