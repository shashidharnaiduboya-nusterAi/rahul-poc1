"""
agents/filter_agent.py -- Filter Agent (Optional)
===================================================
Filters alerts by practice area. For POC-1, scope is Banking & Finance.
Passes through all alerts if no filter criteria fail.
"""

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

ALLOWED_PRACTICE_AREAS = frozenset({
    "banking", "finance", "banking & finance", "banking and finance",
    "b&f", "restructuring", "insolvency",
})


class FilterAgent(BaseAgent):
    """
    Checks alert metadata for practice area filtering.
    Sets state['should_process'] = True/False.
    For POC-1 this is a passthrough -- all alerts proceed.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        alert_meta = state.get("alert_metadata", {})

        practice_area = (alert_meta.get("practice_area") or "").lower().strip()

        should_process = True
        reason = "No filter applied (POC-1 passthrough)"

        if practice_area:
            if any(pa in practice_area for pa in ALLOWED_PRACTICE_AREAS):
                reason = f"Practice area '{practice_area}' matches Banking & Finance scope"
            else:
                reason = f"Practice area '{practice_area}' -- proceeding anyway (POC-1)"

        state["should_process"] = should_process

        yield Event(
            author=self.name,
            content=types.Content(
                parts=[types.Part(text=f"Filter: {reason}. should_process={should_process}")]
            ),
            actions=EventActions(state_delta={"should_process": should_process}),
        )


filter_agent = FilterAgent(
    name="FilterAgent",
    description="Optional filter by practice area. POC-1: passthrough.",
)
