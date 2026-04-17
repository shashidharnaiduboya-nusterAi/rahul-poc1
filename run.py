"""
run.py -- CLI Runner for POC-1 Pipeline
=========================================
Runs the full ADK agent pipeline for a single alert XML file.

Usage:
    python3 run.py <path_to_alert_xml>

Example:
    python3 run.py /data/alerts/CaseNewsAlert_5KWP-JC41-DYHT-G0FR-00000-00.xml
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


async def run_pipeline(alert_xml_path: str) -> dict:
    """Run the full POC-1 pipeline for a single alert."""
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from agent import root_agent

    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="poc1",
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name="poc1",
        user_id="cli_user",
        state={"alert_xml_path": alert_xml_path},
    )

    print(f"\nStarting POC-1 pipeline for: {alert_xml_path}")
    print("=" * 60)

    content = types.Content(
        role="user",
        parts=[types.Part(text=f"Process alert: {alert_xml_path}")],
    )

    final_report = {}
    async for event in runner.run_async(
        user_id="cli_user",
        session_id=session.id,
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(f"\n[{event.author}]")
                    print(part.text)

        if event.actions and event.actions.state_delta:
            if "final_report" in event.actions.state_delta:
                final_report = event.actions.state_delta["final_report"]

    # Save report to JSON
    if final_report:
        report_path = Path("data") / "reports"
        report_path.mkdir(parents=True, exist_ok=True)
        lni = final_report.get("alert", {}).get("lni_id", "unknown")
        out_file = report_path / f"report_{lni}.json"
        out_file.write_text(json.dumps(final_report, indent=2, default=str))
        print(f"\nReport saved to: {out_file}")

    return final_report


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 run.py <path_to_alert_xml>")
        print("Example: python3 run.py /data/alerts/CaseNewsAlert_5KWP-JC41.xml")
        sys.exit(1)

    alert_path = sys.argv[1]
    if not Path(alert_path).is_file():
        print(f"ERROR: File not found -- {alert_path}")
        sys.exit(1)

    asyncio.run(run_pipeline(alert_path))


if __name__ == "__main__":
    main()
