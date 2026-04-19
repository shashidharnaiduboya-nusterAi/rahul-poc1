"""
run.py -- CLI Runner for POC-1 Pipeline
=========================================
Runs the full ADK agent pipeline for a single alert XML file.
After the pipeline completes, prints document-level Precision and Recall
if ground truth is available.

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

DATA_DIR = Path(__file__).resolve().parent / "data"
GT_PATH = DATA_DIR / "ground_truth.json"


def _load_ground_truth() -> list[dict]:
    if not GT_PATH.exists():
        return []
    with open(GT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("cases", [])


def _print_retrieval_metrics(final_report: dict) -> None:
    """
    If ground truth exists for this case, compute and print
    document-level Precision and Recall (nothing else).
    """
    gt_list = _load_ground_truth()
    if not gt_list:
        return

    case_id = final_report.get("case", {}).get("case_id", "")
    lni_id = final_report.get("alert", {}).get("lni_id", "")

    gt_entry = None
    for entry in gt_list:
        if entry.get("case_id") == case_id:
            gt_entry = entry
            break
        if entry.get("alert_xml") and case_id in entry["alert_xml"]:
            gt_entry = entry
            break
        if entry.get("lni_id") and entry["lni_id"] == lni_id:
            gt_entry = entry
            break

    if not gt_entry:
        return

    expected_docs = set(gt_entry.get("expected_pg_docs", []))
    predicted_docs = set(
        d["pg_doc_id"] for d in final_report.get("impacted_documents", [])
    )

    n_correct = len(predicted_docs & expected_docs)
    n_retrieved = len(predicted_docs)
    n_expected = len(expected_docs)

    precision = n_correct / n_retrieved * 100 if n_retrieved else 0.0
    recall = n_correct / n_expected * 100 if n_expected else 0.0

    print()
    print("=" * 60)
    print("  RETRIEVAL ACCURACY")
    print("=" * 60)
    print(f"  Precision : {precision:.1f}%  ({n_correct} correct out of {n_retrieved} retrieved)")
    print(f"  Recall    : {recall:.1f}%  ({n_correct} correct out of {n_expected} expected)")
    print("=" * 60)


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

        _print_retrieval_metrics(final_report)

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
