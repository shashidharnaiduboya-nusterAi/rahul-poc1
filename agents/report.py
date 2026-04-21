"""
agents/report.py -- Report Agent
==================================
Aggregates all results into a final structured report for SME review.
Saves JSON report to data/reports/ for evaluation framework consumption.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

from tools.logging_setup import get_logger, bind_alert

_log = get_logger("agents.report")

REPORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "reports"


class ReportAgent(BaseAgent):
    """
    Aggregates all pipeline results into a structured final report.
    The report is stored in state['final_report'], saved to data/reports/,
    and printed to stdout.
    """

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        alert_meta = state.get("alert_metadata", {}) or {}
        alert_id = alert_meta.get("lni_id") or state.get("case_id") or "-"
        log = bind_alert(_log, alert_id, step="report")
        case_id = state.get("case_id", "unknown")
        case_summary = state.get("case_doc_summary", "")
        candidate_count = len(state.get("candidate_pg_docs", []))
        match_reports = state.get("match_reports", [])
        suggestions = state.get("suggestions", [])

        suggestions = [
            s for s in suggestions
            if any(
                sec.get("suggestion") or sec.get("suggestions")
                for sec in s.get("section_suggestions", [])
            )
        ]

        now = datetime.now()
        report = {
            "generated_at": now.isoformat(),
            "alert": {
                "lni_id": alert_meta.get("lni_id"),
                "court_name": alert_meta.get("court_name"),
                "date_of_decision": alert_meta.get("date_of_decision"),
                "jurisdiction": alert_meta.get("jurisdiction"),
                "practice_area": alert_meta.get("practice_area"),
                "news_summary": alert_meta.get("news_summary") or "",
            },
            "case": {
                "case_id": case_id,
                "summary_excerpt": case_summary[:3_000],
                "keywords": state.get("case_keywords", []),
            },
            "retrieval": {
                "total_candidates": candidate_count,
                "documents_with_matches": len(match_reports),
                "documents_with_suggestions": len(suggestions),
                "total_impacted_sections": sum(
                    len(s.get("section_suggestions", []))
                    for s in suggestions
                ),
            },
            "impacted_documents": [],
        }

        for suggestion_set in suggestions:
            pg_doc_id = suggestion_set["pg_doc_id"]
            pg_doc_title = suggestion_set.get("pg_doc_title", "")
            source_file = suggestion_set.get("source_file", "")
            section_items = suggestion_set.get("section_suggestions", [])

            doc_entry = {
                "pg_doc_id": pg_doc_id,
                "pg_doc_title": pg_doc_title,
                "source_file": source_file,
                "impacted_sections": [],
            }

            for sec in section_items:
                sug = sec.get("suggestion") or {}
                if not sug and sec.get("suggestions"):
                    sug = sec["suggestions"][0] if sec["suggestions"] else {}

                if not sug or not isinstance(sug, dict):
                    continue

                section_entry = {
                    "section_id": sec["section_id"],
                    "section_heading": sec.get("section_heading", ""),
                    "match_strength": sec.get("match_strength", ""),
                    "priority": sec.get("priority", ""),
                    "impact_summary": sec.get("summary", ""),
                    "change": {
                        "where": sug.get("where", ""),
                        "change_type": sug.get("change_type", ""),
                        "what_to_change": sug.get("what_to_change", ""),
                        "suggested_text": sug.get("suggested_text", ""),
                        "why": sug.get("why", ""),
                    },
                }

                doc_entry["impacted_sections"].append(section_entry)

            report["impacted_documents"].append(doc_entry)

        state["final_report"] = report

        # Save JSON report for evaluation framework
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in case_id)
            ts = now.strftime("%Y%m%d_%H%M%S")
            report_path = REPORTS_DIR / f"{safe_id}_{ts}.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            log.info("JSON saved path=%s", report_path)
        except Exception as exc:
            log.error("failed to save JSON: %s", exc)

        log.info(
            "report generated candidates=%d matches=%d suggestions=%d sections=%d",
            candidate_count,
            len(match_reports),
            len(suggestions),
            sum(len(s.get("section_suggestions", [])) for s in suggestions),
        )

        # Format human-readable report
        lines = [
            "=" * 70,
            "  POC-1 IMPACT ANALYSIS REPORT",
            "=" * 70,
            "",
            f"  Generated: {report['generated_at']}",
            "",
            "  ALERT INFORMATION",
            "  -----------------",
            f"  LNI ID       : {report['alert']['lni_id']}",
            f"  Court        : {report['alert']['court_name']}",
            f"  Date         : {report['alert']['date_of_decision']}",
            f"  Jurisdiction : {report['alert']['jurisdiction']}",
            f"  Practice Area: {report['alert']['practice_area']}",
            "",
            "  CASE INFORMATION",
            "  ----------------",
            f"  Case ID      : {report['case']['case_id']}",
            f"  Top Keywords : {', '.join(report['case']['keywords'][:10])}",
            "",
            "  RETRIEVAL SUMMARY",
            "  -----------------",
            f"  Total PG candidates       : {report['retrieval']['total_candidates']}",
            f"  Documents with matches    : {report['retrieval']['documents_with_matches']}",
            f"  Documents with suggestions: {report['retrieval']['documents_with_suggestions']}",
            f"  Total impacted sections   : {report['retrieval']['total_impacted_sections']}",
            "",
        ]

        if report["impacted_documents"]:
            lines.append("  IMPACTED PG DOCUMENTS")
            lines.append("  " + "-" * 50)

            for doc in report["impacted_documents"]:
                lines.append(f"\n  Document: {doc['pg_doc_title']}")
                lines.append(f"  ID      : {doc['pg_doc_id']}")
                lines.append(f"  Source  : {doc['source_file']}")

                for sec in doc["impacted_sections"]:
                    lines.append(f"\n    Section: [{sec['section_id']}] {sec['section_heading']}")
                    lines.append(f"    Match  : {sec['match_strength']} | Priority: {sec['priority']}")
                    lines.append(f"    Impact : {sec['impact_summary']}")

                    change = sec.get("change", {})
                    if change and change.get("what_to_change"):
                        lines.append(f"\n      Change Type: {change['change_type']}")
                        lines.append(f"      WHERE: {change['where']}")
                        lines.append(f"      WHAT : {change['what_to_change']}")
                        lines.append(f"      WHY  : {change['why']}")
        else:
            lines.append("  No impacted PG documents found.")

        lines.append("\n" + "=" * 70)
        lines.append("  END OF REPORT")
        lines.append("=" * 70)

        report_text = "\n".join(lines)

        yield Event(
            author=self.name,
            content=types.Content(parts=[types.Part(text=report_text)]),
            actions=EventActions(state_delta={"final_report": report}),
        )


report_agent = ReportAgent(
    name="ReportAgent",
    description="Aggregates all results into a final structured report for SME review.",
)
