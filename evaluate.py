"""
evaluate.py -- Evaluation Framework for POC-1 Pipeline
=======================================================
Compares pipeline report output against labeled ground truth to compute
document-level Precision and Recall.

Usage:
    python evaluate.py data/reports/case123_2026-04-13.json
    python evaluate.py --report-dir data/reports/
    python evaluate.py --all

Ground truth is stored in data/ground_truth.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


DATA_DIR = Path(__file__).resolve().parent / "data"
GT_PATH = DATA_DIR / "ground_truth.json"
REPORTS_DIR = DATA_DIR / "reports"


def _load_ground_truth() -> list[dict]:
    if not GT_PATH.exists():
        print(f"  [Eval] Ground truth not found at {GT_PATH}")
        return []
    with open(GT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("cases", [])


def _load_report(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_document_level(
    predicted_doc_ids: list[str],
    expected_doc_ids: list[str],
) -> dict:
    """Compute document-level precision and recall."""
    pred_set = set(predicted_doc_ids)
    exp_set = set(expected_doc_ids)

    n_correct = len(pred_set & exp_set)
    n_retrieved = len(pred_set)
    n_expected = len(exp_set)

    precision = n_correct / n_retrieved if n_retrieved else 0.0
    recall = n_correct / n_expected if n_expected else 0.0

    return {
        "precision_pct": round(precision * 100, 1),
        "recall_pct": round(recall * 100, 1),
        "correct": n_correct,
        "retrieved": n_retrieved,
        "expected": n_expected,
        "predicted_docs": sorted(predicted_doc_ids),
        "expected_docs": sorted(expected_doc_ids),
    }


def evaluate_report(report: dict, ground_truth_entry: dict) -> dict:
    """
    Evaluate a single report against its ground truth.
    Returns document-level precision and recall only.
    """
    expected_docs = ground_truth_entry.get("expected_pg_docs", [])

    predicted_doc_ids = [
        d["pg_doc_id"] for d in report.get("impacted_documents", [])
    ]

    doc_metrics = evaluate_document_level(predicted_doc_ids, expected_docs)

    return {
        "case_id": report.get("case", {}).get("case_id", "unknown"),
        "document_level": doc_metrics,
    }


def _find_ground_truth(case_id: str, gt_list: list[dict]) -> Optional[dict]:
    for entry in gt_list:
        if entry.get("case_id") == case_id:
            return entry
        if entry.get("alert_xml") and case_id in entry["alert_xml"]:
            return entry
    return None


def _print_results(results: dict) -> None:
    case_id = results["case_id"]
    doc = results["document_level"]

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION: {case_id}")
    print("=" * 60)
    print()
    print("  DOCUMENT-LEVEL RETRIEVAL")
    print("  ------------------------")
    print(f"  Precision : {doc['precision_pct']}%  ({doc['correct']} correct out of {doc['retrieved']} retrieved)")
    print(f"  Recall    : {doc['recall_pct']}%  ({doc['correct']} correct out of {doc['expected']} expected)")
    print()
    print("=" * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate POC-1 pipeline reports against ground truth.")
    ap.add_argument("report_file", nargs="?", help="Path to a single report JSON file.")
    ap.add_argument("--report-dir", help="Evaluate all reports in this directory.")
    ap.add_argument("--all", action="store_true", help="Evaluate all reports in data/reports/.")
    ap.add_argument("--output", help="Save results JSON to this path.")
    args = ap.parse_args()

    gt_list = _load_ground_truth()
    if not gt_list:
        print("  [Eval] No ground truth entries found. Add cases to data/ground_truth.json first.")
        sys.exit(1)

    report_paths: list[Path] = []
    if args.report_file:
        report_paths.append(Path(args.report_file))
    elif args.report_dir:
        report_paths.extend(sorted(Path(args.report_dir).glob("*.json")))
    elif args.all:
        report_paths.extend(sorted(REPORTS_DIR.glob("*.json")))
    else:
        ap.print_help()
        sys.exit(1)

    if not report_paths:
        print("  [Eval] No report files found.")
        sys.exit(1)

    all_results: list[dict] = []

    for rp in report_paths:
        if not rp.is_file():
            print(f"  [Eval] Skipping {rp} -- not found")
            continue

        report = _load_report(rp)
        case_id = report.get("case", {}).get("case_id", "")

        gt = _find_ground_truth(case_id, gt_list)
        if not gt:
            print(f"  [Eval] No ground truth for case '{case_id}' -- skipping {rp.name}")
            continue

        results = evaluate_report(report, gt)
        results["report_file"] = str(rp)
        all_results.append(results)
        _print_results(results)

    if len(all_results) > 1:
        total_correct = sum(r["document_level"]["correct"] for r in all_results)
        total_retrieved = sum(r["document_level"]["retrieved"] for r in all_results)
        total_expected = sum(r["document_level"]["expected"] for r in all_results)
        agg_p = total_correct / total_retrieved * 100 if total_retrieved else 0.0
        agg_r = total_correct / total_expected * 100 if total_expected else 0.0

        print(f"\n{'=' * 60}")
        print(f"  AGGREGATE ACROSS {len(all_results)} CASES")
        print("=" * 60)
        print(f"  Precision : {agg_p:.1f}%  ({total_correct} correct / {total_retrieved} retrieved)")
        print(f"  Recall    : {agg_r:.1f}%  ({total_correct} correct / {total_expected} expected)")
        print("=" * 60)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n  [Eval] Results saved to {out}")


if __name__ == "__main__":
    main()
