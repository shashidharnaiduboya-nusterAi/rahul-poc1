"""
evaluate.py -- Evaluation Framework for POC-1 Pipeline
=======================================================
Compares pipeline report output against labeled ground truth to compute
document-level and section-level Precision, Recall, and F1 scores.

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


def _precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_document_level(
    predicted_doc_ids: list[str],
    expected_doc_ids: list[str],
    total_pg_docs: int,
) -> dict:
    """Compute document-level precision, recall, F1."""
    pred_set = set(predicted_doc_ids)
    exp_set = set(expected_doc_ids)

    tp = len(pred_set & exp_set)
    fp = len(pred_set - exp_set)
    fn = len(exp_set - pred_set)
    tn = total_pg_docs - tp - fp - fn

    metrics = _precision_recall_f1(tp, fp, fn)
    metrics["tn"] = max(tn, 0)
    metrics["accuracy"] = round(
        (tp + max(tn, 0)) / total_pg_docs, 4
    ) if total_pg_docs > 0 else 0.0
    metrics["predicted_docs"] = sorted(predicted_doc_ids)
    metrics["expected_docs"] = sorted(expected_doc_ids)
    metrics["true_positives"] = sorted(pred_set & exp_set)
    metrics["false_positives"] = sorted(pred_set - exp_set)
    metrics["false_negatives"] = sorted(exp_set - pred_set)
    return metrics


def evaluate_section_level(
    predicted_sections: dict[str, list[str]],
    expected_sections: dict[str, list[str]],
) -> dict:
    """
    Compute section-level precision, recall, F1 across all PG docs.

    predicted_sections: {doc_id: [section_id, ...]}
    expected_sections: {doc_id: [section_id, ...]}
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_doc: dict[str, dict] = {}

    all_doc_ids = set(predicted_sections.keys()) | set(expected_sections.keys())

    for doc_id in all_doc_ids:
        pred = set(predicted_sections.get(doc_id, []))
        exp = set(expected_sections.get(doc_id, []))

        tp = len(pred & exp)
        fp = len(pred - exp)
        fn = len(exp - pred)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_doc[doc_id] = _precision_recall_f1(tp, fp, fn)
        per_doc[doc_id]["predicted"] = sorted(pred)
        per_doc[doc_id]["expected"] = sorted(exp)

    aggregate = _precision_recall_f1(total_tp, total_fp, total_fn)
    return {
        "aggregate": aggregate,
        "per_document": per_doc,
    }


def evaluate_report(report: dict, ground_truth_entry: dict) -> dict:
    """
    Full evaluation of a single report against its ground truth.
    Returns doc-level and section-level metrics.
    """
    expected_docs = ground_truth_entry.get("expected_pg_docs", [])
    expected_sections = ground_truth_entry.get("expected_sections", {})
    total_pg_docs = ground_truth_entry.get("total_pg_docs", 100)

    predicted_doc_ids = [
        d["pg_doc_id"] for d in report.get("impacted_documents", [])
    ]

    predicted_sections: dict[str, list[str]] = {}
    for doc in report.get("impacted_documents", []):
        doc_id = doc["pg_doc_id"]
        section_ids = [
            sec["section_id"]
            for sec in doc.get("impacted_sections", [])
        ]
        if section_ids:
            predicted_sections[doc_id] = section_ids

    doc_metrics = evaluate_document_level(
        predicted_doc_ids, expected_docs, total_pg_docs
    )
    sec_metrics = evaluate_section_level(predicted_sections, expected_sections)

    return {
        "case_id": report.get("case", {}).get("case_id", "unknown"),
        "document_level": doc_metrics,
        "section_level": sec_metrics,
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
    sec = results["section_level"]

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION: {case_id}")
    print("=" * 60)
    print()
    print("  DOCUMENT-LEVEL METRICS")
    print("  ----------------------")
    print(f"  Precision : {doc['precision']:.2%}  ({doc['tp']} TP, {doc['fp']} FP)")
    print(f"  Recall    : {doc['recall']:.2%}  ({doc['tp']} TP, {doc['fn']} FN)")
    print(f"  F1 Score  : {doc['f1']:.2%}")
    print(f"  Accuracy  : {doc['accuracy']:.2%}  ({doc['tp']+doc['tn']} correct / total)")
    if doc["true_positives"]:
        print(f"  Correct   : {', '.join(doc['true_positives'])}")
    if doc["false_positives"]:
        print(f"  Extra     : {', '.join(doc['false_positives'])}")
    if doc["false_negatives"]:
        print(f"  Missed    : {', '.join(doc['false_negatives'])}")

    agg = sec["aggregate"]
    print()
    print("  SECTION-LEVEL METRICS (aggregate)")
    print("  ----------------------------------")
    print(f"  Precision : {agg['precision']:.2%}  ({agg['tp']} TP, {agg['fp']} FP)")
    print(f"  Recall    : {agg['recall']:.2%}  ({agg['tp']} TP, {agg['fn']} FN)")
    print(f"  F1 Score  : {agg['f1']:.2%}")

    if sec["per_document"]:
        print()
        print("  PER-DOCUMENT SECTION BREAKDOWN")
        print("  ------------------------------")
        for doc_id, m in sec["per_document"].items():
            print(f"  {doc_id}: P={m['precision']:.2%} R={m['recall']:.2%} F1={m['f1']:.2%}"
                  f"  (predicted={m['predicted']}, expected={m['expected']})")

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
        total_doc_tp = sum(r["document_level"]["tp"] for r in all_results)
        total_doc_fp = sum(r["document_level"]["fp"] for r in all_results)
        total_doc_fn = sum(r["document_level"]["fn"] for r in all_results)
        total_sec_tp = sum(r["section_level"]["aggregate"]["tp"] for r in all_results)
        total_sec_fp = sum(r["section_level"]["aggregate"]["fp"] for r in all_results)
        total_sec_fn = sum(r["section_level"]["aggregate"]["fn"] for r in all_results)

        doc_agg = _precision_recall_f1(total_doc_tp, total_doc_fp, total_doc_fn)
        sec_agg = _precision_recall_f1(total_sec_tp, total_sec_fp, total_sec_fn)

        print(f"\n{'=' * 60}")
        print(f"  AGGREGATE ACROSS {len(all_results)} CASES")
        print("=" * 60)
        print(f"  Doc-level:  P={doc_agg['precision']:.2%}  R={doc_agg['recall']:.2%}  F1={doc_agg['f1']:.2%}")
        print(f"  Sec-level:  P={sec_agg['precision']:.2%}  R={sec_agg['recall']:.2%}  F1={sec_agg['f1']:.2%}")
        print("=" * 60)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n  [Eval] Results saved to {out}")


if __name__ == "__main__":
    main()
