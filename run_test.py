"""
run_test.py -- Standalone E2E Retrieval Test Runner
====================================================
Pipeline:
  1. Parse XLSX ground truth (alert → expected PG docs)
  2. Index all B&F PG docs (embedding + BM25)
  3. Hybrid retrieval: semantic + BM25 combined scoring
  4. Cross-encoder reranking
  5. Report: for each alert, show expected vs retrieved docs

Usage:
    python3 run_test.py
    python3 run_test.py --top-k 30 --threshold 0.15 --max-alerts 10
"""

from __future__ import annotations

import argparse, json, math, os, re, sys, time, uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import openpyxl
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
XLSX_PATH = BASE_DIR / "ACP PG Golden Data Set.xlsx"
PG_DOCS_DIR = BASE_DIR / "ACP-PG_3PAs_OV-PN-CL-SF" / "BANKINGANDFINANCE"
TEST_QDRANT_DIR = BASE_DIR / "data" / "test_qdrant"
REPORT_DIR = BASE_DIR / "data" / "test_reports"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
COLLECTION_NAME = "test_pg_docs"

_STOPWORDS = frozenset(
    "the a an and or but in on at to for of with by from as is was are were be "
    "been being have has had do does did will would could should may might shall "
    "that this these those it its their they he she we i you not no".split()
)


# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════
@dataclass
class GroundTruthEntry:
    alert_lni: str; alert_title: str; key_terms: str; pg_title: str
    echo_id: str; pg_type: str; section_heading: str
    complexity: str; rationale: str

@dataclass
class PGDoc:
    doc_id: str; echo_id: str; title: str; practice_area: str
    jurisdiction: str; sections: list[dict] = field(default_factory=list)
    full_text: str = ""; source_file: str = ""

@dataclass
class AlertResult:
    alert_lni: str; alert_title: str; key_terms: str
    expected_doc_ids: set; expected_sections: dict
    predicted_doc_ids: set = field(default_factory=set)
    predicted_sections: dict = field(default_factory=dict)
    retrieval_scores: dict = field(default_factory=dict)
    guardrail_kept: set = field(default_factory=set)
    doc_tp: int = 0; doc_fp: int = 0; doc_fn: int = 0
    sec_tp: int = 0; sec_fp: int = 0; sec_fn: int = 0


# ═══════════════════════════════════════════════════════════════════
# 1. XLSX Ground Truth
# ═══════════════════════════════════════════════════════════════════
def parse_ground_truth(xlsx_path: Path):
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True)
    entries = []
    ws = wb["Banking and Finance Alerts"]
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, values_only=True):
        vals = list(row) + [None] * 22
        if not vals[3]: continue
        echo_raw = vals[7]
        eid = str(int(echo_raw)) if isinstance(echo_raw, (int, float)) else str(echo_raw or "")
        entries.append(GroundTruthEntry(
            alert_lni=str(vals[3]).strip(), alert_title=str(vals[2] or ""),
            key_terms=str(vals[5] or ""), pg_title=str(vals[6] or ""),
            echo_id=eid, pg_type=str(vals[10] or ""),
            section_heading=str(vals[12] or ""), complexity=str(vals[13] or ""),
            rationale=str(vals[14] or "")))
    no_impact = []
    if "B and F Alerts No impact" in wb.sheetnames:
        ws2 = wb["B and F Alerts No impact"]
        for row in ws2.iter_rows(min_row=2, max_row=ws2.max_row, values_only=True):
            vals = list(row) + [None] * 5
            if vals[2]: no_impact.append(str(vals[2]).strip())
    wb.close()
    return entries, no_impact

def build_alert_ground_truth(entries):
    alerts = {}
    for e in entries:
        if e.alert_lni not in alerts:
            alerts[e.alert_lni] = {"title": e.alert_title, "key_terms": e.key_terms,
                                   "expected_docs": {}, "expected_sections": {}}
        a = alerts[e.alert_lni]
        if e.echo_id:
            a["expected_docs"][e.echo_id] = e.pg_title
            a["expected_sections"].setdefault(e.echo_id, [])
            if e.section_heading:
                a["expected_sections"][e.echo_id].append(e.section_heading)
    return alerts


# ═══════════════════════════════════════════════════════════════════
# 2. PG Doc XML Parser
# ═══════════════════════════════════════════════════════════════════
def _et(elem):
    parts = []
    if elem.text and elem.text.strip(): parts.append(elem.text.strip())
    for ch in elem:
        t = _et(ch)
        if t: parts.append(t)
        if ch.tail and ch.tail.strip(): parts.append(ch.tail.strip())
    return " ".join(parts)

def _ln(tag): return tag.split("}", 1)[1] if "}" in tag else tag

def parse_pg_xml(xml_path: Path) -> Optional[PGDoc]:
    import xml.etree.ElementTree as ET
    try: root = ET.parse(str(xml_path)).getroot()
    except Exception: return None
    echo_id = xml_path.stem.split("_")[-1]
    doc_id = echo_id
    title = ""
    for el in root.iter():
        if _ln(el.tag) == "front":
            for ch in el:
                if _ln(ch.tag) == "title":
                    for s in ch:
                        if _ln(s.tag) == "text" and s.text: title = s.text.strip(); break
            break
    if not title:
        for el in root.iter():
            if _ln(el.tag) == "document-title" and el.text: title = el.text.strip(); break
    sections, si = [], 0
    for el in root.iter():
        if _ln(el.tag) in ("clause", "inclusion", "section"):
            heading = ""
            for ch in el:
                if _ln(ch.tag) in ("heading", "title", "text"):
                    h = _et(ch).strip()
                    if h and len(h) < 300: heading = h; break
            text = _et(el).strip()
            if text and len(text) > 30:
                sections.append({"id": f"sec_{si}", "heading": heading, "text": text[:3000]}); si += 1
    if not sections:
        for el in root.iter():
            if _ln(el.tag) in ("para", "pgrp", "p"):
                text = _et(el).strip()
                if text and len(text) > 30:
                    sections.append({"id": f"para_{si}", "heading": "", "text": text[:3000]}); si += 1
    headings = [s["heading"] for s in sections if s.get("heading")]
    heading_block = " | ".join(headings[:50])
    lead_paras = [s["text"][:300] for s in sections[:10]]
    full_text = f"{title}\n{title}\n\nSections: {heading_block}\n\n" + "\n\n".join(lead_paras)
    full_text = full_text[:5000]
    return PGDoc(doc_id=doc_id, echo_id=echo_id, title=title, practice_area="",
                 jurisdiction="", sections=sections, full_text=full_text,
                 source_file=str(xml_path))


# ═══════════════════════════════════════════════════════════════════
# 3. BM25 Index
# ═══════════════════════════════════════════════════════════════════
def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z]{2,}", text.lower()) if t not in _STOPWORDS]

class BM25Index:
    def __init__(self, docs: list[PGDoc]):
        self.echo_ids = [d.echo_id for d in docs]
        corpus = [_tokenize(d.full_text) for d in docs]
        self.bm25 = BM25Okapi(corpus)
        self._echo_to_idx = {eid: i for i, eid in enumerate(self.echo_ids)}

    def query(self, text: str, top_k: int = 50) -> dict[str, float]:
        tokens = _tokenize(text)
        if not tokens:
            return {}
        scores = self.bm25.get_scores(tokens)
        max_s = scores.max() if scores.max() > 0 else 1.0
        top_idx = np.argsort(scores)[::-1][:top_k]
        return {self.echo_ids[i]: float(scores[i] / max_s) for i in top_idx if scores[i] > 0}


# ═══════════════════════════════════════════════════════════════════
# 4. Embedding + Qdrant Indexing
# ═══════════════════════════════════════════════════════════════════
def index_pg_docs(docs, qdrant_path, batch_size=64):
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    print("[Index] Loading all-MiniLM-L6-v2 (384-dim)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    qdrant_path.mkdir(parents=True, exist_ok=True)
    qc = QdrantClient(path=str(qdrant_path))
    if COLLECTION_NAME in [c.name for c in qc.get_collections().collections]:
        qc.delete_collection(COLLECTION_NAME)
    qc.create_collection(COLLECTION_NAME,
                         vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    texts = [d.full_text for d in docs]
    total = len(texts)
    print(f"[Index] Embedding {total} docs...", flush=True)
    all_vecs = []
    for s in range(0, total, batch_size):
        e = min(s + batch_size, total)
        all_vecs.extend(model.encode(texts[s:e], show_progress_bar=False, batch_size=batch_size))
        print(f"  {e}/{total}", flush=True)
    points = [PointStruct(id=str(uuid.uuid4()), vector=v.tolist(),
              payload={"doc_id": d.doc_id, "echo_id": d.echo_id, "title": d.title,
                       "source_file": d.source_file, "section_count": len(d.sections)})
              for d, v in zip(docs, all_vecs)]
    for s in range(0, len(points), 100):
        qc.upsert(COLLECTION_NAME, points[s:s+100])
    print(f"[Index] {qc.get_collection(COLLECTION_NAME).points_count} points indexed", flush=True)
    qc.close()
    return model


# ═══════════════════════════════════════════════════════════════════
# 5. Multi-Query Hybrid Retrieval + Keyword Reranker + Cross-Encoder
# ═══════════════════════════════════════════════════════════════════
def _semantic_query(model, qdrant_path, query_text, limit):
    from qdrant_client import QdrantClient
    vec = model.encode(query_text).tolist()
    qc = QdrantClient(path=str(qdrant_path))
    res = qc.query_points(COLLECTION_NAME, query=vec, limit=limit)
    qc.close()
    scores, payloads = {}, {}
    for pt in res.points:
        eid = pt.payload.get("echo_id", "")
        scores[eid] = pt.score
        payloads[eid] = pt.payload
    return scores, payloads


def _keyword_overlap_score(alert_terms: str, doc_title: str, doc_text: str) -> float:
    """Fraction of alert key-term tokens found in doc title/text."""
    terms = set(_tokenize(alert_terms))
    if not terms:
        return 0.0
    title_tokens = set(_tokenize(doc_title))
    text_tokens = set(_tokenize(doc_text[:2000]))
    title_hits = terms & title_tokens
    text_hits = terms & text_tokens
    title_frac = len(title_hits) / len(terms) if terms else 0
    text_frac = len(text_hits) / len(terms) if terms else 0
    return 0.6 * title_frac + 0.4 * text_frac


def hybrid_retrieve(model, qdrant_path, bm25_idx: BM25Index,
                    alert_title, key_terms, top_k=30, threshold=0.15,
                    alpha=0.6, echo_to_doc=None):
    """Multi-query hybrid retrieval with keyword reranking."""
    fetch_limit = top_k * 3

    q_combined = f"{alert_title} {key_terms}"
    q_title = alert_title
    q_terms = key_terms

    sem1, pl1 = _semantic_query(model, qdrant_path, q_combined, fetch_limit)
    sem2, pl2 = _semantic_query(model, qdrant_path, q_title, fetch_limit)
    sem3, pl3 = _semantic_query(model, qdrant_path, q_terms, fetch_limit)

    all_eids = set(sem1) | set(sem2) | set(sem3)
    payload_cache = {**pl3, **pl2, **pl1}

    sem_max = {}
    for eid in all_eids:
        sem_max[eid] = max(sem1.get(eid, 0), sem2.get(eid, 0), sem3.get(eid, 0))

    bm25_scores = bm25_idx.query(q_combined, top_k=fetch_limit)
    all_eids |= set(bm25_scores)

    combined = {}
    for eid in all_eids:
        ss = sem_max.get(eid, 0.0)
        bs = bm25_scores.get(eid, 0.0)
        base = alpha * ss + (1 - alpha) * bs

        doc = echo_to_doc.get(eid) if echo_to_doc else None
        doc_title = payload_cache.get(eid, {}).get("title", "")
        doc_text = doc.full_text if doc else doc_title
        kw_score = _keyword_overlap_score(key_terms, doc_title, doc_text)
        combined[eid] = base + 0.15 * kw_score

    ranked = sorted(combined.items(), key=lambda x: -x[1])[:top_k]
    hits = []
    for eid, score in ranked:
        if score < threshold:
            continue
        pl = payload_cache.get(eid, {})
        hits.append({
            "echo_id": eid, "doc_id": pl.get("doc_id", ""),
            "title": pl.get("title", ""), "score": round(score, 4),
            "sem_score": round(sem_max.get(eid, 0), 4),
            "bm25_score": round(bm25_scores.get(eid, 0), 4),
            "source_file": pl.get("source_file", ""),
        })
    return hits


_cross_encoder = None
def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print("  [CrossEncoder] Loading ms-marco-MiniLM-L-6-v2...", flush=True)
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def cross_encoder_rerank(alert_text: str, hits: list[dict],
                         echo_to_doc: dict, top_k: int = 10) -> list[dict]:
    """Rerank retrieval hits using a cross-encoder for precise relevance."""
    if not hits:
        return []
    ce = _get_cross_encoder()
    pairs = []
    for h in hits:
        doc = echo_to_doc.get(h["echo_id"])
        doc_text = doc.full_text[:512] if doc else h.get("title", "")
        pairs.append((alert_text, doc_text))

    ce_scores = ce.predict(pairs)
    ce_min = float(min(ce_scores))
    ce_max = float(max(ce_scores))
    ce_range = ce_max - ce_min if ce_max > ce_min else 1.0

    for h, cs in zip(hits, ce_scores):
        ce_norm = (float(cs) - ce_min) / ce_range
        h["ce_score"] = round(float(cs), 4)
        h["ce_norm"] = round(ce_norm, 4)
        h["retrieval_score"] = h["score"]
        h["score"] = round(0.35 * h["retrieval_score"] + 0.65 * ce_norm, 4)

    hits.sort(key=lambda x: -x["score"])
    return hits[:top_k]


# ═══════════════════════════════════════════════════════════════════
# 6. Groq LLM Helpers (thread-safe via per-thread clients)
# ═══════════════════════════════════════════════════════════════════
import threading
_groq_local = threading.local()

def _get_groq() -> Groq:
    if not hasattr(_groq_local, "client"):
        _groq_local.client = Groq(api_key=GROQ_API_KEY)
    return _groq_local.client

def _groq_call(system: str, user: str, max_tokens: int = 250) -> str:
    client = _get_groq()
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0, max_completion_tokens=max_tokens)
            raw = resp.choices[0].message.content.strip()
            return re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        except Exception as exc:
            if attempt < 3 and ("rate" in str(exc).lower() or "429" in str(exc)):
                wait = 3 * (2 ** attempt)
                time.sleep(wait)
            else:
                return ""
    return ""


# ═══════════════════════════════════════════════════════════════════
# 7. Guardrail (Groq LLM pre-filter)
# ═══════════════════════════════════════════════════════════════════
GUARDRAIL_SYS = """\
You are a senior Banking & Finance legal analyst performing a relevance pre-filter.
Given a case alert and a PG document title + summary, decide if this PG document
COULD PLAUSIBLY need updating due to the case.

Respond ONLY with JSON: {"relevant": true/false, "reason": "one sentence"}

Be GENEROUS at this stage — only reject if the PG doc is clearly unrelated to the
alert's legal topic. When in doubt, say true."""

def _guardrail_one(alert_title, key_terms, doc_title, doc_text_snippet):
    user = (f"ALERT: {alert_title}\nKEY TERMS: {key_terms}\n\n"
            f"PG DOCUMENT: {doc_title}\nSNIPPET: {doc_text_snippet[:800]}\n\n"
            f"Is this PG document plausibly relevant?")
    raw = _groq_call(GUARDRAIL_SYS, user, 120)
    try:
        r = json.loads(raw)
        return r.get("relevant", True)
    except Exception:
        return True  # pass through on error

def guardrail_filter(alert_title, key_terms, hits, echo_to_doc, workers=6):
    """Sequential guardrail with rate-limit pacing."""
    if not hits:
        return []
    kept = []
    for h in hits:
        doc = echo_to_doc.get(h["echo_id"])
        snippet = doc.full_text[:800] if doc else h.get("title", "")
        if _guardrail_one(alert_title, key_terms, h.get("title", ""), snippet):
            kept.append(h)
        time.sleep(0.5)
    kept.sort(key=lambda x: -x["score"])
    return kept


# ═══════════════════════════════════════════════════════════════════
# 8. Section Matching (Groq LLM)
# ═══════════════════════════════════════════════════════════════════
SEC_MATCH_SYS = """\
You are a senior Banking & Finance legal analyst. Given a case alert and a PG document section,
determine if this specific section needs updating due to the case.

Respond ONLY with JSON: {"is_impacted": true/false, "confidence": "HIGH"|"MEDIUM"|"LOW", "reason": "one sentence"}

STRICT: Only mark impacted if the case DIRECTLY affects this section's legal content.
DEFAULT to false unless clearly relevant."""

def _match_one_section(alert_title, key_terms, pg_title, sec):
    heading = sec.get("heading", "(no heading)")
    text = sec.get("text", "")[:1200]
    if len(text) < 30: return None
    user = (f"ALERT: {alert_title}\nKEY TERMS: {key_terms}\n\n"
            f"PG DOCUMENT: {pg_title}\nSECTION: {heading}\n"
            f"TEXT: {text}\n\nIs this section impacted?")
    raw = _groq_call(SEC_MATCH_SYS, user, 150)
    try:
        r = json.loads(raw)
        if r.get("is_impacted") and r.get("confidence") in ("HIGH", "MEDIUM"):
            return {"section_id": sec["id"], "heading": heading,
                    "confidence": r["confidence"], "reason": r.get("reason", "")}
    except Exception: pass
    return None

def match_sections_sequential(alert_title, key_terms, pg_title, sections):
    matched = []
    secs = [s for s in sections[:20] if len(s.get("text", "")) > 30]
    for s in secs:
        r = _match_one_section(alert_title, key_terms, pg_title, s)
        if r: matched.append(r)
        time.sleep(0.5)
    return matched




# ═══════════════════════════════════════════════════════════════════
# 10. Main Pipeline
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=0.7, help="Semantic weight (1-alpha = BM25)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-guardrail", action="store_true")
    ap.add_argument("--skip-sections", action="store_true")
    ap.add_argument("--skip-rerank", action="store_true", help="Skip cross-encoder reranking")
    ap.add_argument("--max-alerts", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    P = lambda *a, **kw: print(*a, **kw, flush=True)
    P("=" * 70)
    P("  POC-1 Retrieval Test")
    P("=" * 70)

    # ── 1. Ground truth ───────────────────────────────────────────
    P(f"\n[1/5] Parsing XLSX ground truth...")
    gt_entries, no_impact_lnis = parse_ground_truth(XLSX_PATH)
    alert_gt = build_alert_ground_truth(gt_entries)
    echo_to_gt_titles = {}
    for a in alert_gt.values():
        for eid, title in a["expected_docs"].items():
            echo_to_gt_titles[eid] = title
    total_gt_docs = sum(len(a["expected_docs"]) for a in alert_gt.values())
    total_gt_secs = sum(sum(len(v) for v in a["expected_sections"].values()) for a in alert_gt.values())
    P(f"  {len(alert_gt)} alerts | {total_gt_docs} doc mappings | {total_gt_secs} section mappings")

    # ── 2. Parse PG docs ──────────────────────────────────────────
    P(f"\n[2/5] Parsing PG docs from {PG_DOCS_DIR.name}...")
    xml_files = sorted(PG_DOCS_DIR.rglob("*.xml"))
    docs, echo_to_doc, errs = [], {}, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(parse_pg_xml, f): f for f in xml_files}
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            if r: docs.append(r); echo_to_doc[r.echo_id] = r
            else: errs += 1
            if (i+1) % 300 == 0: P(f"  {i+1}/{len(xml_files)}")
    gt_eids = set(echo_to_gt_titles); avail_eids = set(echo_to_doc)
    cov = gt_eids & avail_eids
    P(f"  {len(docs)} docs parsed ({errs} errors) | GT coverage: {len(cov)}/{len(gt_eids)}")

    # ── 3. Index (embeddings + BM25) ──────────────────────────────
    P(f"\n[3/5] Building indexes...")
    model = index_pg_docs(docs, TEST_QDRANT_DIR, batch_size=64)
    bm25_idx = BM25Index(docs)
    P(f"  BM25 index: {len(docs)} docs")
    t_index = time.time()
    P(f"  Indexing done in {t_index - t0:.1f}s")

    # ── 4. Retrieve + guardrail + sections ────────────────────────
    alerts_to_run = list(alert_gt.items())
    if args.max_alerts > 0: alerts_to_run = alerts_to_run[:args.max_alerts]
    P(f"\n[4/5] Running retrieval for {len(alerts_to_run)} alerts...")

    results: list[AlertResult] = []
    total_groq = 0

    for i, (lni, gt) in enumerate(alerts_to_run):
        P(f"\n  [{i+1}/{len(alerts_to_run)}] {lni}")
        P(f"    {gt['title'][:65]}")
        P(f"    Terms: {gt['key_terms'][:65]}")
        P(f"    Expected: {len(gt['expected_docs'])} docs")

        hits = hybrid_retrieve(model, TEST_QDRANT_DIR, bm25_idx,
                               gt["title"], gt["key_terms"],
                               top_k=args.top_k * 2, threshold=args.threshold,
                               alpha=args.alpha, echo_to_doc=echo_to_doc)
        P(f"    Initial candidates: {len(hits)}")

        if not args.skip_rerank and hits:
            alert_text = f"{gt['title']} {gt['key_terms']}"
            hits = cross_encoder_rerank(alert_text, hits, echo_to_doc, top_k=args.top_k)
            P(f"    After reranking: {len(hits)}")

        if not args.skip_guardrail and hits:
            pre = len(hits)
            hits = guardrail_filter(gt["title"], gt["key_terms"], hits, echo_to_doc,
                                    workers=min(6, len(hits)))
            total_groq += pre
            P(f"    After guardrail: {pre} → {len(hits)} kept")

        ar = AlertResult(alert_lni=lni, alert_title=gt["title"], key_terms=gt["key_terms"],
                         expected_doc_ids=set(gt["expected_docs"].keys()),
                         expected_sections=gt["expected_sections"])
        ar.predicted_doc_ids = {h["echo_id"] for h in hits}
        ar.retrieval_scores = {h["echo_id"]: h["score"] for h in hits}
        ar.guardrail_kept = set(ar.predicted_doc_ids)
        ar.doc_tp = len(ar.predicted_doc_ids & ar.expected_doc_ids)
        ar.doc_fp = len(ar.predicted_doc_ids - ar.expected_doc_ids)
        ar.doc_fn = len(ar.expected_doc_ids - ar.predicted_doc_ids)

        P(f"    Retrieved {len(hits)} docs:")
        for h in hits:
            P(f"      doc_id: {h['echo_id']:<10} | score: {h['score']:.3f} | {h['title'][:55]}")

        if not args.skip_sections:
            for h in hits:
                doc = echo_to_doc.get(h["echo_id"])
                if not doc or not doc.sections: continue
                matched = match_sections_sequential(gt["title"], gt["key_terms"],
                                                   doc.title, doc.sections)
                total_groq += min(len(doc.sections), 20)
                if matched:
                    ar.predicted_sections[h["echo_id"]] = [m["heading"] for m in matched]

        results.append(ar)

    t_ret = time.time()
    t_end = time.time()

    # ── Build clean JSON report (SME-friendly) ────────────────────
    echo_info = {d.echo_id: {"doc_id": d.doc_id, "title": d.title} for d in docs}

    report = {
        "generated_at": datetime.now().isoformat(),
        "config": {
            "total_pg_docs_indexed": len(docs),
            "total_alerts_tested": len(results),
            "top_k": args.top_k,
            "similarity_threshold": args.threshold,
        },
        "timing": {
            "total_seconds": round(t_end - t0, 1),
            "indexing_seconds": round(t_index - t0, 1),
            "retrieval_seconds": round(t_ret - t_index, 1),
        },
        "results": [],
    }

    for r in results:
        retrieved = []
        for eid in sorted(r.retrieval_scores, key=lambda x: -r.retrieval_scores[x]):
            inf = echo_info.get(eid, {})
            retrieved.append({
                "doc_id": inf.get("doc_id", eid),
                "title": inf.get("title", ""),
                "relevance_score": round(r.retrieval_scores[eid], 4),
            })

        expected = []
        for eid in sorted(r.expected_doc_ids):
            inf = echo_info.get(eid, {})
            expected.append({
                "doc_id": inf.get("doc_id", eid),
                "title": echo_to_gt_titles.get(eid, inf.get("title", "")),
            })

        report["results"].append({
            "alert_lni": r.alert_lni,
            "alert_title": r.alert_title,
            "key_terms": r.key_terms,
            "expected_pg_docs": expected,
            "retrieved_pg_docs": retrieved,
        })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rp = REPORT_DIR / f"test_report_{ts}.json"
    with open(rp, "w") as f: json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Print clean summary ───────────────────────────────────────
    P("\n" + "=" * 70)
    P("  RETRIEVAL TEST REPORT")
    P("=" * 70)
    P(f"  PG Documents Indexed : {len(docs)}")
    P(f"  Alerts Tested        : {len(results)}")
    P(f"  Time                 : {t_end - t0:.1f}s")
    P("")

    for r in results:
        P(f"  Alert: {r.alert_lni}")
        P(f"    Title : {r.alert_title[:70]}")
        P(f"    Terms : {r.key_terms[:70]}")
        P(f"    Expected PG docs ({len(r.expected_doc_ids)}):")
        for eid in sorted(r.expected_doc_ids):
            inf = echo_info.get(eid, {})
            P(f"      - doc_id: {inf.get('doc_id', eid):<12} | {echo_to_gt_titles.get(eid, '')[:55]}")
        P(f"    Retrieved PG docs ({len(r.retrieval_scores)}):")
        for eid in sorted(r.retrieval_scores, key=lambda x: -r.retrieval_scores[x]):
            inf = echo_info.get(eid, {})
            P(f"      - doc_id: {inf.get('doc_id', eid):<12} | score: {r.retrieval_scores[eid]:.3f} | {inf.get('title','')[:45]}")
        P("")

    P(f"  Report saved: {rp}")
    P("=" * 70)

if __name__ == "__main__":
    main()
