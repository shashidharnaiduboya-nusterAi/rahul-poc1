"""
ingest.py -- Document Pre-Indexing Pipeline (Cases + PG Docs)
==============================================================
Pre-indexes BOTH court case documents AND Practical Guidance (PG) documents.
NO paragraph-level chunking at pre-index time.

For COURT CASES:
    1. Parse case XML
    2. Write .txt file
    3. Save metadata + full text to court_cases.db (SQLite)
    4. Embed whole document -> Qdrant 'case_doc_index' (one point per case)

For PG DOCUMENTS:
    1. Parse PG XML
    2. Write .txt file
    3. Save metadata to pg_docs.db (SQLite)
    4. Optionally generate retrieval profile (LLM)
    5. Embed whole document -> Qdrant 'pg_doc_index' (one point per PG doc)

Chunking is ONLY done at runtime by the CaseProcessingAgent when an alert arrives.

Qdrant collections (doc-level only):
    case_doc_index -- one point per case document
    pg_doc_index   -- one point per PG document
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FilterSelector,
    FieldCondition,
    MatchValue,
)

from tools.xml_parsers import parse_courtcase, parse_pgdoc, detect_doc_type

load_dotenv()

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CC_DB_PATH = DATA_DIR / "court_cases.db"
PG_DB_PATH = DATA_DIR / "pg_docs.db"
QDRANT_DIR = Path(os.getenv("QDRANT_PATH", str(DATA_DIR / "qdrant")))

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBED_DIM = 1024

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
_USE_AZURE = bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)

OPENAI_STRONG_MODEL = os.getenv(
    "AZURE_STRONG_MODEL" if _USE_AZURE else "OPENAI_STRONG_MODEL",
    "gpt-4o",
)

_DOC_SUMMARY_SYSTEM = """\
You are a legal information architect. Given a Practical Guidance document,
produce a compact RETRIEVAL PROFILE optimised for semantic search.

Output ONLY these labelled fields (one per line):
[TITLE]          document title
[DOC_TYPE]       type of document (Practice Note / Precedent / Checklist / Overview)
[PRACTICE_AREA]  primary legal practice area(s)
[JURISDICTION]   jurisdiction
[KEY_TOPICS]     comma-separated list of main legal topics covered
[LEGAL_CONCEPTS] comma-separated list of legal concepts, principles, and doctrines
[ENTITIES]       comma-separated list of relevant parties, instruments, or bodies
[KEYWORDS]       30-50 comma-separated keywords a lawyer would use to find this document
[CANONICAL_PHRASES] 5-10 exact legal phrases distinctive to this document

No headings, no explanations -- labelled fields only."""

# ---------------------------------------------------------------------------
# OpenAI client (lazy singleton)
# ---------------------------------------------------------------------------
_openai_client = None


def _get_openai():
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    if _USE_AZURE:
        try:
            from openai import AzureOpenAI
            _openai_client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            )
            print("  [OpenAI] Using Azure OpenAI endpoint")
            return _openai_client
        except Exception as exc:
            print(f"  [OpenAI] Could not initialise Azure client: {exc}")
            return None

    if not OPENAI_API_KEY:
        return None
    try:
        from openai import OpenAI
        kwargs: dict = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        _openai_client = OpenAI(**kwargs)
        return _openai_client
    except Exception as exc:
        print(f"  [OpenAI] Could not initialise client: {exc}")
        return None


# ---------------------------------------------------------------------------
# Qdrant helpers (single client instance per run)
# ---------------------------------------------------------------------------
_qdrant_client: Optional[QdrantClient] = None


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    _qdrant_client = QdrantClient(path=str(QDRANT_DIR))
    return _qdrant_client


def _ensure_collection(qc: QdrantClient, name: str) -> None:
    existing = [c.name for c in qc.get_collections().collections]
    if name not in existing:
        qc.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def _point_exists(qc: QdrantClient, collection: str, key: str, value: str) -> bool:
    hits, _ = qc.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must=[FieldCondition(key=key, match=MatchValue(value=value))]
        ),
        limit=1,
    )
    return len(hits) > 0


def _delete_points(qc: QdrantClient, collection: str, key: str, value: str) -> None:
    qc.delete(
        collection_name=collection,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key=key, match=MatchValue(value=value))]
            )
        ),
    )


# ---------------------------------------------------------------------------
# AI doc-level retrieval profile (OpenAI strong model, PG only)
# ---------------------------------------------------------------------------
def generate_pg_doc_summary(doc_title: str, paragraphs: list[str]) -> str:
    client = _get_openai()
    if not client:
        return ""

    content = "\n\n".join(paragraphs)[:15_000]
    user_msg = f"DOCUMENT TITLE: {doc_title}\n\nDOCUMENT CONTENT:\n{content}"

    try:
        resp = client.chat.completions.create(
            model=OPENAI_STRONG_MODEL,
            messages=[
                {"role": "system", "content": _DOC_SUMMARY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"  [OpenAI] Doc summary error: {exc}")
        return ""


# ===========================================================================
# SQLite metadata
# ===========================================================================
def _init_cc_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS court_cases (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            lni_id           TEXT    UNIQUE NOT NULL,
            cite_ref         TEXT,
            case_title       TEXT,
            date_of_decision TEXT,
            jurisdiction     TEXT,
            source_file      TEXT,
            full_text        TEXT,
            timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def _init_pg_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pg_docs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id        TEXT    UNIQUE NOT NULL,
            cite_ids      TEXT,
            doc_title     TEXT,
            jurisdiction  TEXT,
            practice_area TEXT,
            source_file   TEXT,
            timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def _prompt_on_duplicate(primary_id: str, id_label: str) -> str:
    print(f"\n  [!] DUPLICATE -- {id_label} '{primary_id}' already exists.")
    while True:
        choice = input("      [s]kip / [r]eplace: ").strip().lower()
        if choice in ("s", "skip"):
            return "skip"
        if choice in ("r", "replace"):
            return "replace"
        print("      Invalid -- enter s or r.")


def save_cc_metadata(doc: dict, source_file: str, on_conflict: str = "prompt") -> bool:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    primary_id = doc["lni_id"]
    conn = sqlite3.connect(str(CC_DB_PATH))
    _init_cc_db(conn)

    full_text = "\n".join(doc.get("text_lines", []))

    exists = conn.execute(
        "SELECT 1 FROM court_cases WHERE lni_id = ?", (primary_id,)
    ).fetchone() is not None

    if exists:
        choice = (
            on_conflict if on_conflict != "prompt"
            else _prompt_on_duplicate(primary_id, "LNI ID")
        )
        if choice == "skip":
            print("  [Metadata] Skipped -- existing record kept.")
            conn.close()
            return False
        conn.execute(
            "UPDATE court_cases SET cite_ref=?, case_title=?, date_of_decision=?, "
            "jurisdiction=?, source_file=?, full_text=?, timestamp=CURRENT_TIMESTAMP "
            "WHERE lni_id=?",
            (
                doc["cite_ref"], doc["case_title"], doc["date_of_decision"],
                doc["jurisdiction"], source_file, full_text, doc["lni_id"],
            ),
        )
        conn.commit()
        conn.close()
        print(f"  [Metadata] Updated -- LNI: {primary_id}")
        return True

    conn.execute(
        "INSERT INTO court_cases "
        "(lni_id, cite_ref, case_title, date_of_decision, jurisdiction, source_file, full_text) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            doc["lni_id"], doc["cite_ref"], doc["case_title"],
            doc["date_of_decision"], doc["jurisdiction"], source_file, full_text,
        ),
    )
    conn.commit()
    conn.close()
    print(f"  [Metadata] Saved to court_cases.db -- LNI: {primary_id}")
    return True


def save_pg_metadata(doc: dict, source_file: str, on_conflict: str = "prompt") -> bool:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    primary_id = doc["doc_id"]
    conn = sqlite3.connect(str(PG_DB_PATH))
    _init_pg_db(conn)

    exists = conn.execute(
        "SELECT 1 FROM pg_docs WHERE doc_id = ?", (primary_id,)
    ).fetchone() is not None

    if exists:
        choice = (
            on_conflict if on_conflict != "prompt"
            else _prompt_on_duplicate(primary_id, "Doc ID")
        )
        if choice == "skip":
            print("  [Metadata] Skipped -- existing record kept.")
            conn.close()
            return False
        conn.execute(
            "UPDATE pg_docs SET cite_ids=?, doc_title=?, jurisdiction=?, "
            "practice_area=?, source_file=?, timestamp=CURRENT_TIMESTAMP "
            "WHERE doc_id=?",
            (
                json.dumps(doc["cite_ids"]), doc["doc_title"], doc["jurisdiction"],
                doc["practice_area"], source_file, doc["doc_id"],
            ),
        )
        conn.commit()
        conn.close()
        print(f"  [Metadata] Updated -- Doc ID: {primary_id}")
        return True

    conn.execute(
        "INSERT INTO pg_docs "
        "(doc_id, cite_ids, doc_title, jurisdiction, practice_area, source_file) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            doc["doc_id"], json.dumps(doc["cite_ids"]), doc["doc_title"],
            doc["jurisdiction"], doc["practice_area"], source_file,
        ),
    )
    conn.commit()
    conn.close()
    print(f"  [Metadata] Saved to pg_docs.db -- Doc ID: {primary_id}")
    return True


# ===========================================================================
# Qdrant: document-level embeddings (NO chunking)
# ===========================================================================
def save_doc_embedding(
    doc: dict, doc_type: str, model: SentenceTransformer, doc_summary: str = ""
) -> None:
    """
    Embed the whole document as a SINGLE vector and store in Qdrant.
    - Court cases -> 'case_doc_index'
    - PG docs -> 'pg_doc_index'
    """
    if doc_type == "court_case":
        collection = "case_doc_index"
        doc_id_key = "lni_id"
        doc_id_val = doc["lni_id"]

        # For cases: embed using title + full content (no LLM summary at ingest)
        title = doc.get("case_title", "")
        cite = doc.get("cite_ref", "")
        jurisdiction = doc.get("jurisdiction", "")
        paragraphs = doc.get("paragraphs", [])
        embed_text = f"{title} {cite} {jurisdiction} " + " ".join(paragraphs)
        embed_text = embed_text[:16_000]

        payload = {
            "lni_id": doc["lni_id"],
            "case_title": title,
            "cite_ref": cite,
            "date_of_decision": doc.get("date_of_decision", ""),
            "jurisdiction": jurisdiction,
        }

    else:  # pg_doc
        collection = "pg_doc_index"
        doc_id_key = "doc_id"
        doc_id_val = doc["doc_id"]

        # For PG: use LLM retrieval profile if available, else title + content
        if doc_summary:
            embed_text = doc_summary[:16_000]
        else:
            title = doc.get("doc_title", "")
            practice_area = doc.get("practice_area", "")
            jurisdiction = doc.get("jurisdiction", "")
            paragraphs = doc.get("paragraphs", [])
            embed_text = f"{title} {practice_area} {jurisdiction} " + " ".join(paragraphs)
            embed_text = embed_text[:16_000]

        payload = {
            "doc_id": doc["doc_id"],
            "doc_title": doc.get("doc_title", ""),
            "cite_ids": doc.get("cite_ids", []),
            "jurisdiction": doc.get("jurisdiction", ""),
            "practice_area": doc.get("practice_area", ""),
            "doc_summary": doc_summary if doc_summary else "",
        }

    qc = _get_qdrant()
    _ensure_collection(qc, collection)

    if _point_exists(qc, collection, doc_id_key, doc_id_val):
        _delete_points(qc, collection, doc_id_key, doc_id_val)

    vector = model.encode(embed_text).tolist()
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload=payload,
    )
    qc.upsert(collection_name=collection, points=[point])
    print(f"  [DocIndex] Document embedding -> Qdrant '{collection}'")


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================
def process(
    xml_path: Path,
    cc_output_dir: Path,
    pg_output_dir: Path,
    on_conflict: str = "prompt",
    skip_ai: bool = False,
) -> None:
    xml_path = Path(xml_path)
    print(f"\n{'=' * 60}")
    print(f"File: {xml_path.name}")
    print("=" * 60)

    doc_type = detect_doc_type(xml_path)
    if doc_type == "alert":
        print("  Type: ALERT -- alerts are processed at runtime via run.py.")
        return
    if doc_type == "unknown":
        print("  ERROR: Unknown XML type -- skipping.")
        return

    print(f"  Type: {doc_type.upper().replace('_', ' ')}")

    hf_token = os.getenv("HF_TOKEN", None)
    model = SentenceTransformer(
        EMBED_MODEL_NAME, trust_remote_code=True, token=hf_token,
    )

    if doc_type == "court_case":
        _process_court_case(xml_path, cc_output_dir, on_conflict, model)
    else:
        _process_pg_doc(xml_path, pg_output_dir, on_conflict, skip_ai, model)


def _process_court_case(
    xml_path: Path, output_dir: Path, on_conflict: str,
    model: SentenceTransformer,
) -> None:
    """Case: parse, store as-is in SQLite (full_text), embed whole doc."""
    doc = parse_courtcase(xml_path)
    primary_id = doc["lni_id"]
    print(f"  LNI ID      : {doc['lni_id']}")
    print(f"  Title       : {doc['case_title']}")
    print(f"  Citation    : {doc['cite_ref']}")
    print(f"  Date        : {doc['date_of_decision']}")
    print(f"  Jurisdiction: {doc['jurisdiction'][:65]}")
    print(f"  Paragraphs  : {len(doc['paragraphs'])}")

    if not primary_id:
        print("  ERROR: Could not determine LNI ID -- skipping.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"{primary_id}.txt"
    txt_path.write_text("\n".join(doc["text_lines"]), encoding="utf-8")
    print(f"  [Text] -> {txt_path}")

    inserted = save_cc_metadata(doc, str(xml_path.resolve()), on_conflict)
    if not inserted:
        return

    # Single doc-level embedding -- NO chunking at pre-index
    save_doc_embedding(doc, "court_case", model)

    print(f"\n  Done -- '{primary_id}'")
    print(f"  SQLite -> {CC_DB_PATH} (metadata + full text)")
    print(f"  Qdrant -> {QDRANT_DIR}/  collection: case_doc_index (1 point)")


def _process_pg_doc(
    xml_path: Path, output_dir: Path, on_conflict: str, skip_ai: bool,
    model: SentenceTransformer,
) -> None:
    """PG doc: parse, store metadata in SQLite, embed whole doc."""
    doc = parse_pgdoc(xml_path)
    primary_id = doc["doc_id"]
    print(f"  Doc ID      : {doc['doc_id']}")
    print(f"  Title       : {doc['doc_title']}")
    print(f"  Cite IDs    : {len(doc['cite_ids'])} entries")
    print(f"  Jurisdiction: {doc['jurisdiction']}")
    print(f"  Practice    : {doc['practice_area']}")
    print(f"  Paragraphs  : {len(doc['paragraphs'])}")

    if not primary_id:
        print("  ERROR: Could not determine document ID -- skipping.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"{primary_id}.txt"
    txt_path.write_text("\n".join(doc["text_lines"]), encoding="utf-8")
    print(f"  [Text] -> {txt_path}")

    inserted = save_pg_metadata(doc, str(xml_path.resolve()), on_conflict)
    if not inserted:
        return

    # Optional: LLM retrieval profile for better PG doc embedding
    doc_summary = ""
    if not skip_ai and (OPENAI_API_KEY or _USE_AZURE):
        print("  [DocSummary] Generating document retrieval profile...")
        doc_summary = generate_pg_doc_summary(doc["doc_title"], doc["paragraphs"])

    # Single doc-level embedding -- NO chunking at pre-index
    save_doc_embedding(doc, "pg_doc", model, doc_summary)

    print(f"\n  Done -- '{primary_id}'")
    print(f"  SQLite -> {PG_DB_PATH} (metadata)")
    print(f"  Qdrant -> {QDRANT_DIR}/  collection: pg_doc_index (1 point)")


def process_batch(
    batch_dir: Path,
    cc_output_dir: Path,
    pg_output_dir: Path,
    on_conflict: str = "replace",
    skip_ai: bool = False,
) -> dict:
    """Process all XML files in a directory. Returns stats dict."""
    xml_files = sorted(
        p for p in batch_dir.rglob("*.xml")
        if p.is_file() and not p.name.startswith(".")
    )

    if not xml_files:
        print(f"  No XML files found in {batch_dir}")
        return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}

    print(f"\n{'=' * 60}")
    print(f"  BATCH INGESTION: {len(xml_files)} XML files from {batch_dir}")
    print("=" * 60)

    stats = {"total": len(xml_files), "processed": 0, "skipped": 0, "errors": 0}

    for i, xml_path in enumerate(xml_files):
        print(f"\n  [{i + 1}/{len(xml_files)}] {xml_path.name}")
        try:
            process(
                xml_path=xml_path,
                cc_output_dir=cc_output_dir,
                pg_output_dir=pg_output_dir,
                on_conflict=on_conflict,
                skip_ai=skip_ai,
            )
            stats["processed"] += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            stats["errors"] += 1

    print(f"\n{'=' * 60}")
    print(f"  BATCH COMPLETE: {stats['processed']} processed, "
          f"{stats['errors']} errors, {stats['skipped']} skipped "
          f"(of {stats['total']} total)")
    print("=" * 60)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre-index court case or PG document XMLs into the RAG pipeline."
    )
    ap.add_argument("xml_file", nargs="?", help="Path to a single XML file (court case or PG doc)")
    ap.add_argument(
        "--batch-dir",
        help="Process all XML files in this directory (recursively).",
    )
    ap.add_argument(
        "--cc-output-dir",
        default=os.getenv("CC_OUTPUT_DIR", str(DATA_DIR / "texts" / "court_cases")),
        help="Folder for court-case .txt files.",
    )
    ap.add_argument(
        "--pg-output-dir",
        default=os.getenv("PG_OUTPUT_DIR", str(DATA_DIR / "texts" / "pg_docs")),
        help="Folder for PG-doc .txt files.",
    )
    ap.add_argument(
        "--on-conflict",
        choices=["skip", "replace", "prompt"],
        default="prompt",
        help="Action when a duplicate doc ID is found (default: prompt).",
    )
    ap.add_argument(
        "--no-ai-summary",
        action="store_true",
        help="Skip OpenAI doc-level summary generation for PG docs.",
    )
    args = ap.parse_args()

    if args.batch_dir:
        batch_path = Path(args.batch_dir)
        if not batch_path.is_dir():
            print(f"Error: Directory not found -- {batch_path}")
            sys.exit(1)
        conflict = args.on_conflict if args.on_conflict != "prompt" else "replace"
        process_batch(
            batch_dir=batch_path,
            cc_output_dir=Path(args.cc_output_dir),
            pg_output_dir=Path(args.pg_output_dir),
            on_conflict=conflict,
            skip_ai=args.no_ai_summary,
        )
    elif args.xml_file:
        xml_path = Path(args.xml_file)
        if not xml_path.is_file():
            print(f"Error: File not found -- {xml_path}")
            sys.exit(1)
        process(
            xml_path=xml_path,
            cc_output_dir=Path(args.cc_output_dir),
            pg_output_dir=Path(args.pg_output_dir),
            on_conflict=args.on_conflict,
            skip_ai=args.no_ai_summary,
        )
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
