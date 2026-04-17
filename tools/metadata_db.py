"""
tools/metadata_db.py -- SQLite Metadata Helpers
================================================
Read-only access to pre-indexed court case and PG document metadata.
Write operations are handled by ingest.py.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ===========================================================================
# Court Cases (court_cases.db)
# ===========================================================================
def _get_cc_db_path() -> Path:
    return Path(os.getenv("CC_DB_PATH", str(_DATA_DIR / "court_cases.db")))


def get_case_by_lni(lni_id: str) -> dict:
    """Retrieve full court case metadata by LNI ID."""
    db_path = _get_cc_db_path()
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM court_cases WHERE lni_id = ?", (lni_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else {}
    except Exception:
        return {}


def get_case_by_cite_ref(cite_ref: str) -> dict:
    """Retrieve court case metadata by cite_ref (normcite). Case-insensitive match."""
    if not cite_ref:
        return {}
    db_path = _get_cc_db_path()
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM court_cases WHERE LOWER(cite_ref) = LOWER(?)", (cite_ref.strip(),)
        ).fetchone()
        conn.close()
        return dict(row) if row else {}
    except Exception:
        return {}


def find_case_by_cite_refs(cite_refs: list[str]) -> dict:
    """
    Try to find a case matching ANY of the provided cite_refs (normcites).
    Tries each cite_ref in order and returns the first match.
    """
    for ref in cite_refs:
        result = get_case_by_cite_ref(ref)
        if result:
            return result
    return {}


def get_case_text_by_lni(lni_id: str) -> str:
    """Retrieve the full text of a court case by LNI ID."""
    case = get_case_by_lni(lni_id)
    if case and case.get("full_text"):
        return case["full_text"]
    txt_dir = Path(os.getenv("PARSED_TXT_DIR", str(_DATA_DIR / "texts" / "court_cases")))
    txt_path = txt_dir / f"{lni_id}.txt"
    if txt_path.is_file():
        return txt_path.read_text(encoding="utf-8", errors="ignore")
    return ""


def get_case_text(case: dict) -> str:
    """Retrieve full text from a case dict (from full_text column or .txt fallback)."""
    if case.get("full_text"):
        return case["full_text"]
    lni_id = case.get("lni_id", "")
    if lni_id:
        return get_case_text_by_lni(lni_id)
    return ""


def get_case_source_file(lni_id: str) -> str:
    """Look up source_file path for a court case by LNI ID."""
    case = get_case_by_lni(lni_id)
    return case.get("source_file", "")


def list_all_cases() -> list[dict]:
    """Return metadata for all pre-indexed court cases."""
    db_path = _get_cc_db_path()
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT lni_id, cite_ref, case_title, date_of_decision, jurisdiction FROM court_cases ORDER BY lni_id").fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ===========================================================================
# PG Documents (pg_docs.db)
# ===========================================================================
def _get_pg_db_path() -> Path:
    return Path(os.getenv("PG_DB_PATH", str(_DATA_DIR / "pg_docs.db")))


def get_pg_source_file(doc_id: str) -> str:
    """Look up source_file path for a PG document by doc_id."""
    db_path = _get_pg_db_path()
    if not db_path.exists():
        return ""
    try:
        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT source_file FROM pg_docs WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        conn.close()
        return row[0] if row else ""
    except Exception:
        return ""


def get_pg_metadata(doc_id: str) -> dict:
    """Retrieve full PG document metadata by doc_id."""
    db_path = _get_pg_db_path()
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM pg_docs WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else {}
    except Exception:
        return {}


def list_all_pg_docs() -> list[dict]:
    """Return metadata for all pre-indexed PG documents."""
    db_path = _get_pg_db_path()
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM pg_docs ORDER BY doc_id").fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []
