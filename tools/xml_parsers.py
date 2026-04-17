"""
tools/xml_parsers.py -- XML Parsers for Alert, Court Cases, and PG Documents
=============================================================================
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

NS_LNMETA = "http://www.lexisnexis.com/xmlschemas/content/shared/lexisnexis-metadata/1/"
NS_JURISINFO = "http://www.lexisnexis.com/xmlschemas/content/legal/jurisdiction-info/1/"
NS_COURTCASE = "http://www.lexisnexis.com/xmlschemas/content/legal/courtcase/1/"
NS_PG_PREC = "http://www.lexisnexis.com/namespace/uk/precedent"
NS_PG_KH = "http://www.lexisnexis.com/namespace/uk/kh"


def local_name(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def elem_text_recursive(elem) -> str:
    parts: list[str] = []
    if elem.text and elem.text.strip():
        parts.append(elem.text.strip())
    for child in elem:
        t = elem_text_recursive(child)
        if t:
            parts.append(t)
        if child.tail and child.tail.strip():
            parts.append(child.tail.strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Document type detection
# ---------------------------------------------------------------------------
def detect_doc_type(xml_path) -> str:
    """Returns 'alert', 'court_case', 'pg_doc', or 'unknown'."""
    root = ET.parse(str(xml_path)).getroot()
    tag = root.tag
    if "casenewsalert" in tag.lower() or "casenews" in tag.lower():
        return "alert"
    if NS_COURTCASE in tag:
        return "court_case"
    if NS_PG_PREC in tag or NS_PG_KH in tag:
        return "pg_doc"
    name = Path(xml_path).name.lower()
    if "alert" in name or "casenews" in name:
        return "alert"
    return "unknown"


# ===========================================================================
# ALERT PARSER
# ===========================================================================
_ALERT_BODY_TAGS = frozenset({
    "bodytext", "body", "text", "newstext", "content",
    "summary", "abstract", "description", "headnote", "catchline",
})

_HEADING_TAGS = frozenset({
    "heading", "title", "h1", "h2", "h3",
    "alerttitle", "casetitle", "fullcasename",
})


def _extract_alert_news_summary(root) -> str:
    """Extract the body text from the alert for key holdings extraction."""
    # Try known body container tags first
    for elem in root.iter():
        if local_name(elem.tag) in _ALERT_BODY_TAGS:
            text = elem_text_recursive(elem).strip()
            if len(text) > 80:
                return text

    # Try paragraph-level extraction
    parts: list[str] = []
    for elem in root.iter():
        lname = local_name(elem.tag)
        if lname in ("p", "para", "pgrp", "paragraph"):
            text = elem_text_recursive(elem).strip()
            if len(text) > 20:
                parts.append(text)
    if parts:
        return " ".join(parts)

    # Fallback: all text from document
    full = " ".join(t.strip() for t in root.itertext() if t.strip())
    return full[:4_000] if full else ""


def _extract_key_holdings_from_xml(root) -> list[str]:
    """
    Extract key holdings directly from XML structure (no LLM needed).
    Looks for headnote, catchwords, ratio/holdings sections.
    """
    holdings: list[str] = []
    holding_tags = frozenset({
        "headnote", "catchwords", "catchphrase", "catchline",
        "holdings", "ratio", "keypoint", "summarytext",
    })
    for elem in root.iter():
        lname = local_name(elem.tag)
        if lname in holding_tags:
            text = elem_text_recursive(elem).strip()
            if text and len(text) > 20 and text not in holdings:
                holdings.append(text)
    return holdings


def _extract_practice_area(root) -> str:
    """
    Extract practice area from alert XML. Checks:
    1. <metaitem name="filterType" value="...">
    2. <classification classscheme="practiceArea"> or similar
    3. Content-based detection for Banking & Finance
    """
    practice_area = None

    # Method 1: metaitem filterType
    for elem in root.iter():
        lname = local_name(elem.tag)
        if lname == "metaitem":
            name_attr = elem.get("name", "")
            if name_attr.lower() in ("filtertype", "filter-type", "practiceareaname", "practicearea"):
                val = elem.get("value", "").strip()
                if val:
                    practice_area = val
                    break

    # Method 2: classification elements
    if not practice_area:
        for elem in root.iter():
            lname = local_name(elem.tag)
            if lname == "classification":
                scheme = (elem.get("classscheme") or "").lower()
                if "practice" in scheme or "area" in scheme or "subject" in scheme:
                    for child in elem.iter():
                        if local_name(child.tag) in ("classname", "classitem-identifier", "classcode"):
                            text = (child.text or "").strip()
                            if text:
                                practice_area = text
                                break
                    if practice_area:
                        break

    # Method 3: content-based detection for Banking & Finance
    if not practice_area:
        full_text = " ".join(root.itertext()).lower()
        bf_signals = [
            "banking and finance", "banking & finance", "banking and financial",
            "financial services", "loan agreement", "guarantee",
            "indemnity", "banking", "lender", "borrower", "creditor",
            "security interest", "debenture", "facility agreement",
        ]
        matches = sum(1 for s in bf_signals if s in full_text)
        if matches >= 2:
            practice_area = "Banking and Financial"

    return practice_area or ""


def parse_alert(xml_path) -> dict:
    """Parse a CaseNewsAlert XML. Returns alert metadata + news_summary."""
    xml_path = Path(xml_path)
    root = ET.parse(str(xml_path)).getroot()

    lni_id = None
    cite_defs: list[str] = []
    cite_refs: list[str] = []
    date_of_decision = None
    court_name = None
    jurisdiction = None

    for elem in root.iter():
        lname = local_name(elem.tag)

        # LNI identifier
        if lname == "identifier":
            for attr, val in elem.attrib.items():
                if local_name(attr) == "identifier-scheme" and val == "LNI":
                    if elem.text:
                        lni_id = elem.text.strip()

        # cite_def = normcite attribute on spans that have CITE-DEF
        # The CITE-DEF attr marks it as a citation definition;
        # the normcite attr on the SAME span is the actual citation string.
        if lname == "span":
            attrs = elem.attrib
            has_cite_def = False
            normcite_val = None

            for attr, val in attrs.items():
                aname = local_name(attr)
                if aname == "CITE-DEF":
                    has_cite_def = True
                if aname == "normcite" and val:
                    normcite_val = val.strip()

            if has_cite_def and normcite_val:
                if normcite_val not in cite_defs:
                    cite_defs.append(normcite_val)
            elif normcite_val:
                if normcite_val not in cite_refs:
                    cite_refs.append(normcite_val)

        if lname == "decisiondate":
            nd = elem.get("normdate")
            if nd:
                date_of_decision = nd

        if lname == "courtname" and elem.text:
            court_name = elem.text.strip()

    # Jurisdiction
    system_tag = f"{{{NS_JURISINFO}}}system"
    for elem in root.iter(system_tag):
        if elem.text and elem.text.strip():
            jurisdiction = elem.text.strip()
            break

    if not jurisdiction:
        classnames: list[str] = []
        for elem in root.iter():
            if local_name(elem.tag) == "classification":
                if elem.get("classscheme") == "jurisdictionAffected":
                    for child in elem.iter():
                        if local_name(child.tag) == "classname" and child.text:
                            j = child.text.strip()
                            if j and j not in classnames:
                                classnames.append(j)
        if classnames:
            jurisdiction = ", ".join(classnames)

    # Practice area (dedicated extraction)
    practice_area = _extract_practice_area(root)

    # News summary
    news_summary = _extract_alert_news_summary(root)

    # Key holdings directly from XML structure
    xml_holdings = _extract_key_holdings_from_xml(root)

    return {
        "lni_id": lni_id,
        "cite_defs": cite_defs,
        "cite_refs": cite_refs,
        "date_of_decision": date_of_decision,
        "court_name": court_name,
        "jurisdiction": jurisdiction,
        "practice_area": practice_area,
        "news_summary": news_summary,
        "xml_holdings": xml_holdings,
        "source_file": str(xml_path),
    }


# ===========================================================================
# COURT CASE PARSER
# ===========================================================================
def _cc_find_first(root, *tags) -> str:
    for elem in root.iter():
        if local_name(elem.tag) in tags:
            if elem.text and elem.text.strip():
                return elem.text.strip()
    return ""


def parse_courtcase(xml_path) -> dict:
    """Parse a court-case XML. Returns metadata + paragraphs."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    lni_id = ""
    for elem in root.iter():
        scheme = (
            elem.attrib.get(f"{{{NS_LNMETA}}}identifier-scheme", "")
            or elem.attrib.get("identifier-scheme", "")
        )
        if scheme == "LNI" and elem.text and elem.text.strip():
            lni_id = elem.text.strip()
            break

    # cite_ref: the normcite attribute from spans in the case document
    cite_ref = ""
    for elem in root.iter():
        lname = local_name(elem.tag)
        if lname == "span":
            nc = elem.attrib.get("normcite", "").strip()
            if nc:
                cite_ref = nc
                break

    # Fallback: nonciteidentifier element
    if not cite_ref:
        for elem in root.iter():
            if local_name(elem.tag) == "nonciteidentifier" and elem.text and elem.text.strip():
                cite_ref = elem.text.strip()
                break

    case_title = _cc_find_first(root, "fullcasename")
    date_of_decision = _cc_find_first(root, "decisiondate") or _cc_find_first(
        root, "date-text"
    )
    court = _cc_find_first(root, "courtname")
    system = _cc_find_first(root, "system")
    jurisdiction = (
        f"{court} | {system}" if court and system else court or system
    )

    paragraphs: list[str] = []
    for elem in root.iter():
        if local_name(elem.tag) == "bodytext":
            for child in elem:
                if local_name(child.tag) in ("pgrp", "list", "heading", "note"):
                    text = elem_text_recursive(child).strip()
                    if text and len(text) > 10:
                        paragraphs.append(text)

    text_lines = [
        f"CASE TITLE: {case_title}",
        f"CITATION: {cite_ref}",
        f"JURISDICTION: {jurisdiction}",
        f"DATE OF DECISION: {date_of_decision}",
        f"LNI ID: {lni_id}",
        "",
        "=" * 60,
        "",
    ]
    for para in paragraphs:
        text_lines.append(para)
        text_lines.append("")

    return {
        "lni_id": lni_id,
        "cite_ref": cite_ref,
        "case_title": case_title,
        "date_of_decision": date_of_decision,
        "jurisdiction": jurisdiction,
        "paragraphs": paragraphs,
        "text_lines": text_lines,
    }


# ===========================================================================
# PG DOC PARSER
# ===========================================================================
def _extract_pg_doc_id(root, xml_path: Path) -> str:
    """
    Extract a stable doc_id from PG XML content, falling back to filename.

    Priority:
    1. <identifier identifier-scheme="DOC-ID"> or scheme="LNI"
    2. <docid> or <document-id> element text
    3. First normcite on a <cite> element
    4. Filename stem (normalized: strip em-dashes, special chars -> underscore)
    """
    ns_meta = NS_LNMETA

    for elem in root.iter():
        lname = local_name(elem.tag)
        if lname == "identifier":
            scheme = (
                elem.attrib.get(f"{{{ns_meta}}}identifier-scheme", "")
                or elem.attrib.get("identifier-scheme", "")
            ).upper()
            if scheme in ("DOC-ID", "LNI", "PGDOC-ID") and elem.text and elem.text.strip():
                return elem.text.strip()

    for elem in root.iter():
        lname = local_name(elem.tag)
        if lname in ("docid", "document-id", "documentid"):
            if elem.text and elem.text.strip():
                return elem.text.strip()

    for elem in root.iter():
        if local_name(elem.tag) == "cite":
            nc = elem.attrib.get("normcite", "").strip()
            if nc:
                return nc

    stem = xml_path.stem.split("\u2014")[0].strip()
    normalized = re.sub(r"[^\w\-.]", "_", stem).strip("_")
    return normalized if normalized else xml_path.stem


def parse_pgdoc(xml_path) -> dict:
    """Parse a PG/precedent XML. Returns metadata + paragraphs + text_lines."""
    xml_path = Path(xml_path)
    root = ET.parse(str(xml_path)).getroot()

    doc_id = _extract_pg_doc_id(root, xml_path)

    doc_title = ""
    for elem in root.iter():
        if local_name(elem.tag) == "front":
            for child in elem:
                if local_name(child.tag) == "title":
                    for sub in child:
                        if local_name(sub.tag) == "text" and sub.text and sub.text.strip():
                            doc_title = sub.text.strip()
                            break
            break
    if not doc_title:
        for elem in root.iter():
            if local_name(elem.tag) == "document-title" and elem.text and elem.text.strip():
                doc_title = elem.text.strip()
                break

    seen: set[str] = set()
    cite_ids: list[str] = []
    for elem in root.iter():
        if local_name(elem.tag) == "cite":
            for key in ("normcite", "citeref"):
                val = elem.attrib.get(key, "").strip()
                if val and val not in seen:
                    seen.add(val)
                    cite_ids.append(val)

    full_text = " ".join(root.itertext())
    jurisdiction = ""
    for jur in ["England and Wales", "England & Wales", "Scotland", "Northern Ireland"]:
        if jur.lower() in full_text.lower():
            jurisdiction = jur
            break

    title_lower = doc_title.lower()
    areas: list[str] = []
    mapping = {
        "guarantee": "Banking & Finance",
        "indemnity": "Banking & Finance",
        "loan": "Banking & Finance",
        "finance": "Banking & Finance",
        "banking": "Banking & Finance",
        "lender": "Banking & Finance",
        "borrower": "Banking & Finance",
        "facility": "Banking & Finance",
        "security": "Banking & Finance",
        "debenture": "Banking & Finance",
        "employment": "Employment",
        "property": "Property",
        "corporate": "Corporate",
        "insolvency": "Restructuring & Insolvency",
        "tax": "Tax",
        "dispute": "Dispute Resolution",
        "litigation": "Dispute Resolution",
        "arbitration": "Dispute Resolution",
    }
    for kw, area in mapping.items():
        if kw in title_lower and area not in areas:
            areas.append(area)
    practice_area = ", ".join(areas)

    candidates: list[str] = []
    for elem in root.iter():
        if local_name(elem.tag) in ("clause", "inclusion", "para"):
            text = elem_text_recursive(elem).strip()
            if text and len(text) > 30:
                candidates.append(text)

    seen_keys: set[str] = set()
    paragraphs: list[str] = []
    for p in candidates:
        key = p[:120]
        if key not in seen_keys:
            seen_keys.add(key)
            paragraphs.append(p)

    text_lines = [
        f"DOCUMENT TITLE: {doc_title}",
        f"DOC ID: {doc_id}",
        f"JURISDICTION: {jurisdiction}",
        f"PRACTICE AREA: {practice_area}",
        f"CITE IDS: {' | '.join(cite_ids[:10])}",
        "",
        "=" * 60,
        "",
    ]
    for para in paragraphs:
        text_lines.append(para)
        text_lines.append("")

    return {
        "doc_id": doc_id,
        "cite_ids": cite_ids,
        "doc_title": doc_title,
        "jurisdiction": jurisdiction,
        "practice_area": practice_area,
        "paragraphs": paragraphs,
        "text_lines": text_lines,
    }


# ===========================================================================
# PG DOC SECTION PARSER (for Matching Agent)
# ===========================================================================
def parse_pgdoc_sections(xml_path) -> list[dict]:
    """
    Parse a PG XML into a list of sections/subsections for the Matching Agent.
    Each section dict contains: section_id, heading, text, subsections[].
    Falls back to paragraph-level splitting if no structural sections found.
    """
    xml_path = Path(xml_path)
    root = ET.parse(str(xml_path)).getroot()

    sections: list[dict] = []
    section_idx = 0

    for elem in root.iter():
        lname = local_name(elem.tag)
        if lname in ("clause", "inclusion", "section"):
            heading = ""
            for child in elem:
                if local_name(child.tag) in ("heading", "title", "text"):
                    h = elem_text_recursive(child).strip()
                    if h and len(h) < 200:
                        heading = h
                        break

            text = elem_text_recursive(elem).strip()
            if not text or len(text) < 30:
                continue

            subsections: list[dict] = []
            sub_idx = 0
            for child in elem:
                child_lname = local_name(child.tag)
                if child_lname in ("clause", "inclusion", "para", "section"):
                    sub_text = elem_text_recursive(child).strip()
                    if sub_text and len(sub_text) > 20:
                        sub_heading = ""
                        for sub_child in child:
                            if local_name(sub_child.tag) in ("heading", "title"):
                                sh = elem_text_recursive(sub_child).strip()
                                if sh and len(sh) < 200:
                                    sub_heading = sh
                                    break
                        subsections.append({
                            "subsection_id": f"sec_{section_idx}_sub_{sub_idx}",
                            "heading": sub_heading,
                            "text": sub_text,
                        })
                        sub_idx += 1

            sections.append({
                "section_id": f"sec_{section_idx}",
                "heading": heading,
                "text": text,
                "subsections": subsections,
            })
            section_idx += 1

    if not sections:
        for elem in root.iter():
            if local_name(elem.tag) in ("para", "pgrp", "p"):
                text = elem_text_recursive(elem).strip()
                if text and len(text) > 30:
                    sections.append({
                        "section_id": f"para_{section_idx}",
                        "heading": "",
                        "text": text,
                        "subsections": [],
                    })
                    section_idx += 1

    return sections
