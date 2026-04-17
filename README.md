# POC-1: Legal Case Impact Analysis Pipeline

Multi-agent system built with Google ADK that automatically analyses incoming case news alerts and determines their impact on stored Practical Guidance (PG) documents.

## How It Works

### Step 1: Pre-Index (one-time)

Parse **both** court case XMLs and PG document XMLs. Store metadata and **document-level** embeddings only (no paragraph-level chunking at pre-index).

```
Case XMLs  ─→ ingest.py ─→ court_cases.db (metadata + full text) + Qdrant (case_doc_index)
PG XMLs    ─→ ingest.py ─→ pg_docs.db (metadata)                + Qdrant (pg_doc_index)
```

### Step 2: Process Alert (runtime)

When an alert XML arrives:
1. Parse the alert, identify the case by LNI, extract `cite_ref` and `cite_def`
2. Retrieve the case **as-is** from stored DB (court_cases.db)
3. Chunk at paragraph level, generate per-paragraph metadata + summary, embed each chunk
4. Generate document-level detailed summary (15-20 pages) + retrieval profile, embed them

### Step 3: Two-Level Hybrid Retrieval (against pg_doc_index)

**Level 1 -- Document-level:** Case summary embedding (semantic search on `pg_doc_index`) + metadata keywords (BM25) → get ALL matching PG documents

**Level 2 -- Paragraph-level refinement:** Case paragraph embeddings (max/mean pooled, searched against `pg_doc_index`) + paragraph metadata keywords (BM25) → refine and score each PG document

**Citation boost:** `cite_ref`/`cite_def` from alert matched against PG doc `cite_ids` → bonus score for citation overlap

Output: `doc_id` + `score` for each affected PG document

### Steps 4-5: Matching + Reasoning

For each affected PG document: identify which sections are impacted and generate WHERE/WHAT/WHY change suggestions.

## Architecture

```
[1. AlertIngestionAgent]  -- reads alert file
[2. FilterAgent]          -- optional practice area filter (B&F)
[3. AlertProcessingAgent] -- parses XML, extracts metadata + key holdings + cite_ref/cite_def
[4. CaseProcessingAgent]  -- retrieves case from DB, chunks, summaries, embeddings (runtime)
[5. RetrievalAgent]       -- two-level hybrid retrieval against pg_doc_index + citation matching
[6. MatchingAgent]        -- per-PG-doc section-level impact matching
[7. ReasoningAgent]       -- WHERE/WHAT/WHY change suggestions
[8. ReportAgent]          -- final structured report for SME review
```

All agents share state via ADK session. Case-derived data at runtime is temporary (in-memory).

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

Edit `.env` and set:

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | API key for OpenAI endpoint |
| `OPENAI_BASE_URL` | Yes | Base URL for internal OpenAI endpoint |
| `QDRANT_PATH` | Yes | Local Qdrant data directory |

### 3. Pre-index documents (one-time)

```bash
# Index court case XMLs (stored as-is: full text in SQLite + 1 doc embedding)
python3 ingest.py /path/to/case.xml --on-conflict replace

# Index PG document XMLs (1 doc embedding per PG doc)
python3 ingest.py /path/to/pg_doc.xml --on-conflict replace

# Skip LLM doc summary for PG docs (uses title + content for embedding instead)
python3 ingest.py /path/to/pg_doc.xml --on-conflict skip --no-ai-summary
```

### 4. Run the pipeline

```bash
python3 run.py /path/to/CaseNewsAlert_XXXX.xml
```

## File Structure

```
poc1/
├── agent.py              # Root SequentialAgent
├── run.py                # CLI runner
├── ingest.py             # Pre-indexing (cases + PG docs) -- doc-level only
├── agents/
│   ├── alert_ingestion.py
│   ├── filter_agent.py
│   ├── alert_processing.py
│   ├── case_processing.py
│   ├── retrieval.py
│   ├── matching.py
│   ├── reasoning.py
│   └── report.py
├── tools/
│   ├── xml_parsers.py    # Alert, Court Case, PG Doc XML parsing
│   ├── chunking.py       # Paragraph-level chunking (used at runtime only)
│   ├── embeddings.py     # SentenceTransformer + Qdrant
│   ├── retrieval.py      # Two-level hybrid search + RRF fusion + citation matching
│   ├── metadata_db.py    # SQLite read helpers (cases + PG)
│   └── llm_helper.py     # Direct OpenAI calls
├── prompts/
│   ├── chunk_summary.py  # Per-chunk structured summary
│   ├── case_summary.py   # Comprehensive case summary + retrieval profile
│   ├── matching.py       # Section-level impact matching
│   └── reasoning.py      # WHERE/WHAT/WHY suggestions
├── requirements.txt
├── .env
└── data/
    ├── court_cases.db    # Case metadata + full text (SQLite)
    ├── pg_docs.db        # PG metadata (SQLite)
    ├── qdrant/           # Vector store (doc-level only)
    │   ├── case_doc_index# One point per case document
    │   └── pg_doc_index  # One point per PG document
    ├── texts/
    │   ├── court_cases/  # Case .txt files
    │   └── pg_docs/      # PG .txt files
    └── reports/          # Generated reports
```

## Qdrant Collections (doc-level only)

| Collection | Contents | Created By |
|---|---|---|
| `case_doc_index` | One point per case document | `ingest.py` (case XML) |
| `pg_doc_index` | One point per PG document | `ingest.py` (PG XML) |

No paragraph-level collections (`case_chunks`, `pg_chunks`) are created at pre-index. Case chunking happens at runtime in the CaseProcessingAgent.

## Report Output

The pipeline generates a structured JSON report containing:
- Alert metadata
- Case summary and keywords
- List of impacted PG documents with:
  - Matched sections/subsections
  - Match strength (HIGH/MEDIUM/LOW)
  - Change suggestions: WHERE / WHAT / WHY
  - Priority rating

Reports are saved to `data/reports/report_{LNI}.json`.
