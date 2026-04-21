# POC-1: Legal Case Impact Analysis Pipeline

Multi-agent system built with Google ADK that automatically analyses incoming case news alerts and determines their impact on stored Practical Guidance (PG) documents.

## How It Works

### Step 1: Pre-Index (one-time)

Parse **both** court case XMLs and PG document XMLs.  Cases are stored as-is
(one doc embedding each).  PG documents get **both** a doc-level embedding
*and* paragraph-level embeddings so the runtime retriever can do real
paragraph-to-paragraph matching.

```
Case XMLs  ─→ ingest.py ─→ court_cases.db (metadata + full text) + Qdrant (case_doc_index)
PG XMLs    ─→ ingest.py ─→ pg_docs.db (metadata)                + Qdrant (pg_doc_index, pg_chunks)
```

### Step 2: Process Alert (runtime)

When an alert XML arrives:
1. Parse the alert, identify the case by `cite_ref` / `cite_def`, fall back to LNI
2. Retrieve the case as-is from `court_cases.db`
3. Chunk at paragraph level; generate per-chunk AI summaries + query embeddings
4. Build the document-level retrieval profile and a **sliding-window pooled**
   full-document embedding (replaces the previous 16 kB truncation)

### Step 3: Three-Level Hybrid Retrieval with Reciprocal Rank Fusion

| Signal | Source | Target | Threshold |
|---|---|---|---|
| L0 (broad sweep) | case summary embedding | `pg_doc_index` | `sim*0.85` |
| L1 (doc refine) | pooled full-doc + retrieval profile | `pg_doc_index` | `sim` |
| L2 (paragraph) | per-case-paragraph embeddings | `pg_chunks` | `sim*0.95` |
| BM25 | case keywords | global corpus over all PG docs (cached) | — |
| Citation | alert `cite_defs` + `cite_refs` + case `cite_ref` | PG `cite_ids` | exact after normalising |

All five signals contribute `weight / (k + rank)` per doc (reciprocal rank
fusion).  Weights are tunable via `RRF_W_L0 / RRF_W_L1 / RRF_W_L2 / RRF_W_BM25
/ RRF_W_CITE` in `.env`.  L2 is **no longer stricter than L1** -- that
inversion was the main recall regression.

### Steps 4-5: Matching + Reasoning

For each affected PG document: identify which sections are impacted and generate WHERE/WHAT/WHY change suggestions.

## Architecture

```
[1. AlertIngestionAgent]  -- reads alert file
[2. FilterAgent]          -- optional practice area filter (B&F)
[3. AlertProcessingAgent] -- parses XML, extracts metadata + key holdings + cite_defs/cite_refs
[4. CaseProcessingAgent]  -- retrieves case from DB, chunks, summaries, pooled full-doc embedding
[5. RetrievalAgent]       -- three-level hybrid retrieval with RRF (L0 + L1 + L2 + BM25 + citations)
[6. MatchingAgent]        -- per-PG-doc section-level impact matching (with fallback evidence)
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

# Index PG document XMLs (doc-level + paragraph-level embeddings)
python3 ingest.py /path/to/pg_doc.xml --on-conflict replace

# Skip LLM doc summary for PG docs (uses title + content for embedding instead)
python3 ingest.py /path/to/pg_doc.xml --on-conflict skip --no-ai-summary

# Batch-ingest a folder
python3 ingest.py --batch-dir /path/to/pg_xmls --on-conflict replace

# Only (re)build the paragraph-level pg_chunks collection, keeping
# metadata + doc-level embeddings intact.  Use this when you upgrade from
# the old pipeline that didn't have pg_chunks.
python3 ingest.py --batch-dir /path/to/pg_xmls --rebuild-chunks
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
│   ├── chunking.py       # Paragraph-level chunking (runtime cases + ingest PG)
│   ├── embeddings.py     # SentenceTransformer + Qdrant (+ sliding-window encode)
│   ├── retrieval.py      # Three-level hybrid search with RRF fusion
│   ├── metadata_db.py    # SQLite read helpers (cases + PG)
│   ├── logging_setup.py  # Structured logging (console + rotating file)
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
    ├── qdrant/           # Vector store
    │   ├── case_doc_index# One point per case document
    │   ├── pg_doc_index  # One point per PG document (doc-level)
    │   └── pg_chunks     # Many points per PG doc (paragraph / section)
    ├── bm25_pg.pkl       # Cached global BM25 index over PG corpus
    ├── logs/             # Rotating log files (poc1.log + backups)
    ├── texts/
    │   ├── court_cases/  # Case .txt files
    │   └── pg_docs/      # PG .txt files
    └── reports/          # Generated reports
```

## Qdrant Collections

| Collection | Contents | Created By |
|---|---|---|
| `case_doc_index` | One point per case document | `ingest.py` (case XML) |
| `pg_doc_index` | One point per PG document (summary/profile embedding) | `ingest.py` (PG XML) |
| `pg_chunks` | Many points per PG document (paragraph / section vectors) | `ingest.py` (PG XML) |

Case chunking still happens at runtime in the `CaseProcessingAgent`; only PG
content is pre-chunked.

## Debugging a bad retrieval

1. Tail the rotating log for this alert:

   ```bash
   tail -f data/logs/poc1.log | grep "alert_id=<LNI>"
   ```

   Every line in the pipeline carries the alert's LNI so you can filter end
   to end.  Console output stays at `INFO`; the file captures `DEBUG` too.

2. Inspect the retrieval scorecard: the `RetrievalAgent` log lines include
   per-doc component breakdowns, e.g.

   ```
   [rank] 140374 score=0.1234 hits=4 paras=3 L0=0.030 L1=0.028 L2=0.045 BM25=0.012 CITE=0.018 | Repayment, prepayment...
   ```

   If the winning docs have `L2=0.000` you probably still have an old
   `pg_doc_index` without `pg_chunks`; run
   `python3 ingest.py --batch-dir <pg_xmls> --rebuild-chunks` to fix it.

3. If specific documents are *just* below the cut-off, look for "near-miss"
   debug lines (`LOG_LEVEL_FILE=DEBUG`) and lower `RETRIEVAL_SIM_THRESHOLD`
   or increase the limit for that level.

4. Citation misses: the `normalize_citation` helper in
   `tools/retrieval.py` logs every alert/case cite at `INFO` and every PG
   overlap at `DEBUG`.  Drift between the alert's `normcite` and the PG
   `cite_ids` (e.g. whitespace, brackets) is handled there -- if you still
   see 0 overlap while the values look identical, check the raw strings in
   the log (they'll be lower-cased but otherwise faithful).

### Tuning knobs (`.env`)

| Variable | Default | Effect |
|---|---|---|
| `RETRIEVAL_SIM_THRESHOLD` | 0.35 | Base cosine cut-off for L1.  L0 uses `*0.85`, L2 uses `*0.95`. |
| `RETRIEVAL_TOP_K` | 20 | Max docs returned |
| `RETRIEVAL_L0_LIMIT` / `_L1_LIMIT` / `_L2_PER_PARA_LIMIT` | 150 / 80 / 12 | Qdrant `limit` per call |
| `RETRIEVAL_MIN_HITS` | 1 | Minimum semantic hits per doc (citation / BM25 only still passes) |
| `RETRIEVAL_SCORE_GAP` | 0.12 | Adaptive tail cut-off (0 disables) |
| `RRF_K` | 60 | Reciprocal rank fusion constant |
| `RRF_W_L0 / L1 / L2 / BM25 / CITE` | 1.0 / 1.0 / 1.2 / 0.5 / 1.5 | Per-signal weights |
| `LOG_LEVEL_CONSOLE` / `LOG_LEVEL_FILE` | INFO / DEBUG | Log verbosity |

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
