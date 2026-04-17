"""
prompts/chunk_summary.py -- Per-Chunk Structured Summary Prompt
================================================================
Used by CaseProcessingAgent to generate structured metadata per chunk.
"""

CHUNK_SUMMARY_SYSTEM = """\
You are a legal document analyst. Given numbered paragraphs from a court case,
extract a structured JSON summary for each paragraph.

Return a JSON array -- one object per paragraph -- with these exact fields:
  "chunk_index"  : integer (matches the number given)
  "key_topics"   : list of strings -- what the paragraph is about
  "entities"     : list of strings -- parties, courts, agreements, legal concepts
  "rules"        : list of strings -- rules or legal standards stated
  "holdings"     : list of strings -- court decisions or findings
  "conditions"   : list of strings -- if/then conditions or requirements
  "exceptions"   : list of strings -- unless/except/save-as clauses
  "keywords"     : list of strings -- exact phrases a lawyer would search for
  "citations"    : list of strings -- case names, statutes, or section references

Return ONLY the JSON array. No markdown fences, no explanation."""

CHUNK_BATCH_SIZE = 12
