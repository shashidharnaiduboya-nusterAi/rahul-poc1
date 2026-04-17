"""
agent.py -- ADK Root Agent
==========================
Defines the root SequentialAgent that orchestrates the full POC-1 pipeline:
  Alert Ingestion -> Filter -> Alert Processing -> Case Processing
  -> Retrieval -> Guardrail -> Matching -> Reasoning -> Report
"""

from __future__ import annotations

from google.adk.agents.sequential_agent import SequentialAgent

from agents.alert_ingestion import alert_ingestion_agent
from agents.filter_agent import filter_agent
from agents.alert_processing import alert_processing_agent
from agents.case_processing import case_processing_agent
from agents.retrieval import retrieval_agent
from agents.guardrail import guardrail_agent
from agents.matching import matching_agent
from agents.reasoning import reasoning_agent
from agents.report import report_agent

root_agent = SequentialAgent(
    name="POC1_PipelineAgent",
    description=(
        "End-to-end legal case processing pipeline. "
        "Reads a news alert, finds the corresponding case, "
        "identifies impacted PG documents, matches affected sections, "
        "and generates change suggestions for SME review."
    ),
    sub_agents=[
        alert_ingestion_agent,
        filter_agent,
        alert_processing_agent,
        case_processing_agent,
        retrieval_agent,
        guardrail_agent,
        matching_agent,
        reasoning_agent,
        report_agent,
    ],
)
