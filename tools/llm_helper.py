"""
tools/llm_helper.py -- Direct OpenAI / Azure OpenAI LLM Calls
===============================================================
Shared helper for agents that need direct LLM calls outside of ADK's
LlmAgent framework (e.g., for structured JSON extraction in custom agents).

Auto-detects Azure vs standard OpenAI based on environment variables:
  - If AZURE_OPENAI_API_KEY is set -> uses AzureOpenAI client
  - Otherwise -> uses standard OpenAI client
"""

from __future__ import annotations

import os
import re
from typing import Optional

_client = None
_is_azure = False


def _get_client():
    global _client, _is_azure
    if _client is not None:
        return _client

    azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")

    if azure_key and azure_endpoint:
        from openai import AzureOpenAI
        _client = AzureOpenAI(
            api_key=azure_key,
            azure_endpoint=azure_endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        )
        _is_azure = True
        print("  [LLM] Using Azure OpenAI endpoint")
        return _client

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "No LLM credentials found. Set either AZURE_OPENAI_API_KEY + "
            "AZURE_OPENAI_ENDPOINT, or OPENAI_API_KEY."
        )
    from openai import OpenAI
    kwargs: dict = {"api_key": api_key}
    base_url = os.getenv("OPENAI_BASE_URL", "")
    if base_url:
        kwargs["base_url"] = base_url
    _client = OpenAI(**kwargs)
    return _client


def _get_model(model_type: str) -> str:
    if _is_azure:
        return (
            os.getenv("AZURE_FAST_MODEL", "gpt-4o-mini")
            if model_type == "fast"
            else os.getenv("AZURE_STRONG_MODEL", "gpt-4o")
        )
    return (
        os.getenv("OPENAI_FAST_MODEL", "gpt-4o-mini")
        if model_type == "fast"
        else os.getenv("OPENAI_STRONG_MODEL", "gpt-4o")
    )


def call_llm(
    system: str,
    user: str,
    model_type: str = "fast",
    temperature: float = 0,
) -> str:
    """
    Single chat completion call.
    model_type: 'fast' or 'strong'
    Returns the assistant text content.
    """
    client = _get_client()
    model = _get_model(model_type)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def call_llm_json(
    system: str,
    user: str,
    model_type: str = "fast",
) -> str:
    """Call LLM and strip markdown fences from the response."""
    raw = call_llm(system, user, model_type)
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
    return raw
