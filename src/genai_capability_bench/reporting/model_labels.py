"""Public model labels for reports and notebook displays."""

from __future__ import annotations

import os
import re

from genai_capability_bench.core.schemas import ModelSpec


def model_public_label(model: ModelSpec) -> str:
    """Return a non-sensitive model label for human-facing reports."""

    explicit = model.metadata.get("display_name") if isinstance(model.metadata, dict) else None
    explicit = explicit or os.environ.get("GENAI_BENCH_MODEL_DISPLAY_NAME")
    if explicit:
        return str(explicit)

    raw = str(model.model or model.name or "").lower()
    family = _model_family_label(raw)
    provider = _provider_label(model)
    return f"{family} ({provider})" if provider else family


def models_public_label(models: list[ModelSpec]) -> str:
    """Return a comma-separated public label for one or more models."""

    return ", ".join(model_public_label(model) for model in models)


def _model_family_label(raw: str) -> str:
    gpt_match = re.search(r"gpt[-_]?(\d+)(?:[-_]?(\d+|o))?", raw)
    if gpt_match:
        major = gpt_match.group(1)
        minor = gpt_match.group(2)
        if minor:
            return f"GPT {major}o" if minor == "o" else f"GPT {major}-{minor}"
        return f"GPT {major}"
    if "llama" in raw:
        return "Llama"
    if "claude" in raw:
        return "Claude"
    if "mistral" in raw:
        return "Mistral"
    if "mock" in raw:
        return "Demo Mock Model"
    return "Target LLM"


def _provider_label(model: ModelSpec) -> str:
    provider = str(model.provider or "").lower()
    base_url = os.environ.get("OPENAI_BASE_URL", "").lower()
    api_version = model.api_version or os.environ.get("OPENAI_API_VERSION", "")
    if provider == "mock":
        return "Local"
    if provider == "openai":
        return "OpenAI"
    if provider == "openai_compatible":
        if api_version:
            return "Azure"
        if "localhost" in base_url or "127.0.0.1" in base_url:
            return "Local"
        if "groq" in base_url:
            return "Groq"
        if "together" in base_url:
            return "Together AI"
        return "OpenAI-Compatible"
    return provider.replace("_", " ").title()
