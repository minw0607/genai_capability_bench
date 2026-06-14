"""Client factory."""

from __future__ import annotations

from genai_capability_bench.clients.base import MockModelClient, ModelClient
from genai_capability_bench.core.schemas import ModelSpec


def create_client(spec: ModelSpec) -> ModelClient:
    """Create a model client from a model spec."""

    if spec.provider == "mock":
        answers = spec.metadata.get("answers", {})
        return MockModelClient(answers=answers)
    if spec.provider in {"openai", "azure_openai", "openai_compatible"}:
        from genai_capability_bench.clients.openai_compatible import OpenAICompatibleClient

        return OpenAICompatibleClient(spec)
    raise ValueError(f"Unsupported provider: {spec.provider}")
