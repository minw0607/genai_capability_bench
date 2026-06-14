"""Model client abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelResponse:
    """Normalized model response."""

    text: str
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    raw: Any = None


class ModelClient(ABC):
    """Minimal interface required by capability evaluators."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> ModelResponse:
        """Generate a response for a prompt."""


class MockModelClient(ModelClient):
    """Deterministic client for smoke tests and notebook demos without API calls."""

    def __init__(self, answers: dict[str, str] | None = None):
        self.answers = answers or {}

    def generate(self, prompt: str, system: str | None = None) -> ModelResponse:
        normalized = " ".join(prompt.lower().split())
        for key, answer in self.answers.items():
            if key.lower() in normalized:
                return ModelResponse(text=answer, latency_ms=0.0)
        return ModelResponse(text="I do not have enough information to answer.", latency_ms=0.0)

