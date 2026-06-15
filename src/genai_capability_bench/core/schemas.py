"""Shared schemas for capability evaluation runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Capability(str, Enum):
    """Top-level capability families."""

    ANSWER_ACCURACY = "answer_accuracy"
    TRUTHFULNESS = "truthfulness"
    INSTRUCTION_FOLLOWING = "instruction_following"
    REASONING_LOGIC = "reasoning_logic"
    RAG = "rag"
    TOOL_USE = "tool_use"
    AGENTIC_TASK_COMPLETION = "agentic_task_completion"


@dataclass
class ModelSpec:
    """Provider/model configuration used for a run."""

    name: str
    provider: str = "mock"
    model: str = "mock-model"
    api_version: str | None = None
    temperature: float | None = 0.0
    max_tokens: int | None = 1000
    token_parameter: str | None = "max_tokens"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalTask:
    """A single evaluation task."""

    task_id: str
    capability: Capability
    input_text: str
    expected_output: str | None = None
    category: str = "general"
    subcategory: str | None = None
    references: list[str] = field(default_factory=list)
    incorrect_references: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityResult:
    """Normalized result emitted by every capability evaluator."""

    run_id: str
    task_id: str
    capability: Capability
    model_name: str
    input_text: str
    actual_output: str
    expected_output: str | None
    category: str
    score: float
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    subcategory: str | None = None
    cost: float | None = None
    latency_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["capability"] = self.capability.value
        return data


@dataclass
class RunMetadata:
    """Metadata saved with every benchmark run."""

    run_id: str
    started_at: str
    config_path: str | None
    models: list[ModelSpec]
    capabilities: list[Capability]
    notes: str = ""

    @classmethod
    def create(
        cls,
        run_id: str,
        config_path: str | None,
        models: list[ModelSpec],
        capabilities: list[Capability],
        notes: str = "",
    ) -> "RunMetadata":
        return cls(
            run_id=run_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            config_path=config_path,
            models=models,
            capabilities=capabilities,
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = [c.value for c in self.capabilities]
        return data
