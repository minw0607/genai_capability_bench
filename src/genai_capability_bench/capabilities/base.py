"""Base evaluator utilities."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

from genai_capability_bench.clients.base import ModelClient
from genai_capability_bench.core.schemas import CapabilityResult, EvalTask, ModelSpec


class CapabilityEvaluator(ABC):
    """Base class for capability evaluators."""

    def __init__(self, pass_threshold: float = 0.7):
        self.pass_threshold = pass_threshold

    @abstractmethod
    def evaluate_task(
        self,
        run_id: str,
        task: EvalTask,
        model: ModelSpec,
        client: ModelClient,
    ) -> CapabilityResult:
        """Evaluate one task."""

    def _generate(self, client: ModelClient, prompt: str, system: str | None = None):
        start = time.time()
        response = client.generate(prompt, system=system)
        if response.latency_ms is None:
            response.latency_ms = (time.time() - start) * 1000
        return response

