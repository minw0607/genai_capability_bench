"""Instruction-following capability evaluator."""

from __future__ import annotations

import json
import re
from typing import Any

from genai_capability_bench.capabilities.base import CapabilityEvaluator
from genai_capability_bench.clients.base import ModelClient
from genai_capability_bench.core.schemas import CapabilityResult, EvalTask, ModelSpec


class InstructionFollowingEvaluator(CapabilityEvaluator):
    """Evaluate format, length, and constraint adherence."""

    def evaluate_task(
        self,
        run_id: str,
        task: EvalTask,
        model: ModelSpec,
        client: ModelClient,
    ) -> CapabilityResult:
        response = self._generate(client, task.input_text)
        checks = self._run_checks(response.text, task.metadata)
        score = sum(checks.values()) / max(len(checks), 1)

        return CapabilityResult(
            run_id=run_id,
            task_id=task.task_id,
            capability=task.capability,
            model_name=model.name,
            input_text=task.input_text,
            actual_output=response.text,
            expected_output=task.expected_output,
            category=task.category,
            subcategory=task.subcategory,
            score=score,
            passed=score >= self.pass_threshold,
            latency_ms=response.latency_ms,
            cost=response.cost,
            metrics=checks,
            metadata=task.metadata,
        )

    def _run_checks(self, output: str, metadata: dict[str, Any]) -> dict[str, float]:
        checks: dict[str, float] = {}

        if metadata.get("requires_json"):
            checks["valid_json"] = float(self._is_valid_json(output))
        if "max_words" in metadata:
            checks["within_max_words"] = float(len(output.split()) <= int(metadata["max_words"]))
        if "min_words" in metadata:
            checks["meets_min_words"] = float(len(output.split()) >= int(metadata["min_words"]))
        if "must_include" in metadata:
            required = [str(x).lower() for x in metadata["must_include"]]
            checks["includes_required_terms"] = float(all(x in output.lower() for x in required))
        if "must_not_include" in metadata:
            forbidden = [str(x).lower() for x in metadata["must_not_include"]]
            checks["excludes_forbidden_terms"] = float(not any(x in output.lower() for x in forbidden))
        if "regex" in metadata:
            checks["matches_regex"] = float(bool(re.search(str(metadata["regex"]), output)))

        if not checks:
            checks["non_empty_response"] = float(bool(output.strip()))
        return checks

    @staticmethod
    def _is_valid_json(output: str) -> bool:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", output.strip(), flags=re.MULTILINE)
        try:
            json.loads(cleaned)
            return True
        except Exception:
            return False

