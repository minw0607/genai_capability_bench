"""Reasoning and logic capability evaluator."""

from __future__ import annotations

from genai_capability_bench.capabilities.base import CapabilityEvaluator
from genai_capability_bench.clients.base import ModelClient
from genai_capability_bench.core.schemas import CapabilityResult, EvalTask, ModelSpec
from genai_capability_bench.metrics.lexical import contains_match, exact_match


class ReasoningLogicEvaluator(CapabilityEvaluator):
    """Evaluate reasoning tasks with final-answer matching."""

    def evaluate_task(
        self,
        run_id: str,
        task: EvalTask,
        model: ModelSpec,
        client: ModelClient,
    ) -> CapabilityResult:
        prompt = (
            "Solve the problem. Think carefully, but end with a final answer in the form "
            "'Final answer: ...'.\n\n"
            f"Problem: {task.input_text}"
        )
        response = self._generate(client, prompt)
        expected = task.expected_output or ""
        final_answer = self._extract_final_answer(response.text)

        em = exact_match(final_answer, expected)
        contains = contains_match(final_answer, expected)
        score = max(em, contains)

        return CapabilityResult(
            run_id=run_id,
            task_id=task.task_id,
            capability=task.capability,
            model_name=model.name,
            input_text=task.input_text,
            actual_output=response.text,
            expected_output=expected,
            category=task.category,
            subcategory=task.subcategory,
            score=score,
            passed=score >= self.pass_threshold,
            latency_ms=response.latency_ms,
            cost=response.cost,
            metrics={
                "final_answer": final_answer,
                "exact_match": em,
                "contains_match": contains,
            },
            metadata=task.metadata,
        )

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        marker = "final answer:"
        lower = text.lower()
        if marker in lower:
            idx = lower.rfind(marker)
            return text[idx + len(marker) :].strip()
        return text.strip().splitlines()[-1] if text.strip() else ""

