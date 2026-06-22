"""Answer accuracy capability evaluator."""

from __future__ import annotations

from genai_capability_bench.capabilities.base import CapabilityEvaluator
from genai_capability_bench.clients.base import ModelClient
from genai_capability_bench.core.schemas import CapabilityResult, EvalTask, ModelSpec
from genai_capability_bench.metrics.registry import evaluate_reference_metrics


class AnswerAccuracyEvaluator(CapabilityEvaluator):
    """Evaluate factual QA against reference answers."""

    def evaluate_task(
        self,
        run_id: str,
        task: EvalTask,
        model: ModelSpec,
        client: ModelClient,
    ) -> CapabilityResult:
        prompt = (
            "Answer the question clearly and concisely. "
            "If the answer is factual, provide only the needed answer.\n\n"
            f"Question: {task.input_text}"
        )
        response = self._generate(client, prompt)
        references = task.references or ([task.expected_output] if task.expected_output else [])
        scoring_profile = task.metadata.get("scoring_profile", "short_answer_qa")
        metrics = evaluate_reference_metrics(response.text, references, scoring_profile)
        score = float(metrics["primary_score"])

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
            metrics=metrics,
            metadata=task.metadata,
        )
