"""Answer accuracy capability evaluator."""

from __future__ import annotations

from genai_capability_bench.capabilities.base import CapabilityEvaluator
from genai_capability_bench.clients.base import ModelClient
from genai_capability_bench.core.schemas import CapabilityResult, EvalTask, ModelSpec
from genai_capability_bench.metrics.lexical import (
    best_reference_score,
    contains_match,
    exact_match,
    token_f1,
)
from genai_capability_bench.metrics.semantic import best_tfidf_similarity


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

        em = best_reference_score(response.text, references, exact_match)
        contains = best_reference_score(response.text, references, contains_match)
        f1 = best_reference_score(response.text, references, token_f1)
        semantic = best_tfidf_similarity(response.text, references)
        score = max(em, contains, 0.45 * f1 + 0.55 * semantic)

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
            metrics={
                "exact_match": em,
                "contains_match": contains,
                "token_f1": f1,
                "tfidf_similarity": semantic,
            },
            metadata=task.metadata,
        )

