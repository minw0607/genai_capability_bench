"""Capability evaluator registry."""

from __future__ import annotations

from genai_capability_bench.capabilities.answer_accuracy import AnswerAccuracyEvaluator
from genai_capability_bench.capabilities.base import CapabilityEvaluator
from genai_capability_bench.capabilities.instruction_following import InstructionFollowingEvaluator
from genai_capability_bench.capabilities.reasoning_logic import ReasoningLogicEvaluator
from genai_capability_bench.capabilities.truthfulness import TruthfulnessEvaluator
from genai_capability_bench.core.schemas import Capability


def get_evaluator(capability: Capability, pass_threshold: float = 0.7) -> CapabilityEvaluator:
    if capability == Capability.ANSWER_ACCURACY:
        return AnswerAccuracyEvaluator(pass_threshold=pass_threshold)
    if capability == Capability.TRUTHFULNESS:
        return TruthfulnessEvaluator(pass_threshold=pass_threshold)
    if capability == Capability.INSTRUCTION_FOLLOWING:
        return InstructionFollowingEvaluator(pass_threshold=pass_threshold)
    if capability == Capability.REASONING_LOGIC:
        return ReasoningLogicEvaluator(pass_threshold=pass_threshold)
    raise ValueError(f"No evaluator registered for capability: {capability.value}")

