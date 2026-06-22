"""Reference-based text generation metrics used across capability notebooks."""

from __future__ import annotations

import math
from collections import Counter

from genai_capability_bench.metrics.lexical import normalize_text


def bleu_score(prediction: str, reference: str, max_n: int = 4) -> float:
    """Sentence-level BLEU with simple smoothing.

    BLEU is included as a secondary diagnostic because it is widely known from
    machine translation. It is usually not a primary metric for short factual QA.
    """

    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_counts = _ngram_counts(pred_tokens, n)
        ref_counts = _ngram_counts(ref_tokens, n)
        if not pred_counts:
            precisions.append(1e-9)
            continue
        overlap = sum((pred_counts & ref_counts).values())
        precisions.append((overlap + 1) / (sum(pred_counts.values()) + 1))

    brevity = 1.0 if len(pred_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(pred_tokens))
    return float(brevity * math.exp(sum(math.log(p) for p in precisions) / max_n))


def rouge_n(prediction: str, reference: str, n: int = 1) -> float:
    """ROUGE-N recall over normalized tokens."""

    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = _ngram_counts(pred_tokens, n)
    ref_counts = _ngram_counts(ref_tokens, n)
    if not ref_counts:
        return 0.0
    overlap = sum((pred_counts & ref_counts).values())
    return float(overlap / sum(ref_counts.values()))


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure based on longest common subsequence."""

    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def best_reference_generation_scores(prediction: str, references: list[str]) -> dict[str, float]:
    """Return best BLEU/ROUGE scores across references."""

    if not references:
        return {"bleu": 0.0, "rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    return {
        "bleu": max(bleu_score(prediction, ref) for ref in references),
        "rouge_1": max(rouge_n(prediction, ref, n=1) for ref in references),
        "rouge_2": max(rouge_n(prediction, ref, n=2) for ref in references),
        "rouge_l": max(rouge_l(prediction, ref) for ref in references),
    }


def _ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(max(len(tokens) - n + 1, 0)))


def _lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]
