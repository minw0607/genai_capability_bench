"""Lexical metrics aligned with the existing RAG framework conventions."""

from __future__ import annotations

import re
from collections import Counter


def normalize_text(text: str | None) -> str:
    """Lowercase, remove punctuation/articles, and collapse whitespace."""

    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def contains_match(prediction: str, reference: str) -> float:
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    return 1.0 if ref and ref in pred else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    tp = sum(common.values())
    if tp == 0:
        return 0.0
    precision = tp / len(pred_tokens)
    recall = tp / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def best_reference_score(prediction: str, references: list[str], scorer) -> float:
    if not references:
        return 0.0
    return max(scorer(prediction, ref) for ref in references)

