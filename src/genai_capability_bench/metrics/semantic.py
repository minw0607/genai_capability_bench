"""Lightweight semantic similarity helpers."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_similarity(prediction: str, reference: str) -> float:
    """Dependency-light semantic-ish similarity for offline demos.

    This is intentionally not a replacement for embedding similarity. It gives
    smoke tests and demos a deterministic local signal; production configs can
    add provider embeddings later.
    """

    if not prediction or not reference:
        return 0.0
    vectorizer = TfidfVectorizer().fit([prediction, reference])
    matrix = vectorizer.transform([prediction, reference])
    return float(cosine_similarity(matrix[0], matrix[1])[0][0])


def best_tfidf_similarity(prediction: str, references: list[str]) -> float:
    if not references:
        return 0.0
    return max(tfidf_similarity(prediction, ref) for ref in references)

