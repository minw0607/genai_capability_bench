"""Semantic similarity helpers."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_similarity(prediction: str, reference: str) -> float:
    """Dependency-light semantic-ish similarity for offline demos.

    This is intentionally not a replacement for embedding similarity. It gives
    smoke tests and demos a deterministic local signal; workflows can opt into
    provider embeddings when configured.
    """

    if not prediction or not reference:
        return 0.0
    try:
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit([prediction, reference])
        matrix = vectorizer.transform([prediction, reference])
        return float(cosine_similarity(matrix[0], matrix[1])[0][0])
    except ValueError:
        # Empty or punctuation-only text can still reach this point after the
        # vectorizer tokenization step. Treat it as no semantic similarity.
        return 0.0


def best_tfidf_similarity(prediction: str, references: list[str]) -> float:
    if not references:
        return 0.0
    return max(tfidf_similarity(prediction, ref) for ref in references)


def best_semantic_similarity(
    prediction: str,
    references: list[str],
    *,
    mode: str = "tfidf",
) -> tuple[float, str]:
    """Return best semantic similarity and the method used.

    Modes:
    - ``tfidf``: deterministic local TF-IDF cosine similarity.
    - ``api_embeddings_if_configured``: use provider embeddings only when an
      embedding model is configured; otherwise fall back to TF-IDF.
    - ``api_embeddings``: require provider embeddings and raise on missing
      configuration or API errors.
    """

    references = references or []
    normalized_mode = (mode or "tfidf").strip().lower()
    if normalized_mode == "tfidf":
        return best_tfidf_similarity(prediction, references), "tfidf"

    if normalized_mode not in {"api_embeddings", "api_embeddings_if_configured"}:
        raise ValueError(
            "semantic similarity mode must be 'tfidf', 'api_embeddings', "
            "or 'api_embeddings_if_configured'"
        )

    if not _embedding_model_name():
        if normalized_mode == "api_embeddings_if_configured":
            return best_tfidf_similarity(prediction, references), "tfidf"
        raise RuntimeError("OPENAI_EMBEDDING_MODEL or EMBEDDING_MODEL must be set to use API embeddings.")

    if not prediction or not references:
        return 0.0, "api_embeddings"

    try:
        return best_embedding_similarity(prediction, references), "api_embeddings"
    except Exception:
        if normalized_mode == "api_embeddings_if_configured":
            return best_tfidf_similarity(prediction, references), "tfidf_fallback"
        raise


def best_embedding_similarity(prediction: str, references: list[str]) -> float:
    """Compute best cosine similarity using an OpenAI-compatible embedding API."""

    if not prediction or not references:
        return 0.0
    texts = [prediction] + [ref for ref in references if ref]
    if len(texts) <= 1:
        return 0.0
    vectors = _embed_texts(tuple(texts))
    prediction_vector = vectors[0]
    reference_vectors = vectors[1:]
    return max(_cosine(prediction_vector, reference_vector) for reference_vector in reference_vectors)


@lru_cache(maxsize=2048)
def _embed_texts(texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
    load_dotenv()
    model = _embedding_model_name()
    if not model:
        raise RuntimeError("OPENAI_EMBEDDING_MODEL or EMBEDDING_MODEL must be set to use API embeddings.")

    response = _embedding_client().embeddings.create(model=model, input=list(texts))
    return tuple(tuple(item.embedding) for item in response.data)


@lru_cache(maxsize=1)
def _embedding_client():
    from openai import AzureOpenAI, OpenAI

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_version = os.environ.get("OPENAI_API_VERSION", "")
    headers = {}
    header_name = os.environ.get("OPENAI_APIM_HEADER_NAME", "")
    header_value = os.environ.get("OPENAI_APIM_SUBSCRIPTION_KEY", "")
    if header_name and header_value:
        headers[header_name] = header_value

    if api_version:
        return AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=base_url,
            default_headers=headers or None,
        )
    return OpenAI(api_key=api_key, base_url=base_url, default_headers=headers or None)


def _embedding_model_name() -> str:
    load_dotenv()
    return os.environ.get("OPENAI_EMBEDDING_MODEL") or os.environ.get("EMBEDDING_MODEL", "")


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    denominator = np.linalg.norm(left_array) * np.linalg.norm(right_array)
    if denominator == 0:
        return 0.0
    return float(np.dot(left_array, right_array) / denominator)
