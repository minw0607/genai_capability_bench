"""Diagnostics for result review."""

from __future__ import annotations

import ast
from typing import Any

import pandas as pd


def parse_metric_dict(value: Any) -> dict[str, Any]:
    """Parse metrics from dict or CSV string representation."""

    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def add_answer_accuracy_diagnostics(results_df: pd.DataFrame, pass_threshold: float) -> pd.DataFrame:
    """Add review-priority and metric-disagreement columns."""

    if results_df.empty:
        return results_df.copy()

    df = results_df.copy()
    metric_rows = [parse_metric_dict(v) for v in df["metrics"]]
    for key in ["exact_match", "contains_match", "token_f1", "tfidf_similarity"]:
        df[key] = [float(m.get(key, 0.0) or 0.0) for m in metric_rows]

    df["metric_spread"] = df[["exact_match", "contains_match", "token_f1", "tfidf_similarity"]].max(axis=1) - df[
        ["exact_match", "contains_match", "token_f1", "tfidf_similarity"]
    ].min(axis=1)
    df["metric_disagreement"] = df["metric_spread"] >= 0.5

    def priority(score: float, disagreement: bool) -> str:
        if score < 0.4:
            return "High review priority"
        if score < pass_threshold:
            return "Near threshold"
        if disagreement:
            return "Metric disagreement"
        return "Looks good"

    df["review_priority"] = [priority(float(s), bool(d)) for s, d in zip(df["score"], df["metric_disagreement"])]
    return df


def metric_guide_table() -> pd.DataFrame:
    """Human-readable metric methodology table."""

    return pd.DataFrame(
        [
            {
                "Metric": "Exact Match",
                "What it checks": "Normalized model answer exactly equals at least one reference answer.",
                "Best for": "Short factual answers, IDs, dates, names.",
                "Limitations": "Too strict for paraphrases and explanatory answers.",
            },
            {
                "Metric": "Contains Match",
                "What it checks": "A normalized reference answer appears inside the model answer.",
                "Best for": "Cases where the model gives a sentence but includes the correct short answer.",
                "Limitations": "Can over-credit answers that contain the right phrase but add wrong claims.",
            },
            {
                "Metric": "Token F1",
                "What it checks": "Overlap between prediction tokens and reference tokens.",
                "Best for": "Partial matches and multi-token answers.",
                "Limitations": "Lexical only; does not understand meaning.",
            },
            {
                "Metric": "TF-IDF Similarity",
                "What it checks": "Cosine similarity over local TF-IDF vectors.",
                "Best for": "Lightweight offline semantic-ish signal in demos.",
                "Limitations": "Not a true embedding model; should be upgraded for production scoring.",
            },
            {
                "Metric": "Composite Score",
                "What it checks": "Maximum of exact match, contains match, and a weighted F1/TF-IDF blend.",
                "Best for": "A practical starter score across short answer tasks.",
                "Limitations": "Should be complemented with embeddings or LLM-judge rubrics for open-ended answers.",
            },
        ]
    )


def score_interpretation_table() -> pd.DataFrame:
    """Score interpretation bands."""

    return pd.DataFrame(
        [
            {"Score Band": "0.90 - 1.00", "Interpretation": "Strong match; usually correct for short factual QA."},
            {"Score Band": "0.70 - 0.89", "Interpretation": "Likely acceptable; review for high-stakes use."},
            {"Score Band": "0.40 - 0.69", "Interpretation": "Partial or uncertain; human review recommended."},
            {"Score Band": "0.00 - 0.39", "Interpretation": "Likely incorrect or unsupported."},
        ]
    )

