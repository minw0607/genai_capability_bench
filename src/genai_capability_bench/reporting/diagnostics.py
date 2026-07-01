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
    for key in [
        "primary_score",
        "exact_match",
        "contains_match",
        "token_f1",
        "semantic_similarity",
        "tfidf_similarity",
        "bleu",
        "rouge_1",
        "rouge_2",
        "rouge_l",
    ]:
        df[key] = [float(m.get(key, 0.0) or 0.0) for m in metric_rows]
    df["scoring_profile"] = [str(m.get("scoring_profile", "")) for m in metric_rows]
    df["semantic_similarity_method"] = [str(m.get("semantic_similarity_method", "")) for m in metric_rows]

    df["semantic_blend_score"] = 0.65 * df["token_f1"] + 0.35 * df["semantic_similarity"]
    df["strict_score"] = df[["exact_match", "semantic_blend_score"]].max(axis=1)
    df["contains_only_credit"] = (
        (df["contains_match"] >= 1.0)
        & (df["exact_match"] < 1.0)
        & (df["primary_score"] < pass_threshold)
    )
    df["reference_shape_warning"] = df.apply(_reference_shape_warning, axis=1)
    metric_cols = ["exact_match", "token_f1", "semantic_similarity", "rouge_l"]
    df["metric_spread"] = df[metric_cols].max(axis=1) - df[metric_cols].min(axis=1)
    df["metric_disagreement"] = df["metric_spread"] >= 0.5
    df["score"] = df["primary_score"]
    if "passed" in df:
        df["passed"] = df["score"] >= pass_threshold

    df["would_pass_contains_rule"] = (
        (df["contains_match"] >= 1.0)
        & (df["exact_match"] < 1.0)
        & (df["primary_score"] < pass_threshold)
    )

    def priority(score: float, disagreement: bool, contains_only_credit: bool, reference_warning: str) -> str:
        if score < 0.4:
            return "High review priority"
        if score < pass_threshold:
            return "Near threshold"
        if reference_warning:
            return "Reference-shape warning"
        if contains_only_credit:
            return "Contains-only credit"
        if disagreement:
            return "Metric disagreement"
        return "Looks good"

    df["review_priority"] = [
        priority(float(s), bool(d), bool(c), str(w))
        for s, d, c, w in zip(
            df["score"], df["metric_disagreement"], df["contains_only_credit"], df["reference_shape_warning"]
        )
    ]
    df["flagged"] = df["review_priority"] != "Looks good"
    df["flag_reason"] = [
        _flag_reason(row, pass_threshold)
        for row in df[
            [
                "score",
                "exact_match",
                "contains_match",
                "token_f1",
                "semantic_similarity",
                "bleu",
                "rouge_l",
                "metric_disagreement",
                "contains_only_credit",
                "strict_score",
                "reference_shape_warning",
                "review_priority",
            ]
        ].to_dict(orient="records")
    ]
    df["recommended_action"] = df["review_priority"].map(
        {
            "High review priority": "Inspect manually; likely incorrect or missing expected answer.",
            "Near threshold": "Review with stronger semantic metric or judge model.",
            "Contains-only credit": "Review manually; contains-match would over-credit this answer under older scoring.",
            "Metric disagreement": "Check whether lexical metrics under/over-credit a paraphrase.",
            "Reference-shape warning": "Inspect dataset normalization; reference may be a long passage rather than a concise answer.",
            "Looks good": "No immediate review required.",
        }
    )
    return df


def _flag_reason(row: dict[str, Any], pass_threshold: float) -> str:
    if row["score"] < 0.4:
        return f"Composite score {row['score']:.2f} is well below review threshold."
    if row["score"] < pass_threshold:
        return f"Composite score {row['score']:.2f} is below pass threshold {pass_threshold:.2f}."
    if row.get("reference_shape_warning"):
        return str(row["reference_shape_warning"])
    if row["contains_only_credit"]:
        return (
            "Reference appeared in the response, but the profile primary score did not pass; "
            "contains-match is diagnostic only."
        )
    if row["metric_disagreement"]:
        return (
            "Metrics disagree: exact/contains/F1/semantic signals are far apart, "
            "so the case may need semantic or judge review."
        )
    return "No flag triggered."


def _reference_shape_warning(row: pd.Series) -> str:
    metadata = row.get("metadata", {})
    if isinstance(metadata, str):
        metadata = parse_metric_dict(metadata)
    if not isinstance(metadata, dict):
        return ""
    if metadata.get("reference_shape") == "passage_or_long_answer":
        return (
            "Dataset reference is a long passage; concise correct answers may score low without "
            "short-answer extraction or judge review."
        )
    return ""


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
                "Metric": "Semantic Similarity",
                "What it checks": "Cosine similarity using local TF-IDF by default or provider embeddings when enabled.",
                "Best for": "Paraphrase-tolerant comparison where wording differs from the reference.",
                "Limitations": "TF-IDF is not contextual; provider embeddings add cost and dependency.",
            },
            {
                "Metric": "BLEU",
                "What it checks": "N-gram precision with brevity penalty.",
                "Best for": "Translation-like outputs and wording-sensitive generation.",
                "Limitations": "Usually weak as a primary factual QA metric.",
            },
            {
                "Metric": "ROUGE-L",
                "What it checks": "Longest common subsequence overlap with reference text.",
                "Best for": "Long-form answers and summary-style references.",
                "Limitations": "Lexical; can miss correct concise answers when references are long passages.",
            },
            {
                "Metric": "Profile Primary Score",
                "What it checks": "Dataset-specific score selected from the repo-wide scoring profile.",
                "Best for": "Comparing within a dataset/task type.",
                "Limitations": "Do not compare across datasets without checking scoring profile and reference shape.",
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
