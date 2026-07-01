"""Capability and measurement rating helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RatingAssessment:
    """Separated model-performance and measurement-quality assessment."""

    capability_rating: str
    capability_rationale: str
    reliability_rating: str
    reliability_rationale: str
    review_posture: str
    review_rationale: str


def assess_answer_accuracy_run(
    *,
    results_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    dataset_manifest_df: pd.DataFrame,
) -> RatingAssessment:
    """Assess capability separately from measurement reliability."""

    total = len(results_df)
    avg_score = float(results_df["score"].mean()) if total and "score" in results_df else 0.0
    pass_rate = float(results_df["passed"].mean()) if total and "passed" in results_df else 0.0
    flagged = int(diagnostics_df["flagged"].sum()) if "flagged" in diagnostics_df else 0
    flag_rate = flagged / total if total else 0.0
    judge_rescues = judge_rescue_count(diagnostics_df)
    judge_failures = judge_failure_count(diagnostics_df)
    reference_warning = has_reference_shape_warning(dataset_manifest_df)
    disagreement_rate = _metric_disagreement_rate(diagnostics_df)

    if pass_rate >= 0.85 and avg_score >= 0.80:
        capability = "Strong"
    elif pass_rate >= 0.70 and avg_score >= 0.65:
        capability = "Moderate-Strong"
    elif pass_rate >= 0.50 and avg_score >= 0.50:
        capability = "Mixed"
    else:
        capability = "Weak"

    reliability_points = 0
    reliability_reasons: list[str] = []
    if total < 50:
        reliability_points += 1
        reliability_reasons.append(f"small sample size ({total} responses)")
    if reference_warning:
        reliability_points += 2
        reliability_reasons.append("one or more datasets use long/context-shaped references")
    if flag_rate > 0.40:
        reliability_points += 1
        reliability_reasons.append(f"high diagnostic review rate ({flag_rate:.0%})")
    if disagreement_rate > 0.25:
        reliability_points += 1
        reliability_reasons.append(f"high metric-disagreement rate ({disagreement_rate:.0%})")
    if judge_rescues:
        rescue_rate = judge_rescues / total if total else 0.0
        reliability_points += 2 if rescue_rate >= 0.05 else 1
        reliability_reasons.append(f"judge identified {judge_rescues} likely deterministic false negative(s)")
    if judge_failures:
        reliability_points += 1
        reliability_reasons.append(f"judge review failed for {judge_failures} case(s)")

    if reliability_points >= 3:
        reliability = "Low"
    elif reliability_points >= 1:
        reliability = "Medium"
    else:
        reliability = "High"

    if reliability == "Low":
        review_posture = "Interpret With Caution"
    elif capability in {"Weak", "Mixed"} or reliability == "Medium":
        review_posture = "Targeted Review Recommended"
    else:
        review_posture = "Directionally Reliable"

    capability_rationale = (
        f"pass rate {pass_rate:.0%}; average profile score {avg_score:.2f}; "
        f"{total} response(s) evaluated."
    )
    reliability_rationale = "; ".join(reliability_reasons) if reliability_reasons else "no major measurement caveats triggered"
    review_rationale = (
        f"flagged review rate {flag_rate:.0%}; judge rescues {judge_rescues}; judge failures {judge_failures}."
    )
    return RatingAssessment(
        capability_rating=capability,
        capability_rationale=capability_rationale,
        reliability_rating=reliability,
        reliability_rationale=reliability_rationale + ".",
        review_posture=review_posture,
        review_rationale=review_rationale,
    )


def dataset_rollup(dataset_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate category-level dataset summary into dataset-level portfolio rows."""

    if dataset_summary_df.empty:
        return pd.DataFrame(columns=["model_name", "dataset_key", "n", "avg_score", "pass_rate", "categories"])
    weighted = dataset_summary_df.copy()
    weighted["weighted_score"] = weighted["avg_score"] * weighted["n"]
    weighted["weighted_pass"] = weighted["pass_rate"] * weighted["n"]
    group_cols = ["model_name", "dataset_key"] if "model_name" in weighted else ["dataset_key"]
    rollup = (
        weighted.groupby(group_cols, dropna=False)
        .agg(
            n=("n", "sum"),
            weighted_score=("weighted_score", "sum"),
            weighted_pass=("weighted_pass", "sum"),
            categories=("category", "nunique"),
        )
        .reset_index()
    )
    rollup["avg_score"] = (rollup["weighted_score"] / rollup["n"]).round(4)
    rollup["pass_rate"] = (rollup["weighted_pass"] / rollup["n"]).round(4)
    return rollup.drop(columns=["weighted_score", "weighted_pass"]).sort_values(
        ["pass_rate", "avg_score"], ascending=False
    )


def judge_rescue_count(diagnostics_df: pd.DataFrame) -> int:
    if "judge_score" not in diagnostics_df or "passed" not in diagnostics_df:
        return 0
    judge_scores = pd.to_numeric(diagnostics_df["judge_score"], errors="coerce")
    return int(((judge_scores >= 0.7) & (~diagnostics_df["passed"].astype(bool))).sum())


def judge_failure_count(diagnostics_df: pd.DataFrame) -> int:
    if "judge_score" not in diagnostics_df or "judge_reason" not in diagnostics_df:
        return 0
    judge_scores = pd.to_numeric(diagnostics_df["judge_score"], errors="coerce")
    reasons = diagnostics_df["judge_reason"].fillna("").astype(str)
    return int((judge_scores.isna() & reasons.str.startswith("Judge review failed:")).sum())


def has_reference_shape_warning(dataset_manifest_df: pd.DataFrame) -> bool:
    return (
        "reference_shape" in dataset_manifest_df
        and (dataset_manifest_df["reference_shape"].astype(str) == "passage_or_long_answer").any()
    )


def _metric_disagreement_rate(diagnostics_df: pd.DataFrame) -> float:
    if diagnostics_df.empty or "metric_disagreement" not in diagnostics_df:
        return 0.0
    return float(diagnostics_df["metric_disagreement"].fillna(False).astype(bool).mean())

