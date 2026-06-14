"""Deterministic executive summaries."""

from __future__ import annotations

import pandas as pd


def generate_summary(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "No results were available for summary."

    overall = summary_df["avg_score"].mean()
    best_score = summary_df["avg_score"].max()
    weakest_score = summary_df["avg_score"].min()
    best_rows = summary_df[summary_df["avg_score"] == best_score]
    weakest_rows = summary_df[summary_df["avg_score"] == weakest_score]

    best = best_rows.iloc[0]
    weakest = weakest_rows.iloc[0]
    if best_score == weakest_score:
        return (
            f"Overall average capability score was {overall:.2f}. "
            f"All evaluated slices tied at {best_score:.2f}, so no relative strength or weakness "
            "was detected in this run. Treat this as a workflow validation result if the target "
            "model is a mock model or if the sample size is small."
        )

    return (
        f"Overall average capability score was {overall:.2f}. "
        f"The strongest observed slice was {best['capability']} / {best['category']} "
        f"for {best['model_name']} with an average score of {best['avg_score']:.2f}. "
        f"The weakest observed slice was {weakest['capability']} / {weakest['category']} "
        f"for {weakest['model_name']} with an average score of {weakest['avg_score']:.2f}. "
        "Review low-scoring examples before drawing production conclusions."
    )
