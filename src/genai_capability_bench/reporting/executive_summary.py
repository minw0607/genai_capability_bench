"""Deterministic executive summaries."""

from __future__ import annotations

import pandas as pd


def generate_summary(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "No results were available for summary."

    overall = summary_df["avg_score"].mean()
    best = summary_df.sort_values("avg_score", ascending=False).iloc[0]
    weakest = summary_df.sort_values("avg_score", ascending=True).iloc[0]

    return (
        f"Overall average capability score was {overall:.2f}. "
        f"The strongest observed slice was {best['capability']} / {best['category']} "
        f"for {best['model_name']} with an average score of {best['avg_score']:.2f}. "
        f"The weakest observed slice was {weakest['capability']} / {weakest['category']} "
        f"for {weakest['model_name']} with an average score of {weakest['avg_score']:.2f}. "
        "Review low-scoring examples before drawing production conclusions."
    )

