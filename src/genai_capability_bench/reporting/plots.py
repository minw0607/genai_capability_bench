"""Plot helpers for notebooks and reports."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_capability_scores(summary_df: pd.DataFrame, title: str = "Capability Scores"):
    """Create a compact bar chart from a summary dataframe."""

    if summary_df.empty:
        raise ValueError("summary_df is empty")
    labels = summary_df["capability"] + " / " + summary_df["category"]
    fig, ax = plt.subplots(figsize=(10, max(4, len(summary_df) * 0.35)))
    ax.barh(labels, summary_df["avg_score"], color="#3b82f6")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Average score")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig, ax

