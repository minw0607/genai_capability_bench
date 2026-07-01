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


def plot_metric_heatmap(
    diagnostics_df: pd.DataFrame,
    *,
    group_col: str = "dataset_key",
    metric_cols: list[str] | None = None,
    title: str = "Metric Profile by Dataset",
):
    """Plot average metric values by dataset/category as a heatmap."""

    if diagnostics_df.empty:
        raise ValueError("diagnostics_df is empty")
    metric_cols = metric_cols or [
        "primary_score",
        "exact_match",
        "token_f1",
        "semantic_similarity",
        "rouge_l",
        "bleu",
        "contains_match",
    ]
    available = [col for col in metric_cols if col in diagnostics_df.columns]
    if not available:
        raise ValueError("No requested metric columns are available")

    plot_df = diagnostics_df.copy()
    if group_col not in plot_df.columns:
        group_col = "category"
    matrix = plot_df.groupby(group_col, dropna=False)[available].mean().sort_index()

    fig, ax = plt.subplots(figsize=(max(8, len(available) * 1.1), max(3.5, len(matrix) * 0.75)))
    image = ax.imshow(matrix.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([_pretty_metric_name(col) for col in available], rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index.astype(str))
    ax.set_title(title, loc="left", fontweight="bold")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix.iloc[y, x]
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=9, color="#111827")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Average metric value")
    fig.tight_layout()
    return fig, ax


def plot_metric_bars_by_group(
    diagnostics_df: pd.DataFrame,
    *,
    group_col: str = "dataset_key",
    metric_cols: list[str] | None = None,
    title: str = "Metric Comparison by Dataset",
):
    """Plot grouped bars for selected metrics by dataset/category."""

    if diagnostics_df.empty:
        raise ValueError("diagnostics_df is empty")
    metric_cols = metric_cols or ["primary_score", "token_f1", "semantic_similarity", "rouge_l"]
    available = [col for col in metric_cols if col in diagnostics_df.columns]
    if not available:
        raise ValueError("No requested metric columns are available")
    if group_col not in diagnostics_df.columns:
        group_col = "category"

    matrix = diagnostics_df.groupby(group_col, dropna=False)[available].mean().sort_index()
    fig, ax = plt.subplots(figsize=(max(9, len(matrix) * 1.4), 4.8))
    x = range(len(matrix.index))
    width = 0.8 / len(available)
    colors = ["#2563eb", "#0f766e", "#9333ea", "#ea580c", "#64748b"]
    for idx, metric in enumerate(available):
        positions = [v - 0.4 + width / 2 + idx * width for v in x]
        ax.bar(positions, matrix[metric], width=width, label=_pretty_metric_name(metric), color=colors[idx % len(colors)])
    ax.set_xticks(list(x))
    ax.set_xticklabels(matrix.index.astype(str), rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Average metric value")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=min(4, len(available)), frameon=False, bbox_to_anchor=(0, 1.02), loc="lower left")
    fig.tight_layout()
    return fig, ax


def plot_score_distribution_by_group(
    results_df: pd.DataFrame,
    *,
    group_col: str = "dataset_key",
    title: str = "Score Distribution by Dataset",
):
    """Plot score distributions split by dataset/category."""

    if results_df.empty:
        raise ValueError("results_df is empty")
    if group_col not in results_df.columns:
        group_col = "category"
    groups = [(str(k), g["score"].dropna()) for k, g in results_df.groupby(group_col, dropna=False)]
    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.8), 4.8))
    ax.boxplot([values for _, values in groups], labels=[label for label, _ in groups], patch_artist=True)
    for patch in ax.artists:
        patch.set_facecolor("#dbeafe")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def _pretty_metric_name(metric: str) -> str:
    return {
        "primary_score": "Primary",
        "exact_match": "Exact",
        "contains_match": "Contains",
        "token_f1": "Token F1",
        "semantic_similarity": "Semantic",
        "tfidf_similarity": "TF-IDF",
        "bleu": "BLEU",
        "rouge_1": "ROUGE-1",
        "rouge_2": "ROUGE-2",
        "rouge_l": "ROUGE-L",
    }.get(metric, metric.replace("_", " ").title())
