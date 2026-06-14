"""Tabular summaries for capability results."""

from __future__ import annotations

import pandas as pd


def results_dataframe(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results)


def summarize_results(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby(["model_name", "capability", "category"], dropna=False)
        .agg(
            n=("task_id", "count"),
            avg_score=("score", "mean"),
            pass_rate=("passed", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
        )
        .reset_index()
    )
    grouped["avg_score"] = grouped["avg_score"].round(4)
    grouped["pass_rate"] = grouped["pass_rate"].round(4)
    grouped["avg_latency_ms"] = grouped["avg_latency_ms"].round(1)
    return grouped


def capability_leaderboard(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    if df.empty:
        return pd.DataFrame()
    board = (
        df.groupby(["model_name", "capability"], dropna=False)
        .agg(n=("task_id", "count"), avg_score=("score", "mean"), pass_rate=("passed", "mean"))
        .reset_index()
    )
    return board.sort_values(["capability", "avg_score"], ascending=[True, False])

