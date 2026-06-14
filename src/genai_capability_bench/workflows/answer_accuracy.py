"""Answer accuracy workflow used by the showcase notebook."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from genai_capability_bench.capabilities.registry import get_evaluator
from genai_capability_bench.clients.factory import create_client
from genai_capability_bench.core.runner import load_config, parse_models
from genai_capability_bench.core.schemas import Capability, EvalTask, ModelSpec
from genai_capability_bench.datasets import materialize_dataset
from genai_capability_bench.reporting.diagnostics import add_answer_accuracy_diagnostics
from genai_capability_bench.reporting.executive_summary import generate_summary
from genai_capability_bench.reporting.tables import capability_leaderboard, summarize_results


@dataclass
class AnswerAccuracyRunConfig:
    """Configuration for a multi-dataset answer-accuracy run."""

    repo_root: Path
    model_config_path: Path
    dataset_keys: list[str]
    dataset_splits: dict[str, str | None] = field(default_factory=dict)
    sample_size_per_dataset: int | None = 10
    selected_categories: list[str] | str = "ALL"
    custom_dataset_path: Path | None = None
    random_seed: int = 42
    pass_threshold: float | None = None
    run_id_prefix: str = "answer_accuracy_demo"
    output_root: Path | None = None
    download_if_missing: bool = True
    cache_local_copy: bool = True
    refresh_dataset_cache: bool = False


@dataclass
class AnswerAccuracyRunResult:
    """Artifacts returned by the answer-accuracy workflow."""

    run_id: str
    run_dir: Path
    results_df: pd.DataFrame
    summary_df: pd.DataFrame
    dataset_summary_df: pd.DataFrame
    leaderboard_df: pd.DataFrame
    diagnostics_df: pd.DataFrame
    dataset_manifest_df: pd.DataFrame
    report_path: Path
    summary_text: str


def load_answer_accuracy_tasks(config: AnswerAccuracyRunConfig) -> tuple[list[EvalTask], pd.DataFrame]:
    """Materialize selected datasets and return answer-accuracy tasks plus manifest."""

    all_tasks: list[EvalTask] = []
    manifest_rows: list[dict[str, Any]] = []

    for dataset_key in config.dataset_keys:
        split = config.dataset_splits.get(dataset_key)
        tasks, cache_path = materialize_dataset(
            dataset_key,
            repo_root=config.repo_root,
            split=split,
            sample_size=config.sample_size_per_dataset,
            seed=config.random_seed,
            download_if_missing=config.download_if_missing,
            cache_local_copy=config.cache_local_copy,
            refresh_cache=config.refresh_dataset_cache,
            custom_path=config.custom_dataset_path,
        )

        answer_tasks = [task for task in tasks if task.capability == Capability.ANSWER_ACCURACY]
        for task in answer_tasks:
            task.metadata = {
                **task.metadata,
                "dataset_key": dataset_key,
                "dataset_split": split or "default",
                "dataset_cache_path": str(cache_path) if cache_path else None,
            }

        all_tasks.extend(answer_tasks)
        manifest_rows.append(
            {
                "dataset_key": dataset_key,
                "split": split or "default",
                "cache_or_source": str(cache_path),
                "raw_tasks_loaded": len(tasks),
                "answer_accuracy_tasks": len(answer_tasks),
                "categories": ", ".join(sorted({t.category for t in answer_tasks})),
            }
        )

    if config.selected_categories != "ALL":
        selected = set(config.selected_categories)
        all_tasks = [task for task in all_tasks if task.category in selected]

    return all_tasks, pd.DataFrame(manifest_rows)


def run_answer_accuracy_workflow(config: AnswerAccuracyRunConfig, show_progress: bool = True) -> AnswerAccuracyRunResult:
    """Run answer-accuracy evaluation and write artifacts."""

    model_config = load_config(config.model_config_path)
    models = parse_models(model_config)
    pass_threshold = (
        float(config.pass_threshold)
        if config.pass_threshold is not None
        else float(model_config.get("default_pass_threshold", 0.7))
    )

    tasks, dataset_manifest_df = load_answer_accuracy_tasks(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{config.run_id_prefix}_{timestamp}"
    output_root = config.output_root or config.repo_root / "outputs" / "runs"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for model in models:
        client = create_client(model)
        evaluator = get_evaluator(Capability.ANSWER_ACCURACY, pass_threshold=pass_threshold)
        iterator = tqdm(tasks, desc=f"Evaluating {model.name}", disable=not show_progress)
        for task in iterator:
            result = evaluator.evaluate_task(run_id, task, model, client)
            results.append(result.to_dict())

    results_df = pd.DataFrame(results)
    results_df = _add_dataset_columns(results_df)
    summary_df = summarize_results(results)
    dataset_summary_df = _summarize_by_dataset(results_df)
    leaderboard_df = capability_leaderboard(results)
    diagnostics_df = add_answer_accuracy_diagnostics(results_df, pass_threshold)
    summary_text = generate_summary(summary_df)

    report = _build_markdown_report(
        run_id=run_id,
        models=models,
        tasks=tasks,
        pass_threshold=pass_threshold,
        dataset_manifest_df=dataset_manifest_df,
        dataset_summary_df=dataset_summary_df,
        summary_text=summary_text,
    )
    report_path = run_dir / "answer_accuracy_report.md"

    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    results_df.to_csv(run_dir / "results.csv", index=False)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    dataset_summary_df.to_csv(run_dir / "dataset_summary.csv", index=False)
    leaderboard_df.to_csv(run_dir / "leaderboard.csv", index=False)
    diagnostics_df.to_csv(run_dir / "diagnostics.csv", index=False)
    dataset_manifest_df.to_csv(run_dir / "dataset_manifest.csv", index=False)
    report_path.write_text(report, encoding="utf-8")

    return AnswerAccuracyRunResult(
        run_id=run_id,
        run_dir=run_dir,
        results_df=results_df,
        summary_df=summary_df,
        dataset_summary_df=dataset_summary_df,
        leaderboard_df=leaderboard_df,
        diagnostics_df=diagnostics_df,
        dataset_manifest_df=dataset_manifest_df,
        report_path=report_path,
        summary_text=summary_text,
    )


def _build_markdown_report(
    *,
    run_id: str,
    models: list[ModelSpec],
    tasks: list[EvalTask],
    pass_threshold: float,
    dataset_manifest_df: pd.DataFrame,
    dataset_summary_df: pd.DataFrame,
    summary_text: str,
) -> str:
    model_names = ", ".join(m.name for m in models)
    dataset_lines = "\n".join(
        f"- `{row.dataset_key}` ({row.split}): {row.answer_accuracy_tasks} tasks from `{row.cache_or_source}`"
        for row in dataset_manifest_df.itertuples(index=False)
    )
    dataset_summary_lines = "\n".join(
        f"- `{row.dataset_key}` / `{row.category}`: avg score {row.avg_score:.2f}, pass rate {row.pass_rate:.2f}, n={row.n}"
        for row in dataset_summary_df.itertuples(index=False)
    )
    return f"""# Answer Accuracy Evaluation Report

**Run ID:** `{run_id}`  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Models:** {model_names}  
**Tasks evaluated:** {len(tasks)}  
**Pass threshold:** {pass_threshold:.2f}

## Dataset Manifest

{dataset_lines}

## Executive Summary

{summary_text}

## Dataset-Level Summary

{dataset_summary_lines or 'No dataset-level summary available.'}

## Artifacts

- `results.json`: raw normalized capability results
- `results.csv`: tabular result details
- `summary.csv`: category-level score summary
- `dataset_summary.csv`: dataset/category-level score summary
- `leaderboard.csv`: model/capability leaderboard
- `diagnostics.csv`: review-priority and metric-disagreement diagnostics
- `dataset_manifest.csv`: dataset source/cache record

## Recommended Review

Review high-priority, near-threshold, and metric-disagreement cases before using
these results for model selection, validation, or governance evidence.
"""


def _add_dataset_columns(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty or "metadata" not in results_df.columns:
        return results_df
    df = results_df.copy()
    df["dataset_key"] = df["metadata"].apply(lambda m: (m or {}).get("dataset_key") if isinstance(m, dict) else None)
    df["dataset_split"] = df["metadata"].apply(lambda m: (m or {}).get("dataset_split") if isinstance(m, dict) else None)
    return df


def _summarize_by_dataset(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    group_cols = ["model_name", "dataset_key", "category"]
    summary = (
        results_df.groupby(group_cols, dropna=False)
        .agg(n=("task_id", "count"), avg_score=("score", "mean"), pass_rate=("passed", "mean"))
        .reset_index()
    )
    summary["avg_score"] = summary["avg_score"].round(4)
    summary["pass_rate"] = summary["pass_rate"].round(4)
    return summary
