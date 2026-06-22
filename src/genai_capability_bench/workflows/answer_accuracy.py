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
from genai_capability_bench.core.runner import config_float, load_config, parse_models
from genai_capability_bench.core.schemas import Capability, EvalTask, ModelSpec
from genai_capability_bench.datasets import get_dataset_spec, materialize_dataset
from genai_capability_bench.reporting.diagnostics import add_answer_accuracy_diagnostics
from genai_capability_bench.reporting.executive_summary import generate_summary
from genai_capability_bench.reporting.tables import capability_leaderboard, summarize_results
from genai_capability_bench.metrics.llm_judge import judge_with_rubric


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
    enable_judge_review: bool = False
    judge_model_name: str | None = None
    judge_max_cases: int = 10
    max_tasks_per_run: int | None = 500
    allow_full_public_dataset: bool = False
    checkpoint_every: int = 50
    checkpoint_dir: Path | None = None
    resume_from_checkpoint: Path | None = None


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
    checkpoint_path: Path | None
    summary_text: str
    judge_enabled: bool = False
    reliability_notes: list[str] = field(default_factory=list)


def load_answer_accuracy_tasks(config: AnswerAccuracyRunConfig) -> tuple[list[EvalTask], pd.DataFrame]:
    """Materialize selected datasets and return answer-accuracy tasks plus manifest."""

    all_tasks: list[EvalTask] = []
    manifest_rows: list[dict[str, Any]] = []

    for dataset_key in config.dataset_keys:
        spec = get_dataset_spec(dataset_key)
        if (
            spec.source_type == "huggingface"
            and config.sample_size_per_dataset is None
            and not config.allow_full_public_dataset
        ):
            raise ValueError(
                f"Refusing to load the full public dataset split for '{dataset_key}'. "
                "Set sample_size_per_dataset to a bounded value such as 10, 25, or 100. "
                "If you intentionally want the full split, set allow_full_public_dataset=True."
            )
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
                "scoring_profile": spec.scoring_profile,
                "reference_shape": spec.reference_shape,
                "primary_metrics": ", ".join(spec.primary_metrics),
                "dataset_caveats": spec.caveats,
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
        else config_float(model_config.get("default_pass_threshold"), 0.7)
    )

    tasks, dataset_manifest_df = load_answer_accuracy_tasks(config)
    total_expected = len(tasks) * len(models)
    if config.max_tasks_per_run is not None and total_expected > config.max_tasks_per_run:
        raise ValueError(
            f"Refusing to evaluate {total_expected:,} model calls because max_tasks_per_run="
            f"{config.max_tasks_per_run:,}. Reduce SAMPLE_SIZE_PER_DATASET / DATASET_KEYS, "
            "or intentionally raise max_tasks_per_run for a planned large benchmark."
        )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{config.run_id_prefix}_{timestamp}"
    output_root = config.output_root or config.repo_root / "outputs" / "runs"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.checkpoint_dir or output_root / "_checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "results_checkpoint.jsonl"
    checkpoint_state_path = checkpoint_dir / "checkpoint_state.json"
    resumed_results = _load_checkpoint_results(config.resume_from_checkpoint)
    completed_keys = {_result_checkpoint_key(row) for row in resumed_results}
    completed_keys.discard("")
    if resumed_results:
        _write_checkpoint_snapshot(
            checkpoint_path=checkpoint_path,
            checkpoint_state_path=checkpoint_state_path,
            results=resumed_results,
            run_id=run_id,
            total_expected=len(tasks) * len(models),
            completed=len(completed_keys),
            dataset_manifest_df=dataset_manifest_df,
            source_checkpoint=config.resume_from_checkpoint,
        )

    results: list[dict[str, Any]] = [_with_run_id(row, run_id) for row in resumed_results]
    newly_completed = 0
    checkpoint_buffer: list[dict[str, Any]] = []
    for model in models:
        client = create_client(model)
        evaluator = get_evaluator(Capability.ANSWER_ACCURACY, pass_threshold=pass_threshold)
        pending_tasks = [task for task in tasks if _task_checkpoint_key(model, task) not in completed_keys]
        iterator = tqdm(
            pending_tasks,
            desc=f"Evaluating {model.name}",
            disable=not show_progress,
            initial=total_expected - len(pending_tasks) if len(models) == 1 else 0,
            total=total_expected if len(models) == 1 else len(pending_tasks),
            unit="sample",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%] elapsed={elapsed} remaining={remaining}",
        )
        for task in iterator:
            result = evaluator.evaluate_task(run_id, task, model, client)
            row = result.to_dict()
            results.append(row)
            checkpoint_buffer.append(row)
            completed_keys.add(_result_checkpoint_key(row))
            newly_completed += 1
            if newly_completed % max(config.checkpoint_every, 1) == 0:
                _append_checkpoint_rows(checkpoint_path, checkpoint_buffer)
                checkpoint_buffer = []
                _write_checkpoint_state(
                    checkpoint_path=checkpoint_path,
                    checkpoint_state_path=checkpoint_state_path,
                    run_id=run_id,
                    total_expected=total_expected,
                    completed=len(completed_keys),
                    dataset_manifest_df=dataset_manifest_df,
                    source_checkpoint=config.resume_from_checkpoint,
                )
                iterator.set_postfix_str(f"checkpoint saved: {len(completed_keys)}/{total_expected}")

    if checkpoint_buffer:
        _append_checkpoint_rows(checkpoint_path, checkpoint_buffer)
    _write_checkpoint_state(
        checkpoint_path=checkpoint_path,
        checkpoint_state_path=checkpoint_state_path,
        run_id=run_id,
        total_expected=total_expected,
        completed=len(completed_keys),
        dataset_manifest_df=dataset_manifest_df,
        source_checkpoint=config.resume_from_checkpoint,
    )

    results_df = pd.DataFrame(results)
    results_df = _add_dataset_columns(results_df)
    summary_df = summarize_results(results)
    dataset_summary_df = _summarize_by_dataset(results_df)
    leaderboard_df = capability_leaderboard(results)
    diagnostics_df = add_answer_accuracy_diagnostics(results_df, pass_threshold)
    if config.enable_judge_review:
        diagnostics_df = _add_judge_review(
            diagnostics_df,
            model_config=model_config,
            judge_model_name=config.judge_model_name,
            max_cases=config.judge_max_cases,
        )
    reliability_notes = _build_reliability_notes(
        results_df=results_df,
        diagnostics_df=diagnostics_df,
        dataset_manifest_df=dataset_manifest_df,
    )
    summary_text = generate_summary(summary_df)
    if reliability_notes:
        summary_text = summary_text + " " + " ".join(reliability_notes)

    report = _build_markdown_report(
        run_id=run_id,
        models=models,
        tasks=tasks,
        pass_threshold=pass_threshold,
        dataset_manifest_df=dataset_manifest_df,
        dataset_summary_df=dataset_summary_df,
        reliability_notes=reliability_notes,
        checkpoint_path=checkpoint_path,
        summary_text=summary_text,
        judge_enabled=config.enable_judge_review,
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
        checkpoint_path=checkpoint_path,
        summary_text=summary_text,
        judge_enabled=config.enable_judge_review,
        reliability_notes=reliability_notes,
    )


def _build_markdown_report(
    *,
    run_id: str,
    models: list[ModelSpec],
    tasks: list[EvalTask],
    pass_threshold: float,
    dataset_manifest_df: pd.DataFrame,
    dataset_summary_df: pd.DataFrame,
    reliability_notes: list[str],
    checkpoint_path: Path | None,
    summary_text: str,
    judge_enabled: bool,
) -> str:
    model_names = ", ".join(m.name for m in models)
    dataset_lines = "\n".join(
        f"- `{row.dataset_key}` ({row.split}): {row.answer_accuracy_tasks} tasks, "
        f"profile `{getattr(row, 'scoring_profile', 'unknown')}`, "
        f"reference shape `{getattr(row, 'reference_shape', 'unknown')}` from `{row.cache_or_source}`"
        for row in dataset_manifest_df.itertuples(index=False)
    )
    dataset_summary_lines = "\n".join(
        f"- `{row.dataset_key}` / `{row.category}`: avg score {row.avg_score:.2f}, pass rate {row.pass_rate:.2f}, n={row.n}"
        for row in dataset_summary_df.itertuples(index=False)
    )
    reliability_lines = "\n".join(f"- {note}" for note in reliability_notes)
    return f"""# Answer Accuracy Evaluation Report

**Run ID:** `{run_id}`  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Models:** {model_names}  
**Tasks evaluated:** {len(tasks)}  
**Pass threshold:** {pass_threshold:.2f}
**Judge review:** {"Enabled" if judge_enabled else "Disabled"}
**Checkpoint:** `{checkpoint_path}`

## Dataset Manifest

{dataset_lines}

## Executive Summary

{summary_text}

## Benchmark Reliability Notes

{reliability_lines or '- No reliability notes triggered.'}

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
- `{checkpoint_path}`: resumable JSONL checkpoint for interrupted long runs

## Recommended Review

Review high-priority, near-threshold, and metric-disagreement cases before using
these results for model selection, validation, or governance evidence.
"""


def _checkpoint_results_file(path: Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    if path.is_dir():
        return path / "results_checkpoint.jsonl"
    return path


def _load_checkpoint_results(path: Path | None) -> list[dict[str, Any]]:
    """Load checkpoint rows from a JSONL, JSON, CSV, or checkpoint directory."""

    checkpoint_file = _checkpoint_results_file(path)
    if checkpoint_file is None or not checkpoint_file.exists():
        return []

    if checkpoint_file.suffix.lower() == ".jsonl":
        rows = []
        for line in checkpoint_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    if checkpoint_file.suffix.lower() == ".json":
        rows = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        return rows if isinstance(rows, list) else rows.get("results", [])

    if checkpoint_file.suffix.lower() == ".csv":
        return pd.read_csv(checkpoint_file).to_dict(orient="records")

    raise ValueError(f"Unsupported checkpoint file type: {checkpoint_file.suffix}")


def _write_checkpoint_snapshot(
    *,
    checkpoint_path: Path,
    checkpoint_state_path: Path,
    results: list[dict[str, Any]],
    run_id: str,
    total_expected: int,
    completed: int,
    dataset_manifest_df: pd.DataFrame,
    source_checkpoint: Path | None,
) -> None:
    """Write a resumable checkpoint snapshot for long notebook runs."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, default=str) + "\n")
    _write_checkpoint_state(
        checkpoint_path=checkpoint_path,
        checkpoint_state_path=checkpoint_state_path,
        run_id=run_id,
        total_expected=total_expected,
        completed=completed,
        dataset_manifest_df=dataset_manifest_df,
        source_checkpoint=source_checkpoint,
    )


def _append_checkpoint_rows(checkpoint_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def _write_checkpoint_state(
    *,
    checkpoint_path: Path,
    checkpoint_state_path: Path,
    run_id: str,
    total_expected: int,
    completed: int,
    dataset_manifest_df: pd.DataFrame,
    source_checkpoint: Path | None,
) -> None:
    progress = completed / total_expected if total_expected else 0.0
    state = {
        "run_id": run_id,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": str(checkpoint_path),
        "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
        "completed": completed,
        "total_expected": total_expected,
        "progress_pct": round(progress * 100, 2),
        "dataset_manifest": dataset_manifest_df.to_dict(orient="records"),
    }
    checkpoint_state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def _with_run_id(row: dict[str, Any], run_id: str) -> dict[str, Any]:
    out = dict(row)
    out["run_id"] = run_id
    return out


def _task_checkpoint_key(model: ModelSpec, task: EvalTask) -> str:
    dataset_key = task.metadata.get("dataset_key", "") if isinstance(task.metadata, dict) else ""
    return f"{model.name}::{dataset_key}::{task.task_id}"


def _result_checkpoint_key(row: dict[str, Any]) -> str:
    model_name = str(row.get("model_name", ""))
    task_id = str(row.get("task_id", ""))
    dataset_key = str(row.get("dataset_key") or "")
    if not dataset_key:
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        if isinstance(metadata, dict):
            dataset_key = str(metadata.get("dataset_key", ""))
    if not model_name or not task_id:
        return ""
    return f"{model_name}::{dataset_key}::{task_id}"


def _build_reliability_notes(
    *,
    results_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    dataset_manifest_df: pd.DataFrame,
) -> list[str]:
    """Generate transparent caveats about evidence strength for a run."""

    notes: list[str] = []
    n = len(results_df)
    dataset_keys = set(dataset_manifest_df.get("dataset_key", pd.Series(dtype=str)).dropna().astype(str))
    avg_score = float(results_df["score"].mean()) if n else 0.0
    pass_rate = float(results_df["passed"].mean()) if n and "passed" in results_df else 0.0

    if n < 30:
        notes.append(
            f"Only {n} tasks were evaluated; treat this as a smoke/demo run, not statistically meaningful evidence."
        )
    if dataset_keys == {"answer_accuracy_sample"}:
        notes.append(
            "The run used the local `answer_accuracy_sample` smoke set; use public benchmarks or a larger custom golden set before drawing model-quality conclusions."
        )
    if n and avg_score >= 0.99 and pass_rate >= 0.99 and n < 100:
        notes.append(
            "The run achieved near-perfect scores on fewer than 100 tasks; this should trigger benchmark-hardness review rather than be read as broad model mastery."
        )
    if "contains_only_credit" in diagnostics_df and bool(diagnostics_df["contains_only_credit"].any()):
        count = int(diagnostics_df["contains_only_credit"].sum())
        notes.append(
            f"{count} case(s) would be over-credited by contains-match; review these for answer correctness."
        )
    if "reference_shape" in dataset_manifest_df and (dataset_manifest_df["reference_shape"] == "passage_or_long_answer").any():
        datasets = ", ".join(
            sorted(dataset_manifest_df.loc[dataset_manifest_df["reference_shape"] == "passage_or_long_answer", "dataset_key"])
        )
        notes.append(
            f"{datasets} uses long passage-style references; concise answers may need short-answer extraction or judge review."
        )

    return notes


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


def _add_judge_review(
    diagnostics_df: pd.DataFrame,
    *,
    model_config: dict[str, Any],
    judge_model_name: str | None,
    max_cases: int,
) -> pd.DataFrame:
    """Run optional LLM judge review for flagged cases only."""

    if diagnostics_df.empty:
        return diagnostics_df

    df = diagnostics_df.copy()
    df["judge_score"] = None
    df["judge_reason"] = None

    flagged_idx = df[df["flagged"]].head(max_cases).index
    if len(flagged_idx) == 0:
        return df

    model_row = dict(model_config.get("models", [{}])[0])
    judge_model = judge_model_name or model_config.get("judge_model") or model_row.get("model")
    if not judge_model:
        df.loc[flagged_idx, "judge_reason"] = "Judge review skipped: no judge model configured."
        return df

    judge_spec = ModelSpec(
        name=f"Judge-{judge_model}",
        provider=model_row.get("provider", "openai_compatible"),
        model=judge_model,
        api_version=model_row.get("api_version"),
        temperature=0.0,
        max_tokens=500,
        metadata=model_row.get("metadata", {}),
    )
    client = create_client(judge_spec)
    rubric = (
        "Score whether the answer correctly answers the question given the reference. "
        "Use 1.0 for fully correct, 0.5 for partially correct, and 0.0 for incorrect. "
        "Penalize unsupported extra claims."
    )

    for idx in flagged_idx:
        row = df.loc[idx]
        reference = str(row.get("expected_output") or "")
        task = f"Question: {row.get('input_text')}"
        answer = str(row.get("actual_output") or "")
        try:
            judge = judge_with_rubric(client, task=task, answer=answer, reference=reference, rubric=rubric)
            df.at[idx, "judge_score"] = judge.score
            df.at[idx, "judge_reason"] = judge.reason
        except Exception as exc:
            df.at[idx, "judge_reason"] = f"Judge review failed: {exc}"
    return df
