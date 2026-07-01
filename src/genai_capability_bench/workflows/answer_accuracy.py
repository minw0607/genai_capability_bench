"""Answer accuracy workflow used by the showcase notebook."""

from __future__ import annotations

import json
import hashlib
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from genai_capability_bench.capabilities.registry import get_evaluator
from genai_capability_bench.clients.factory import create_client
from genai_capability_bench.core.runner import (
    config_float,
    config_optional_float,
    config_optional_int,
    config_optional_str,
    load_config,
    parse_models,
)
from genai_capability_bench.core.schemas import Capability, EvalTask, ModelSpec
from genai_capability_bench.datasets import get_dataset_spec, materialize_dataset
from genai_capability_bench.reporting.diagnostics import add_answer_accuracy_diagnostics
from genai_capability_bench.reporting.executive_summary import generate_summary
from genai_capability_bench.reporting.model_labels import models_public_label
from genai_capability_bench.reporting.ratings import assess_answer_accuracy_run, dataset_rollup
from genai_capability_bench.reporting.tables import capability_leaderboard, summarize_results
from genai_capability_bench.metrics.llm_judge import judge_with_rubric


ANSWER_ACCURACY_METHOD_VERSION = "answer_accuracy_profile_metrics_v5_20260629"


@dataclass
class AnswerAccuracyRunConfig:
    """Configuration for a multi-dataset answer-accuracy run."""

    repo_root: Path
    model_config_path: Path
    dataset_keys: list[str]
    dataset_splits: dict[str, str | None] = field(default_factory=dict)
    sample_size_per_dataset: int | None = 10
    sample_strategy: str | None = None
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
    auto_resume_from_latest: bool = True
    cleanup_incompatible_checkpoints: bool = True
    semantic_similarity_mode: str = "tfidf"


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
    resumed_from_checkpoint: Path | None
    artifact_paths: dict[str, Path]
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
            sample_strategy=config.sample_strategy,
            download_if_missing=config.download_if_missing,
            cache_local_copy=config.cache_local_copy,
            refresh_cache=config.refresh_dataset_cache,
            custom_path=config.custom_dataset_path,
        )

        answer_tasks = [task for task in tasks if task.capability == Capability.ANSWER_ACCURACY]
        split_label = split or spec.default_split
        sample_strategy = config.sample_strategy or spec.default_sample_strategy
        for task in answer_tasks:
            task.metadata = {
                **task.metadata,
                "dataset_key": dataset_key,
                "dataset_split": split_label,
                "sample_strategy": sample_strategy,
                "dataset_cache_path": str(cache_path) if cache_path else None,
            }

        all_tasks.extend(answer_tasks)
        source_fingerprint = _source_fingerprint(cache_path)
        manifest_rows.append(
            {
                "dataset_key": dataset_key,
                "split": split_label,
                "cache_or_source": str(cache_path),
                "source_fingerprint": source_fingerprint,
                "sample_strategy": sample_strategy,
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


def _source_fingerprint(path: Path | None) -> str:
    """Return a compact fingerprint for checkpoint compatibility checks."""

    if path is None or not Path(path).exists():
        return ""
    source = Path(path)
    stat = source.stat()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()[:16]}|bytes:{stat.st_size}"


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
    checkpoint_root = output_root / "_checkpoints"
    checkpoint_fingerprint = _build_checkpoint_fingerprint(
        config=config,
        models=models,
        pass_threshold=pass_threshold,
        dataset_manifest_df=dataset_manifest_df,
    )
    if config.cleanup_incompatible_checkpoints:
        _cleanup_incompatible_checkpoints(checkpoint_root, checkpoint_fingerprint)

    resume_source = config.resume_from_checkpoint
    if resume_source is not None and not _checkpoint_is_compatible(resume_source, checkpoint_fingerprint):
        raise ValueError(
            "The requested checkpoint is not compatible with the current dataset/model/scoring configuration. "
            "Re-run the evaluation or choose a checkpoint created with the same method version and benchmark settings."
        )
    if resume_source is None and config.auto_resume_from_latest:
        resume_source = _find_latest_compatible_checkpoint(checkpoint_root, checkpoint_fingerprint)

    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.checkpoint_dir or checkpoint_root / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "results_checkpoint.jsonl"
    checkpoint_state_path = checkpoint_dir / "checkpoint_state.json"
    resumed_results = _load_checkpoint_results(resume_source)
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
            source_checkpoint=resume_source,
            checkpoint_fingerprint=checkpoint_fingerprint,
        )

    results: list[dict[str, Any]] = [_with_run_id(row, run_id) for row in resumed_results]
    newly_completed = 0
    checkpoint_buffer: list[dict[str, Any]] = []
    for model in models:
        client = create_client(model)
        evaluator = get_evaluator(
            Capability.ANSWER_ACCURACY,
            pass_threshold=pass_threshold,
            semantic_similarity_mode=config.semantic_similarity_mode,
        )
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
                    source_checkpoint=resume_source,
                    checkpoint_fingerprint=checkpoint_fingerprint,
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
        source_checkpoint=resume_source,
        checkpoint_fingerprint=checkpoint_fingerprint,
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

    artifact_stamp = timestamp
    report_path = run_dir / f"answer_accuracy_technical_report_{artifact_stamp}.md"
    html_report_path = run_dir / f"answer_accuracy_executive_summary_{artifact_stamp}.html"
    artifact_paths = {
        "raw_results_json": run_dir / f"answer_accuracy_raw_results_{artifact_stamp}.json",
        "raw_results_csv": run_dir / f"answer_accuracy_raw_results_{artifact_stamp}.csv",
        "category_summary_csv": run_dir / f"answer_accuracy_category_summary_{artifact_stamp}.csv",
        "dataset_summary_csv": run_dir / f"answer_accuracy_dataset_summary_{artifact_stamp}.csv",
        "leaderboard_csv": run_dir / f"answer_accuracy_model_leaderboard_{artifact_stamp}.csv",
        "diagnostics_csv": run_dir / f"answer_accuracy_diagnostics_{artifact_stamp}.csv",
        "dataset_manifest_csv": run_dir / f"answer_accuracy_dataset_manifest_{artifact_stamp}.csv",
        "executive_summary_html": html_report_path,
        "technical_report_md": report_path,
        "checkpoint_jsonl": checkpoint_path,
    }

    report = _build_markdown_report(
        run_id=run_id,
        models=models,
        tasks=tasks,
        pass_threshold=pass_threshold,
        results_df=results_df,
        summary_df=summary_df,
        dataset_manifest_df=dataset_manifest_df,
        dataset_summary_df=dataset_summary_df,
        diagnostics_df=diagnostics_df,
        reliability_notes=reliability_notes,
        checkpoint_path=checkpoint_path,
        resumed_from_checkpoint=resume_source,
        summary_text=summary_text,
        judge_enabled=config.enable_judge_review,
        artifact_paths=artifact_paths,
    )

    artifact_paths["raw_results_json"].write_text(json.dumps(results, indent=2), encoding="utf-8")
    results_df.to_csv(artifact_paths["raw_results_csv"], index=False)
    summary_df.to_csv(artifact_paths["category_summary_csv"], index=False)
    dataset_summary_df.to_csv(artifact_paths["dataset_summary_csv"], index=False)
    leaderboard_df.to_csv(artifact_paths["leaderboard_csv"], index=False)
    diagnostics_df.to_csv(artifact_paths["diagnostics_csv"], index=False)
    dataset_manifest_df.to_csv(artifact_paths["dataset_manifest_csv"], index=False)
    report_path.write_text(report, encoding="utf-8")
    html_report_path.write_text(
        _build_html_report_artifact(
            run_id=run_id,
            models=models,
            run_dir=run_dir,
            results_df=results_df,
            summary_df=summary_df,
            dataset_summary_df=dataset_summary_df,
            leaderboard_df=leaderboard_df,
            diagnostics_df=diagnostics_df,
            dataset_manifest_df=dataset_manifest_df,
            report_path=report_path,
            checkpoint_path=checkpoint_path,
            resumed_from_checkpoint=resume_source,
            artifact_paths=artifact_paths,
            summary_text=summary_text,
            judge_enabled=config.enable_judge_review,
            reliability_notes=reliability_notes,
        ),
        encoding="utf-8",
    )

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
        resumed_from_checkpoint=resume_source,
        artifact_paths=artifact_paths,
        summary_text=summary_text,
        judge_enabled=config.enable_judge_review,
        reliability_notes=reliability_notes,
    )


def _build_html_report_artifact(
    *,
    run_id: str,
    models: list[ModelSpec],
    run_dir: Path,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    dataset_summary_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    dataset_manifest_df: pd.DataFrame,
    report_path: Path,
    checkpoint_path: Path | None,
    resumed_from_checkpoint: Path | None,
    artifact_paths: dict[str, Path | None],
    summary_text: str,
    judge_enabled: bool,
    reliability_notes: list[str],
) -> str:
    from genai_capability_bench.reporting.notebook_views import html_summary_report

    run_view = SimpleNamespace(
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
        resumed_from_checkpoint=resumed_from_checkpoint,
        artifact_paths=artifact_paths,
        summary_text=summary_text,
        judge_enabled=judge_enabled,
        reliability_notes=reliability_notes,
    )
    body = html_summary_report(run_view, target_label=models_public_label(models), judge_enabled=judge_enabled)
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Answer Accuracy Executive Summary</title>"
        "<style>body{margin:24px;background:#f5f5f5;}</style></head><body>"
        f"{body}</body></html>"
    )


def _build_markdown_report(
    *,
    run_id: str,
    models: list[ModelSpec],
    tasks: list[EvalTask],
    pass_threshold: float,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    dataset_manifest_df: pd.DataFrame,
    dataset_summary_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    reliability_notes: list[str],
    checkpoint_path: Path | None,
    resumed_from_checkpoint: Path | None,
    summary_text: str,
    judge_enabled: bool,
    artifact_paths: dict[str, Path | None],
) -> str:
    public_model_label = models_public_label(models)
    total = len(results_df)
    avg_score = float(results_df["score"].mean()) if total else 0.0
    pass_rate = float(results_df["passed"].mean()) if total and "passed" in results_df else 0.0
    flagged = int(diagnostics_df["flagged"].sum()) if "flagged" in diagnostics_df else 0
    flag_rate = flagged / total if total else 0.0
    assessment = assess_answer_accuracy_run(
        results_df=results_df,
        diagnostics_df=diagnostics_df,
        dataset_manifest_df=dataset_manifest_df,
    )
    dataset_lines = "\n".join(
        f"- `{row.dataset_key}` ({row.split}): {row.answer_accuracy_tasks} tasks, "
        f"profile `{getattr(row, 'scoring_profile', 'unknown')}`, "
        f"reference shape `{getattr(row, 'reference_shape', 'unknown')}`"
        for row in dataset_manifest_df.itertuples(index=False)
    )
    dataset_summary_lines = "\n".join(
        f"- `{row.dataset_key}` / `{row.category}`: avg score {row.avg_score:.2f}, pass rate {row.pass_rate:.2f}, n={row.n}"
        for row in dataset_summary_df.itertuples(index=False)
    )
    dataset_portfolio_lines = "\n".join(
        f"- `{row.dataset_key}`: avg score {row.avg_score:.2f}, pass rate {row.pass_rate:.0%}, "
        f"n={row.n}, categories={row.categories}"
        for row in dataset_rollup(dataset_summary_df).itertuples(index=False)
    )
    approach_lines = _evaluation_approach_lines(dataset_manifest_df, pass_threshold)
    finding_lines = _finding_lines(
        results_df=results_df,
        diagnostics_df=diagnostics_df,
        dataset_summary_df=dataset_summary_df,
        pass_threshold=pass_threshold,
    )
    reliability_lines = "\n".join(f"- {note}" for note in reliability_notes)
    artifact_lines = _markdown_artifact_table(artifact_paths)
    public_summary_text = _sanitize_model_names(summary_text, models)
    provenance = (
        f"Checkpoint replay from previous compatible run `{_checkpoint_run_label(resumed_from_checkpoint)}`; no new target-model responses were required."
        if resumed_from_checkpoint
        else "Fresh target-model evaluation; no prior checkpoint was used."
    )
    active_checkpoint = _checkpoint_run_label(checkpoint_path) if checkpoint_path else "None"
    conclusion = _report_conclusion(
        capability_rating=assessment.capability_rating,
        reliability_rating=assessment.reliability_rating,
        pass_rate=pass_rate,
        flagged=flagged,
        total=total,
        reliability_notes=reliability_notes,
    )
    return f"""# Answer Accuracy Evaluation Memo

| Field | Value |
|---|---|
| Run ID | `{run_id}` |
| Generated | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
| Target model | {public_model_label} |
| Evaluation scope | Answer accuracy capability |
| Responses evaluated | {len(tasks)} |
| Pass threshold | {pass_threshold:.2f} |
| Judge review | {"Enabled" if judge_enabled else "Disabled"} |
| Run provenance | {provenance} |
| Active checkpoint | `{active_checkpoint}` |

## 1. Management Summary

This memo evaluates whether the target LLM produced answers that align with accepted benchmark references across the selected closed-book knowledge benchmark. The purpose is not to produce a single leaderboard number in isolation; it is to identify capability signal, scoring limitations, and review priorities that should guide the next evaluation iteration.

**Capability rating:** **{assessment.capability_rating}**  
**Capability basis:** {assessment.capability_rationale}

**Evaluation reliability:** **{assessment.reliability_rating}**  
**Reliability basis:** {assessment.reliability_rationale}

**Review posture:** **{assessment.review_posture}**  
**Posture basis:** {assessment.review_rationale}

| Measure | Value |
|---|---:|
| Total responses | {total} |
| Average deterministic score | {avg_score:.2f} |
| Pass rate | {pass_rate:.0%} |
| Flagged for review | {flagged} ({flag_rate:.0%}) |

**Interpretation:** Capability and reliability are intentionally separated. A weak or mixed capability rating suggests model-performance concern. A low reliability rating suggests the result may be driven by benchmark design, scoring fit, sample size, or adjudication issues and should not be treated as a final model-quality conclusion without review.

## 2. Scope and Evidence Base

{dataset_lines or '- No dataset manifest available.'}

## 3. Evaluation Methodology

{approach_lines}

The primary score is dataset-profile specific. Exact match, token F1, ROUGE-L, BLEU, and semantic similarity are retained as supporting evidence signals. The pass/fail decision uses the profile primary score against the configured threshold. LLM judge review, when enabled, is used as a targeted adjudication aid for flagged cases; it is not treated as an automatic replacement for deterministic scoring.

## 4. Key Findings

{finding_lines}

## 5. Dataset Portfolio Performance

{dataset_portfolio_lines or '- No dataset portfolio summary available.'}

## 6. Category-Level Performance

{dataset_summary_lines or '- No dataset-level summary available.'}

## 7. Reliability and Calibration Notes

{reliability_lines or '- No reliability notes triggered.'}

## 8. Conclusion and Recommended Action

{conclusion}

Supporting deterministic summary: {public_summary_text}

## 9. Saved Artifacts

{artifact_lines}

## 10. Recommended Review

Review high-priority, near-threshold, metric-disagreement, and reference-shape-warning cases before using these results for model selection, validation, or governance evidence.
"""


def _overall_evaluation_risk(
    *,
    avg_score: float,
    pass_rate: float,
    flag_rate: float,
    diagnostics_df: pd.DataFrame,
    dataset_manifest_df: pd.DataFrame,
) -> tuple[str, str]:
    has_reference_warning = (
        "reference_shape" in dataset_manifest_df
        and (dataset_manifest_df["reference_shape"] == "passage_or_long_answer").any()
    )
    judge_rescues = _judge_rescue_count(diagnostics_df)
    judge_failures = _judge_failure_count(diagnostics_df)
    if pass_rate < 0.60 or flag_rate > 0.40 or has_reference_warning:
        rating = "High"
    elif pass_rate < 0.80 or avg_score < 0.75 or flag_rate > 0.20 or judge_rescues > 0 or judge_failures > 0:
        rating = "Medium"
    else:
        rating = "Low"

    reasons = [
        f"pass rate {pass_rate:.0%}",
        f"average score {avg_score:.2f}",
        f"review flag rate {flag_rate:.0%}",
    ]
    if has_reference_warning:
        reasons.append("one or more datasets use long passage-style references")
    if judge_rescues:
        reasons.append(f"judge review marked {judge_rescues} deterministic failures as likely correct")
    if judge_failures:
        reasons.append(f"judge review failed for {judge_failures} attempted case(s)")
    return rating, "; ".join(reasons) + "."


def _markdown_artifact_table(artifact_paths: dict[str, Path | None]) -> str:
    rows = [
        ("Executive reporting", "Styled HTML report for leadership review", "executive_summary_html"),
        ("Technical reporting", "Markdown memo for source control and text review", "technical_report_md"),
        ("Raw outputs", "Per-question model responses and normalized scores", "raw_results_csv"),
        ("Diagnostics", "Flag reasons, metric details, and review-priority fields", "diagnostics_csv"),
        ("Dataset performance", "Dataset/category-level score and pass-rate summary", "dataset_summary_csv"),
        ("Dataset manifest", "Dataset source, split, reference shape, and scoring profile", "dataset_manifest_csv"),
        ("Checkpoint", "Resumable JSONL checkpoint for interrupted or replayed runs", "checkpoint_jsonl"),
    ]
    lines = ["| Category | Purpose | File |", "|---|---|---|"]
    for category, purpose, key in rows:
        path = artifact_paths.get(key)
        if path is not None:
            lines.append(f"| {category} | {purpose} | `{path.name}` |")
    return "\n".join(lines)


def _checkpoint_run_label(path: Path | None) -> str:
    if path is None:
        return "None"
    parts = Path(path).parts
    return parts[-2] if len(parts) >= 2 else Path(path).name


def _sanitize_model_names(text: str, models: list[ModelSpec]) -> str:
    sanitized = text
    for model in models:
        public = models_public_label([model])
        for raw in {model.name, model.model}:
            if raw:
                sanitized = sanitized.replace(str(raw), public)
    return sanitized


def _evaluation_approach_lines(dataset_manifest_df: pd.DataFrame, pass_threshold: float) -> str:
    lines = [f"- Pass threshold: `{pass_threshold:.2f}` on the profile primary score."]
    for row in dataset_manifest_df.itertuples(index=False):
        lines.append(
            f"- `{row.dataset_key}` uses scoring profile `{getattr(row, 'scoring_profile', 'unknown')}` "
            f"with primary metrics `{getattr(row, 'primary_metrics', 'unknown')}` and reference shape "
            f"`{getattr(row, 'reference_shape', 'unknown')}`."
        )
    return "\n".join(lines)


def _finding_lines(
    *,
    results_df: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    dataset_summary_df: pd.DataFrame,
    pass_threshold: float,
) -> str:
    if results_df.empty:
        return "- No observations available because no results were produced."

    findings: list[str] = []
    strongest = dataset_summary_df.sort_values("avg_score", ascending=False).head(1)
    weakest = dataset_summary_df.sort_values("avg_score", ascending=True).head(1)
    if not strongest.empty:
        row = strongest.iloc[0]
        findings.append(
            f"- Strongest slice: `{row['dataset_key']}` / `{row['category']}` with average score "
            f"{row['avg_score']:.2f} and pass rate {row['pass_rate']:.0%}."
        )
    if not weakest.empty:
        row = weakest.iloc[0]
        findings.append(
            f"- Weakest slice: `{row['dataset_key']}` / `{row['category']}` with average score "
            f"{row['avg_score']:.2f} and pass rate {row['pass_rate']:.0%}."
        )
    if "review_priority" in diagnostics_df:
        counts = diagnostics_df["review_priority"].value_counts()
        priority_text = ", ".join(f"{label}: {count}" for label, count in counts.items())
        findings.append(f"- Diagnostic review distribution: {priority_text}.")
    if "contains_only_credit" in diagnostics_df and bool(diagnostics_df["contains_only_credit"].any()):
        findings.append(
            f"- {int(diagnostics_df['contains_only_credit'].sum())} case(s) contain a reference string but still fail "
            "the profile score, indicating potential over-credit risk from simple contains-match rules."
        )
    judge_reviewed = _judge_reviewed_count(diagnostics_df)
    judge_rescues = _judge_rescue_count(diagnostics_df)
    judge_failures = _judge_failure_count(diagnostics_df)
    if judge_reviewed:
        findings.append(
            f"- LLM judge reviewed {judge_reviewed} flagged case(s); {judge_rescues} deterministic failure(s) "
            "were judged likely correct and should be inspected as possible metric false negatives."
        )
    if judge_failures:
        findings.append(
            f"- LLM judge was attempted for {judge_failures} flagged case(s), but no valid judge score was returned. "
            "Check provider parameter compatibility and judge configuration before relying on judge-assisted findings."
        )
    below_threshold = int((results_df["score"] < pass_threshold).sum())
    findings.append(f"- {below_threshold} response(s) scored below the configured pass threshold of {pass_threshold:.2f}.")
    return "\n".join(findings)


def _judge_reviewed_count(diagnostics_df: pd.DataFrame) -> int:
    if "judge_score" not in diagnostics_df:
        return 0
    return int(diagnostics_df["judge_score"].notna().sum())


def _judge_rescue_count(diagnostics_df: pd.DataFrame) -> int:
    if "judge_score" not in diagnostics_df or "passed" not in diagnostics_df:
        return 0
    judge_scores = pd.to_numeric(diagnostics_df["judge_score"], errors="coerce")
    return int(((judge_scores >= 0.7) & (~diagnostics_df["passed"].astype(bool))).sum())


def _judge_failure_count(diagnostics_df: pd.DataFrame) -> int:
    if "judge_score" not in diagnostics_df or "judge_reason" not in diagnostics_df:
        return 0
    judge_scores = pd.to_numeric(diagnostics_df["judge_score"], errors="coerce")
    reasons = diagnostics_df["judge_reason"].fillna("").astype(str)
    return int((judge_scores.isna() & reasons.str.startswith("Judge review failed:")).sum())


def _report_conclusion(
    *,
    capability_rating: str,
    reliability_rating: str,
    pass_rate: float,
    flagged: int,
    total: int,
    reliability_notes: list[str],
) -> str:
    if reliability_rating == "Low":
        notes = " ".join(reliability_notes)
        return (
            f"The run should be interpreted cautiously. The capability signal is {capability_rating.lower()}, "
            f"but evaluation reliability is low; pass rate is {pass_rate:.0%}, with {flagged} of {total} cases "
            f"flagged for review. {notes} Resolve measurement caveats before drawing final capability conclusions."
        )
    if capability_rating in {"Strong", "Moderate-Strong"} and reliability_rating == "High":
        return (
            "The benchmark evidence is directionally strong for this scope. Continue with broader category coverage, "
            "larger samples, and regression tracking before treating the result as production-grade evidence."
        )
    if capability_rating in {"Mixed", "Weak"} or reliability_rating == "Medium":
        return (
            f"The run shows {capability_rating.lower()} answer-accuracy evidence with {reliability_rating.lower()} "
            f"measurement reliability: pass rate is {pass_rate:.0%}, with {flagged} of {total} "
            "cases requiring review. Use the diagnostics table and optional judge review to separate model errors "
            "from metric/reference limitations before making a model-selection decision."
        )
    return (
        f"The run provides {capability_rating.lower()} capability evidence with {reliability_rating.lower()} "
        "measurement reliability. Review flagged examples and expand coverage before production reliance."
    )


def _checkpoint_results_file(path: Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    if path.is_dir():
        return path / "results_checkpoint.jsonl"
    return path


def _checkpoint_state_file(path: Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path)
    if path.is_dir():
        return path / "checkpoint_state.json"
    return path.parent / "checkpoint_state.json"


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


def _build_checkpoint_fingerprint(
    *,
    config: AnswerAccuracyRunConfig,
    models: list[ModelSpec],
    pass_threshold: float,
    dataset_manifest_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build a compatibility fingerprint for checkpoint reuse."""

    selected_categories = (
        config.selected_categories
        if isinstance(config.selected_categories, str)
        else sorted(str(x) for x in config.selected_categories)
    )
    payload = {
        "method_version": ANSWER_ACCURACY_METHOD_VERSION,
        "model_config_path": str(Path(config.model_config_path).resolve()),
        "models": [
            {
                "name": model.name,
                "provider": model.provider,
                "model": model.model,
                "api_version": model.api_version,
            }
            for model in models
        ],
        "dataset_keys": list(config.dataset_keys),
        "dataset_splits": dict(sorted(config.dataset_splits.items())),
        "sample_size_per_dataset": config.sample_size_per_dataset,
        "sample_strategy": config.sample_strategy,
        "selected_categories": selected_categories,
        "custom_dataset_path": str(config.custom_dataset_path) if config.custom_dataset_path else None,
        "random_seed": config.random_seed,
        "pass_threshold": pass_threshold,
        "semantic_similarity_mode": config.semantic_similarity_mode,
        "dataset_manifest": dataset_manifest_df.fillna("").to_dict(orient="records"),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return {
        "method_version": ANSWER_ACCURACY_METHOD_VERSION,
        "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "payload": payload,
    }


def _load_checkpoint_state(path: Path | None) -> dict[str, Any]:
    state_file = _checkpoint_state_file(path)
    if state_file is None or not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _checkpoint_is_compatible(path: Path | None, checkpoint_fingerprint: dict[str, Any]) -> bool:
    state = _load_checkpoint_state(path)
    return (
        state.get("method_version") == checkpoint_fingerprint["method_version"]
        and state.get("benchmark_fingerprint") == checkpoint_fingerprint["fingerprint"]
    )


def _find_latest_compatible_checkpoint(
    checkpoint_root: Path,
    checkpoint_fingerprint: dict[str, Any],
) -> Path | None:
    if not checkpoint_root.exists():
        return None
    candidates = []
    for state_file in checkpoint_root.glob("*/checkpoint_state.json"):
        state = _load_checkpoint_state(state_file)
        if (
            state.get("method_version") == checkpoint_fingerprint["method_version"]
            and state.get("benchmark_fingerprint") == checkpoint_fingerprint["fingerprint"]
        ):
            checkpoint_path = Path(state.get("checkpoint_path", state_file.parent / "results_checkpoint.jsonl"))
            if checkpoint_path.exists():
                candidates.append((state_file.stat().st_mtime, checkpoint_path))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _cleanup_incompatible_checkpoints(checkpoint_root: Path, checkpoint_fingerprint: dict[str, Any]) -> None:
    if not checkpoint_root.exists():
        return
    for checkpoint_dir in checkpoint_root.iterdir():
        if not checkpoint_dir.is_dir():
            continue
        state_file = checkpoint_dir / "checkpoint_state.json"
        state = _load_checkpoint_state(state_file)
        if not state:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            continue
        if state.get("method_version") != checkpoint_fingerprint["method_version"]:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)


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
    checkpoint_fingerprint: dict[str, Any],
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
        checkpoint_fingerprint=checkpoint_fingerprint,
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
    checkpoint_fingerprint: dict[str, Any],
) -> None:
    progress = completed / total_expected if total_expected else 0.0
    state = {
        "run_id": run_id,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_path": str(checkpoint_path),
        "source_checkpoint": str(source_checkpoint) if source_checkpoint else None,
        "method_version": checkpoint_fingerprint["method_version"],
        "benchmark_fingerprint": checkpoint_fingerprint["fingerprint"],
        "benchmark_fingerprint_payload": checkpoint_fingerprint["payload"],
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
        temperature=config_optional_float(model_row.get("temperature"), None),
        max_tokens=config_optional_int(model_row.get("max_tokens"), None),
        token_parameter=config_optional_str(model_row.get("token_parameter"), None),
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
