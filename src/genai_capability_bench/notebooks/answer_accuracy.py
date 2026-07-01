"""Notebook helpers for the answer-accuracy demo.

These helpers keep the showcase notebook focused on decisions and interpretation
while reusable workflow, display, and artifact-preview logic lives in Python.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, Markdown, display

from genai_capability_bench.core.runner import config_float, load_config, parse_models
from genai_capability_bench.reporting.diagnostics import metric_guide_table, score_interpretation_table
from genai_capability_bench.reporting.model_labels import models_public_label
from genai_capability_bench.reporting.notebook_views import (
    dataset_catalog_table,
    dataset_preset_table,
    embedding_config_table,
    html_summary_report,
    judge_config_table,
    metric_standards_display_table,
    model_config_table,
    provider_environment_table,
    sample_size_guidance_table,
    scoring_profiles_display_table,
    selected_dataset_plan_table,
)
from genai_capability_bench.reporting.plots import (
    plot_capability_scores,
    plot_metric_bars_by_group,
    plot_metric_heatmap,
    plot_score_distribution_by_group,
)
from genai_capability_bench.reporting.ratings import dataset_rollup
from genai_capability_bench.workflows.answer_accuracy import (
    AnswerAccuracyRunConfig,
    load_answer_accuracy_tasks,
)


ANSWER_ACCURACY_DATASET_PRESETS = {
    "local_smoke": ["answer_accuracy_sample"],
    "curated_knowledge": ["curated_knowledge_v1"],
    "knowledge_portfolio": ["mmlu", "triviaqa", "natural_questions", "arc"],
    "broad_knowledge": ["mmlu"],
    "open_domain_qa": ["triviaqa", "natural_questions"],
    "science_exam": ["arc"],
    "custom_golden_set": ["custom"],
}

RELATED_CAPABILITY_PRESETS = {
    "context_grounded_rag": ["squad"],
    "multi_hop_rag_reasoning": ["hotpotqa"],
    "truthfulness_misconceptions": ["truthfulqa"],
}

CONTEXT_REQUIRED_DATASETS = {"squad", "hotpotqa"}

RUN_PROFILES = {
    "quick_check": {
        "sample_size_per_dataset": 10,
        "max_tasks_per_run": 50,
        "allow_full_public_dataset": False,
        "guidance": "Credential, dataset, scoring, and artifact path check.",
    },
    "exploratory": {
        "sample_size_per_dataset": 100,
        "max_tasks_per_run": 500,
        "allow_full_public_dataset": False,
        "guidance": "Recommended first public benchmark run.",
    },
    "report_sample": {
        "sample_size_per_dataset": 500,
        "max_tasks_per_run": 2500,
        "allow_full_public_dataset": False,
        "guidance": "Larger sampled run for a stronger report artifact.",
    },
    "full_public": {
        "sample_size_per_dataset": None,
        "max_tasks_per_run": None,
        "allow_full_public_dataset": True,
        "guidance": "Full selected public split. Use only with budget and runtime approval.",
    },
}

ARTIFACT_CATALOG = [
    ("Executive reporting", "Styled HTML report for leadership review", "executive_summary_html"),
    ("Technical reporting", "Markdown memo for source control and text review", "technical_report_md"),
    ("Raw outputs", "Per-question model responses and normalized scores as JSON", "raw_results_json"),
    ("Raw outputs", "Per-question model responses and normalized scores", "raw_results_csv"),
    ("Category performance", "Category-level score and pass-rate summary", "category_summary_csv"),
    ("Diagnostics", "Flag reasons, metric details, and review-priority fields", "diagnostics_csv"),
    ("Dataset performance", "Dataset/category-level score and pass-rate summary", "dataset_summary_csv"),
    ("Model comparison", "Model/capability leaderboard", "leaderboard_csv"),
    ("Dataset manifest", "Dataset source, split, reference shape, and scoring profile", "dataset_manifest_csv"),
    ("Checkpoint", "Resumable JSONL checkpoint for interrupted or replayed runs", "checkpoint_jsonl"),
]

DEFAULT_METRIC_COLUMNS = [
    "primary_score",
    "exact_match",
    "token_f1",
    "semantic_similarity",
    "rouge_l",
    "bleu",
    "contains_match",
]


@dataclass
class AnswerAccuracyNotebookPlan:
    """Prepared notebook configuration and display context."""

    model_config: dict[str, Any]
    models: list[Any]
    public_model_label: str
    workflow_config: AnswerAccuracyRunConfig
    dataset_preset: str
    dataset_keys: list[str]
    run_mode: str
    profile: dict[str, Any]
    sample_label: str
    planned_calls: str | int
    safety_cap_label: str
    sample_strategy_label: str
    semantic_similarity_label: str


def display_notebook_ready(repo_root: Path) -> None:
    """Display a compact environment-ready callout."""

    display(
        Markdown(
            "✅ **Notebook environment ready**  \n"
            f"Repository root: `{repo_root}`"
        )
    )


def display_reference_catalogs() -> None:
    """Display the benchmark catalog, sampling guidance, and metric standards."""

    _callout("Dataset presets", "Benchmark portfolios available to this notebook and adjacent capability notebooks.")
    display(dataset_preset_table())

    _callout("Sample-size guidance", "Use small runs to validate mechanics, then scale once scoring behavior looks calibrated.")
    display(sample_size_guidance_table())

    _callout(
        "Closed-book knowledge QA inventory",
        "Use curated_knowledge as the recommended default; use source benchmarks when you want original benchmark reporting.",
    )
    catalog = dataset_catalog_table()
    display(catalog[~catalog["Dataset Key"].isin(CONTEXT_REQUIRED_DATASETS | {"truthfulqa"})].reset_index(drop=True))

    _callout("Metric standards", "These repo-wide definitions keep scoring language consistent across future capability notebooks.")
    display(metric_standards_display_table())
    display(scoring_profiles_display_table())
    display(score_interpretation_table())

    display(
        Markdown(
            "> **Default dataset note:** `curated_knowledge` is the recommended default for this notebook. "
            "It uses the repo-native `curated_knowledge_v1` JSONL benchmark built from all available compatible "
            "closed-book public benchmark caches. The curation process preserves source questions, expected "
            "answers, and references, while adding broad categories and provenance metadata. Run modes still "
            "sample from the curated file so users do not accidentally launch a full benchmark.\n\n"
            "> **Multi-dataset note:** Use `knowledge_portfolio` or set `DATASET_KEYS` manually to run several "
            "closed-book datasets in one report. Results are evaluated with dataset-specific scoring profiles, "
            "then summarized as a portfolio rather than treated as one interchangeable score.\n\n"
            "> **Routing note:** SQuAD and HotpotQA are intentionally excluded from active Answer Accuracy "
            "presets because they require supplied context or multi-hop evidence. They belong in RAG/context "
            "or reasoning workflows. TruthfulQA belongs in the Truthfulness notebook because it requires "
            "correct-vs-incorrect reference handling."
        )
    )


def display_curated_dataset_preview(repo_root: Path) -> None:
    """Display the curated answer-accuracy dataset manifest when available."""

    manifest_path = repo_root / "datasets" / "curated" / "answer_accuracy_knowledge_v1_manifest.json"
    if not manifest_path.exists():
        _callout(
            "Curated dataset",
            "Curated Knowledge QA v1 has not been generated yet. Run scripts/build_curated_answer_accuracy.py first.",
        )
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _callout(
        "Curated dataset preview",
        f"{manifest.get('display_name', 'Curated dataset')} contains {manifest.get('row_count', 0)} normalized rows.",
    )
    display(
        Markdown(
            "**How this curated dataset is constructed:** source datasets are first normalized into the shared "
            "`EvalTask` schema, then compatible closed-book answer-accuracy rows are copied into a versioned "
            "local JSONL asset. The curation layer preserves source questions, expected answers, and references; "
            "it adds only standardized taxonomy and provenance metadata. Notebook runs sample from this local "
            "asset, with stratified sampling as the default for the curated benchmark."
        )
    )
    if manifest.get("selection_policy"):
        display(Markdown(f"**Selection policy:** {manifest['selection_policy']}"))
    display(
        pd.DataFrame(
            [
                {"Source": key, "Rows": value}
                for key, value in (manifest.get("source_counts") or {}).items()
            ]
        )
    )
    display(
        pd.DataFrame(
            [
                {"Curated Category": key, "Rows": value}
                for key, value in (manifest.get("category_counts") or {}).items()
            ]
        ).sort_values(["Rows", "Curated Category"], ascending=[False, True]).reset_index(drop=True)
    )
    decisions = manifest.get("source_decisions") or {}
    if decisions:
        display(
            pd.DataFrame(
                [{"Source": key, "Curation Decision": value} for key, value in decisions.items()]
            )
        )
    compatible_columns = manifest.get("compatible_columns") or []
    if compatible_columns:
        display(
            pd.DataFrame(
                [{"Curated Schema Field": column} for column in compatible_columns]
            )
        )
    display(Markdown(f"**Integrity policy:** {manifest.get('integrity_policy', '')}"))


def prepare_answer_accuracy_plan(
    *,
    repo_root: Path,
    output_root: Path,
    model_config_path: Path,
    dataset_preset: str,
    run_mode: str,
    dataset_keys: list[str] | None = None,
    dataset_splits: dict[str, str | None] | None = None,
    custom_dataset_path: Path | None = None,
    selected_categories: list[str] | str = "ALL",
    sample_strategy: str | None = None,
    random_seed: int = 42,
    checkpoint_every: int = 50,
    checkpoint_dir: Path | None = None,
    resume_from_checkpoint: Path | None = None,
    auto_resume_from_latest: bool = True,
    cleanup_incompatible_checkpoints: bool = True,
    download_if_missing: bool = True,
    cache_local_copy: bool = True,
    refresh_dataset_cache: bool = False,
    pass_threshold: float | None = None,
    run_id_prefix: str = "answer_accuracy_demo",
    enable_judge_review: bool = True,
    judge_model_name: str | None = None,
    judge_max_cases: int = 10,
    use_api_embeddings: bool = False,
) -> AnswerAccuracyNotebookPlan:
    """Build the workflow config and display context from notebook choices."""

    if dataset_preset not in ANSWER_ACCURACY_DATASET_PRESETS:
        if dataset_preset in RELATED_CAPABILITY_PRESETS:
            routed = ", ".join(RELATED_CAPABILITY_PRESETS[dataset_preset])
            raise ValueError(
                f"DATASET_PRESET='{dataset_preset}' is a routed suite preset ({routed}), not an active "
                "closed-book Answer Accuracy preset. Use the Truthfulness, RAG/context-grounded, or "
                "Reasoning notebook for that dataset, or choose one of the active Answer Accuracy presets: "
                f"{', '.join(sorted(ANSWER_ACCURACY_DATASET_PRESETS))}."
            )
        valid = ", ".join(sorted(ANSWER_ACCURACY_DATASET_PRESETS))
        raise ValueError(f"Unknown DATASET_PRESET='{dataset_preset}'. Choose one of: {valid}.")
    if run_mode not in RUN_PROFILES:
        valid = ", ".join(sorted(RUN_PROFILES))
        raise ValueError(f"Unknown RUN_MODE='{run_mode}'. Choose one of: {valid}.")

    profile = RUN_PROFILES[run_mode]
    active_dataset_keys = dataset_keys or ANSWER_ACCURACY_DATASET_PRESETS[dataset_preset]
    invalid_context_keys = sorted(set(active_dataset_keys) & CONTEXT_REQUIRED_DATASETS)
    if invalid_context_keys:
        invalid = ", ".join(invalid_context_keys)
        raise ValueError(
            f"{invalid} requires supplied context and is excluded from this closed-book knowledge Answer Accuracy notebook. "
            "Route it to the RAG/context-grounded or reasoning notebook instead."
        )
    sample_size_per_dataset = profile["sample_size_per_dataset"]
    max_tasks_per_run = profile["max_tasks_per_run"]
    allow_full_public_dataset = profile["allow_full_public_dataset"]

    model_config = load_config(model_config_path)
    models = parse_models(model_config)
    public_model_label = models_public_label(models)

    planned_calls: str | int
    if sample_size_per_dataset is None:
        planned_calls = "Full selected split"
    else:
        planned_calls = sample_size_per_dataset * len(active_dataset_keys) * len(models)
    sample_label = "Full selected split" if sample_size_per_dataset is None else f"{sample_size_per_dataset} per dataset"
    safety_cap_label = "No cap" if max_tasks_per_run is None else f"{max_tasks_per_run} model calls"
    resolved_sample_strategy = None if sample_strategy in {None, "auto"} else sample_strategy
    strategy_label = (
        "Auto: curated datasets use source/category stratified sampling"
        if resolved_sample_strategy is None
        else resolved_sample_strategy
    )
    semantic_similarity_mode = "api_embeddings_if_configured" if use_api_embeddings else "tfidf"
    semantic_similarity_label = (
        "API embeddings if configured, otherwise TF-IDF fallback"
        if use_api_embeddings
        else "Local TF-IDF cosine similarity"
    )

    workflow_config = AnswerAccuracyRunConfig(
        repo_root=repo_root,
        model_config_path=model_config_path,
        dataset_keys=active_dataset_keys,
        dataset_splits=dataset_splits or {},
        sample_size_per_dataset=sample_size_per_dataset,
        sample_strategy=resolved_sample_strategy,
        selected_categories=selected_categories,
        custom_dataset_path=custom_dataset_path,
        random_seed=random_seed,
        pass_threshold=pass_threshold,
        run_id_prefix=run_id_prefix,
        output_root=output_root,
        download_if_missing=download_if_missing,
        cache_local_copy=cache_local_copy,
        refresh_dataset_cache=refresh_dataset_cache,
        enable_judge_review=enable_judge_review,
        judge_model_name=judge_model_name,
        judge_max_cases=judge_max_cases,
        max_tasks_per_run=max_tasks_per_run,
        allow_full_public_dataset=allow_full_public_dataset,
        checkpoint_every=checkpoint_every,
        checkpoint_dir=checkpoint_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        auto_resume_from_latest=auto_resume_from_latest,
        cleanup_incompatible_checkpoints=cleanup_incompatible_checkpoints,
        semantic_similarity_mode=semantic_similarity_mode,
    )

    return AnswerAccuracyNotebookPlan(
        model_config=model_config,
        models=models,
        public_model_label=public_model_label,
        workflow_config=workflow_config,
        dataset_preset=dataset_preset,
        dataset_keys=active_dataset_keys,
        run_mode=run_mode,
        profile=profile,
        sample_label=sample_label,
        planned_calls=planned_calls,
        safety_cap_label=safety_cap_label,
        sample_strategy_label=strategy_label,
        semantic_similarity_label=semantic_similarity_label,
    )


def display_configuration_summary(plan: AnswerAccuracyNotebookPlan) -> None:
    """Display the active benchmark plan in reader-friendly form."""

    config = plan.workflow_config
    _callout("Configuration ready", "Review this plan before running target-model calls.")
    display(
        pd.DataFrame(
            [
                {
                    "Dataset Preset": plan.dataset_preset,
                    "Datasets Selected": ", ".join(plan.dataset_keys),
                    "Run Mode": plan.run_mode,
                    "Run Mode Guidance": plan.profile["guidance"],
                    "Requested Sample": plan.sample_label,
                    "Sampling Strategy": plan.sample_strategy_label,
                    "Estimated Model Calls": plan.planned_calls,
                    "Safety Cap": plan.safety_cap_label,
                    "Semantic Similarity": plan.semantic_similarity_label,
                    "Checkpoint Every": config.checkpoint_every,
                    "Auto Resume": "Enabled" if config.auto_resume_from_latest else "Disabled",
                    "Judge Review": "Enabled" if config.enable_judge_review else "Disabled",
                }
            ]
        )
    )

    display(
        Markdown(
            "**How to interpret the two run-size controls:** `Requested Sample` is the per-dataset sampling request. "
            "`Safety Cap` is a hard stop on total model calls after datasets and models are combined. "
            "For `curated_knowledge`, the default sampling strategy is stratified by source dataset first, then broad category."
        )
    )

    _callout("Active dataset plan", "The evaluator loads these datasets, normalizes their schemas, and applies the registered scoring profile.")
    display(
        selected_dataset_plan_table(
            plan.dataset_keys,
            preset_name=plan.dataset_preset,
            sample_size_per_dataset=config.sample_size_per_dataset,
            sample_strategy=config.sample_strategy,
            dataset_splits=config.dataset_splits,
            selected_categories=config.selected_categories,
        )
    )

    _callout("Routed suite datasets", "Visible for roadmap continuity, but intentionally not active in this closed-book Answer Accuracy workflow.")
    display(
        pd.DataFrame(
            [
                {"Preset": preset, "Dataset Keys": ", ".join(keys), "Primary Notebook": "Truthfulness"}
                for preset, keys in RELATED_CAPABILITY_PRESETS.items()
            ]
        )
    )


def display_model_context(plan: AnswerAccuracyNotebookPlan) -> None:
    """Display target, judge, provider, and embedding context without secrets."""

    _callout("Target model", "Human-facing reports use the public label below; raw deployments remain in machine-readable artifacts.")
    display(model_config_table(plan.models))
    display(provider_environment_table())

    _callout(
        "Judge and embedding context",
        "Judge review supports flagged-case adjudication; API embeddings can replace the default TF-IDF semantic metric when enabled.",
    )
    display(judge_config_table())
    display(embedding_config_table())

    if any(model.provider == "mock" for model in plan.models):
        display(Markdown("> **Mode:** This run uses `DemoMock`. It validates the workflow but is not real LLM evidence."))
    else:
        display(Markdown(f"> **Target LLM confirmed:** **{plan.public_model_label}**."))

    if plan.workflow_config.enable_judge_review:
        display(Markdown("> **Judge review enabled:** flagged cases may receive targeted LLM adjudication."))
    else:
        display(Markdown("> **Judge review disabled:** deterministic metrics will still flag cases for optional review."))


def display_preflight_check(plan: AnswerAccuracyNotebookPlan) -> tuple[list[Any], pd.DataFrame]:
    """Load tasks once and present a concise go/no-go table."""

    config = plan.workflow_config
    tasks, dataset_manifest_df = load_answer_accuracy_tasks(config)
    preflight_df = pd.DataFrame(
        [
            _check_row("Model config exists", config.model_config_path.exists(), config.model_config_path.name),
            _check_row("Target model configured", len(plan.models) > 0, plan.public_model_label),
            _check_row("Dataset preset selected", bool(plan.dataset_preset), plan.dataset_preset),
            _check_row("Dataset keys selected", len(plan.dataset_keys) > 0, ", ".join(plan.dataset_keys)),
            _check_row("Answer-accuracy tasks available", len(tasks) > 0, f"{len(tasks):,} normalized task(s)"),
            _check_row("Output directory available", config.output_root.exists() or config.output_root.parent.exists(), str(config.output_root)),
        ]
    )

    _callout("Pre-flight check", "No target-model calls have been made yet. This confirms the benchmark is runnable.")
    display(preflight_df)

    _callout("Loaded dataset manifest", "This is the actual evaluation scope after normalization and capability filtering.")
    display(_clean_manifest_for_display(dataset_manifest_df))

    category_df = pd.DataFrame({"category": [task.category for task in tasks]}).value_counts().reset_index(name="n")
    _callout("Category distribution", "A quick coverage view of the selected benchmark sample.")
    display(category_df)

    if not all(preflight_df["Ready"]):
        raise RuntimeError("Pre-flight failed. Fix the review items before continuing.")

    display(Markdown("✅ **Pre-flight passed.** The next cell will run or resume model evaluation."))
    return tasks, dataset_manifest_df


def display_run_completion(run, plan: AnswerAccuracyNotebookPlan, *, max_preview_rows: int = 10) -> pd.DataFrame:
    """Display run completion status, artifact catalog, and a compact result preview."""

    artifact_table = artifact_display_table(run)
    resumed = _checkpoint_label(getattr(run, "resumed_from_checkpoint", None))
    display(
        Markdown(
            "✅ **Evaluation complete.**  \n"
            f"Run ID: `{run.run_id}`  \n"
            f"Saved to: `{run.run_dir}`  \n"
            f"Checkpoint: `{_checkpoint_label(run.checkpoint_path)}`  \n"
            f"Resumed from: `{resumed}`"
        )
    )
    display(artifact_table)

    results_preview = _sanitize_model_columns(
        run.results_df[
            [
                "model_name",
                "dataset_key",
                "task_id",
                "category",
                "actual_output",
                "expected_output",
                "score",
                "passed",
            ]
        ].head(max_preview_rows),
        plan.public_model_label,
    )
    _callout("Result preview", f"Showing the first {min(max_preview_rows, len(run.results_df)):,} rows only. Full row-level outputs are saved in artifacts.")
    display(results_preview)
    return artifact_table


def prepare_display_tables(run, public_model_label: str) -> dict[str, pd.DataFrame]:
    """Return sanitized result tables for notebook display and plots."""

    tables = {
        "summary": run.summary_df.copy(),
        "dataset_summary": run.dataset_summary_df.copy(),
        "leaderboard": run.leaderboard_df.copy(),
        "diagnostics": run.diagnostics_df.copy(),
        "results": run.results_df.copy(),
    }
    return {name: _sanitize_model_columns(df, public_model_label) for name, df in tables.items()}


def display_result_diagnostics(
    run,
    plan: AnswerAccuracyNotebookPlan,
    *,
    max_flagged_rows: int = 20,
) -> dict[str, pd.DataFrame]:
    """Display summary tables and the highest-priority flagged cases."""

    tables = prepare_display_tables(run, plan.public_model_label)
    _callout("Category summary", "Aggregate score and pass-rate by normalized answer-accuracy category.")
    display(tables["summary"])

    _callout("Dataset/category summary", "Use this table to identify whether low scores are broad or concentrated in one dataset shape.")
    portfolio_df = _sanitize_model_columns(dataset_rollup(tables["dataset_summary"]), plan.public_model_label)
    _callout(
        "Dataset portfolio summary",
        "Dataset-level rollup. Compare directionally, but interpret each dataset through its own scoring profile and reference shape.",
    )
    display(portfolio_df)

    _callout("Dataset/category detail", "Category-level detail for diagnostics and targeted follow-up.")
    display(tables["dataset_summary"])

    _callout("Model leaderboard", "Human-facing model names are generalized; raw deployments are preserved only in artifacts.")
    display(tables["leaderboard"])

    review_cols = _available_columns(
        tables["diagnostics"],
        [
            "flagged",
            "review_priority",
            "flag_reason",
            "recommended_action",
            "model_name",
            "dataset_key",
            "task_id",
            "category",
            "scoring_profile",
            "actual_output",
            "expected_output",
            "score",
            "exact_match",
            "token_f1",
            "semantic_similarity",
            "bleu",
            "rouge_l",
            "contains_match",
            "would_pass_contains_rule",
            "reference_shape_warning",
            "metric_disagreement",
            "judge_score",
            "judge_reason",
        ],
    )
    flagged_df = tables["diagnostics"][tables["diagnostics"]["flagged"]].copy()
    _callout("Flagged-case review queue", "Prioritized exceptions show where the benchmark needs calibration or where the model likely struggled.")
    if flagged_df.empty:
        display(Markdown("✅ No cases were flagged by the current diagnostic rules."))
    else:
        display(flagged_df.sort_values(["score", "task_id"])[review_cols].head(max_flagged_rows))
        if len(flagged_df) > max_flagged_rows:
            display(Markdown(f"Showing the top {max_flagged_rows:,} flagged rows. Full diagnostics are saved in `{run.artifact_paths['diagnostics_csv'].name}`."))

    return tables


def display_answer_accuracy_visuals(
    run,
    tables: dict[str, pd.DataFrame],
    plan: AnswerAccuracyNotebookPlan,
) -> None:
    """Render the standard notebook visualization pack."""

    model_config = plan.model_config
    threshold = plan.workflow_config.pass_threshold or config_float(model_config.get("default_pass_threshold"), 0.7)

    _callout("Dataset-level capability score", "A compact view of average primary score by dataset/category.")
    plot_capability_scores(tables["summary"], title="Answer Accuracy by Dataset / Category")
    plt.show()

    _callout("Metric profile heatmap", "Average metric values by dataset. Divergence across metrics often signals reference-shape or scoring-fit issues.")
    plot_metric_heatmap(tables["diagnostics"], group_col="dataset_key", metric_cols=DEFAULT_METRIC_COLUMNS, title="Metric Profile by Dataset")
    plt.show()

    _callout("Primary and supporting metrics", "Grouped comparison of the profile score against supporting lexical/semantic signals.")
    plot_metric_bars_by_group(
        tables["diagnostics"],
        group_col="dataset_key",
        metric_cols=["primary_score", "token_f1", "semantic_similarity", "rouge_l"],
        title="Primary and Supporting Metrics by Dataset",
    )
    plt.show()

    _callout("Score distributions", "Box plots reveal whether performance is consistently weak or driven by a subset of difficult samples.")
    plot_score_distribution_by_group(run.results_df, group_col="dataset_key", title="Score Distribution by Dataset")
    plt.show()

    _callout("Pass-rate and review-priority dashboard", "This view connects benchmark outcome to review workload.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    pass_rate_df = tables["dataset_summary"].sort_values("pass_rate")
    axes[0].barh(pass_rate_df["dataset_key"], pass_rate_df["pass_rate"], color="#2563eb")
    axes[0].axvline(threshold, color="#dc2626", linestyle="--", label=f"Pass threshold ({threshold:.2f})")
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Pass rate")
    axes[0].set_title("Pass Rate by Dataset", loc="left", fontweight="bold")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].legend(frameon=False)

    priority_counts = tables["diagnostics"]["review_priority"].value_counts().sort_values(ascending=True)
    priority_colors = {
        "High review priority": "#dc2626",
        "Near threshold": "#f97316",
        "Metric disagreement": "#9333ea",
        "Reference-shape warning": "#ca8a04",
        "Contains-only credit": "#0f766e",
        "Looks good": "#16a34a",
    }
    axes[1].barh(
        priority_counts.index,
        priority_counts.values,
        color=[priority_colors.get(label, "#64748b") for label in priority_counts.index],
    )
    axes[1].set_title("Review Priority Mix", loc="left", fontweight="bold")
    axes[1].set_xlabel("Cases")
    axes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout()
    plt.show()


def display_executive_report(run, plan: AnswerAccuracyNotebookPlan, artifact_table: pd.DataFrame) -> None:
    """Display the HTML executive report and concise artifact inventory."""

    display(HTML(html_summary_report(run, target_label=plan.public_model_label, judge_enabled=plan.workflow_config.enable_judge_review)))
    _callout("Saved artifact inventory", "The run directory contains these report and evidence assets.")
    display(artifact_table)


def display_saved_artifacts(run) -> pd.DataFrame:
    """Display a complete artifact inventory with folders."""

    table = artifact_display_table(run)
    _callout(
        "Saved artifacts",
        "These files are generated by the workflow and saved under the run directory for review, reporting, or replay.",
    )
    display(table)
    return table


def artifact_display_table(run) -> pd.DataFrame:
    """Return a stable, human-friendly artifact inventory."""

    rows = []
    for category, purpose, key in ARTIFACT_CATALOG:
        path = run.artifact_paths.get(key)
        if path is not None:
            rows.append(
                {
                    "Category": category,
                    "Purpose": purpose,
                    "File": Path(path).name,
                    "Folder": str(Path(path).parent),
                }
            )
    return pd.DataFrame(rows)


def _callout(title: str, body: str) -> None:
    display(Markdown(f"**{title}**  \n{body}"))


def _check_row(check: str, ready: bool, detail: str) -> dict[str, Any]:
    return {"Check": check, "Ready": bool(ready), "Status": "Ready" if ready else "Review", "Detail": detail}


def _clean_manifest_for_display(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    if "cache_or_source" in display_df:
        display_df["cache_or_source"] = display_df["cache_or_source"].apply(lambda value: Path(str(value)).name if value else "")
    return display_df


def _sanitize_model_columns(df: pd.DataFrame, public_model_label: str) -> pd.DataFrame:
    display_df = df.copy()
    for col in ["model_name", "model"]:
        if col in display_df:
            display_df[col] = public_model_label
    return display_df


def _available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def _checkpoint_label(path: Path | None) -> str:
    if path is None:
        return "None"
    path = Path(path)
    if path.parent.name:
        return path.parent.name
    return path.name


def default_judge_model_name() -> str | None:
    """Return the preferred judge-model environment variable."""

    return os.environ.get("OPENAI_JUDGE_MODEL") or os.environ.get("JUDGE_MODEL")
