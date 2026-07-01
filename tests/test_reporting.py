from pathlib import Path

import pandas as pd

from genai_capability_bench.core.schemas import ModelSpec
from genai_capability_bench.notebooks.answer_accuracy import (
    ANSWER_ACCURACY_DATASET_PRESETS,
    RELATED_CAPABILITY_PRESETS,
    prepare_answer_accuracy_plan,
)
from genai_capability_bench.reporting.notebook_views import dataset_preset_table
from genai_capability_bench.reporting.model_labels import model_public_label
from genai_capability_bench.reporting.ratings import assess_answer_accuracy_run, dataset_rollup
from genai_capability_bench.workflows.answer_accuracy import _build_markdown_report


def test_dataset_preset_table_excludes_rag_and_truthfulness_datasets():
    table = dataset_preset_table()
    preset_text = " ".join(table["Preset"].astype(str).tolist() + table["Datasets"].astype(str).tolist())

    assert "knowledge_portfolio" in set(table["Preset"])
    assert "curated_knowledge" in set(table["Preset"])
    assert "curated_knowledge_v1" in preset_text
    assert "squad" not in preset_text
    assert "hotpotqa" not in preset_text
    assert "truthfulqa" not in preset_text


def test_answer_accuracy_notebook_routes_adjacent_capability_datasets():
    active_keys = {key for keys in ANSWER_ACCURACY_DATASET_PRESETS.values() for key in keys}
    routed_keys = {key for keys in RELATED_CAPABILITY_PRESETS.values() for key in keys}

    assert ANSWER_ACCURACY_DATASET_PRESETS["knowledge_portfolio"] == ["mmlu", "triviaqa", "natural_questions", "arc"]
    assert ANSWER_ACCURACY_DATASET_PRESETS["curated_knowledge"] == ["curated_knowledge_v1"]
    assert {"squad", "hotpotqa", "truthfulqa"} <= routed_keys
    assert not {"squad", "hotpotqa", "truthfulqa"} & active_keys


def test_rating_separates_capability_from_measurement_reliability():
    results_df = pd.DataFrame(
        [
            {"score": 1.0, "passed": True},
            {"score": 1.0, "passed": True},
            {"score": 1.0, "passed": True},
            {"score": 0.0, "passed": False},
        ]
    )
    diagnostics_df = pd.DataFrame(
        [
            {"flagged": False, "metric_disagreement": False, "passed": True, "judge_score": None, "judge_reason": ""},
            {"flagged": False, "metric_disagreement": False, "passed": True, "judge_score": None, "judge_reason": ""},
            {"flagged": False, "metric_disagreement": False, "passed": True, "judge_score": None, "judge_reason": ""},
            {"flagged": True, "metric_disagreement": True, "passed": False, "judge_score": 1.0, "judge_reason": "Likely correct."},
        ]
    )
    manifest_df = pd.DataFrame([{"dataset_key": "mmlu", "reference_shape": "option_text_and_label"}])

    assessment = assess_answer_accuracy_run(
        results_df=results_df,
        diagnostics_df=diagnostics_df,
        dataset_manifest_df=manifest_df,
    )

    assert assessment.capability_rating == "Moderate-Strong"
    assert assessment.reliability_rating == "Low"


def test_dataset_rollup_uses_weighted_category_results():
    summary_df = pd.DataFrame(
        [
            {"model_name": "m", "dataset_key": "d1", "category": "a", "n": 1, "avg_score": 1.0, "pass_rate": 1.0},
            {"model_name": "m", "dataset_key": "d1", "category": "b", "n": 3, "avg_score": 0.0, "pass_rate": 0.0},
        ]
    )

    rollup = dataset_rollup(summary_df)

    assert rollup.iloc[0]["n"] == 4
    assert rollup.iloc[0]["avg_score"] == 0.25
    assert rollup.iloc[0]["pass_rate"] == 0.25


def test_answer_accuracy_notebook_rejects_context_required_manual_dataset():
    repo_root = Path(".").resolve()

    try:
        prepare_answer_accuracy_plan(
            repo_root=repo_root,
            output_root=repo_root / "outputs" / "runs",
            model_config_path=repo_root / "configs" / "eval_core_demo.yaml",
            dataset_preset="open_domain_qa",
            run_mode="quick_check",
            dataset_keys=["squad"],
            enable_judge_review=False,
        )
    except ValueError as exc:
        assert "requires supplied context" in str(exc)
    else:
        raise AssertionError("Expected context-required dataset guard to raise ValueError")


def test_answer_accuracy_notebook_rejects_routed_truthfulness_preset():
    repo_root = Path(".").resolve()

    try:
        prepare_answer_accuracy_plan(
            repo_root=repo_root,
            output_root=repo_root / "outputs" / "runs",
            model_config_path=repo_root / "configs" / "eval_core_demo.yaml",
            dataset_preset="truthfulness_misconceptions",
            run_mode="quick_check",
            enable_judge_review=False,
        )
    except ValueError as exc:
        assert "routed suite preset" in str(exc)
        assert "Truthfulness" in str(exc)
    else:
        raise AssertionError("Expected routed preset guard to raise ValueError")


def test_model_public_label_hides_deployment_suffix():
    model = ModelSpec(
        name="gpt-5-4-20260305-gs",
        provider="openai_compatible",
        model="gpt-5-4-20260305-gs",
        api_version="2025-04-01-preview",
    )

    assert model_public_label(model) == "GPT 5-4 (Azure)"


def test_answer_accuracy_report_uses_public_model_label():
    raw_name = "gpt-5-4-20260305-gs"
    model = ModelSpec(
        name=raw_name,
        provider="openai_compatible",
        model=raw_name,
        api_version="2025-04-01-preview",
    )
    results_df = pd.DataFrame(
        [
            {
                "model_name": raw_name,
                "task_id": "task_1",
                "score": 0.8,
                "passed": True,
                "dataset_key": "triviaqa",
            }
        ]
    )
    summary_df = pd.DataFrame(
        [
            {
                "model_name": raw_name,
                "capability": "answer_accuracy",
                "category": "triviaqa",
                "n": 1,
                "avg_score": 0.8,
                "pass_rate": 1.0,
            }
        ]
    )
    dataset_summary_df = pd.DataFrame(
        [{"model_name": raw_name, "dataset_key": "triviaqa", "category": "triviaqa", "n": 1, "avg_score": 0.8, "pass_rate": 1.0}]
    )
    diagnostics_df = pd.DataFrame(
        [{"model_name": raw_name, "score": 0.8, "passed": True, "flagged": False, "review_priority": "Looks good"}]
    )
    dataset_manifest_df = pd.DataFrame(
        [
            {
                "dataset_key": "triviaqa",
                "split": "validation",
                "answer_accuracy_tasks": 1,
                "scoring_profile": "short_answer_qa",
                "reference_shape": "aliases",
                "cache_or_source": "cache.json",
                "primary_metrics": "exact_match, token_f1",
            }
        ]
    )

    report = _build_markdown_report(
        run_id="run_1",
        models=[model],
        tasks=[object()],
        pass_threshold=0.7,
        results_df=results_df,
        summary_df=summary_df,
        dataset_manifest_df=dataset_manifest_df,
        dataset_summary_df=dataset_summary_df,
        diagnostics_df=diagnostics_df,
        reliability_notes=[],
        checkpoint_path=Path("checkpoint.jsonl"),
        resumed_from_checkpoint=None,
        summary_text=f"Strongest model was {raw_name}.",
        judge_enabled=False,
        artifact_paths={"raw_results_csv": Path("answer_accuracy_raw_results_20260622_000000.csv")},
    )

    assert "GPT 5-4 (Azure)" in report
    assert raw_name not in report
