from pathlib import Path

from genai_capability_bench.workflows.answer_accuracy import (
    AnswerAccuracyRunConfig,
    load_answer_accuracy_tasks,
    run_answer_accuracy_workflow,
)


def test_answer_accuracy_workflow_local_sample(tmp_path):
    repo_root = Path(".").resolve()
    config = AnswerAccuracyRunConfig(
        repo_root=repo_root,
        model_config_path=repo_root / "configs" / "eval_core_demo.yaml",
        dataset_keys=["answer_accuracy_sample"],
        sample_size_per_dataset=2,
        output_root=tmp_path,
        run_id_prefix="pytest_answer_accuracy",
    )

    result = run_answer_accuracy_workflow(config, show_progress=False)

    assert len(result.results_df) == 2
    assert result.report_path.exists()
    assert (result.run_dir / "diagnostics.csv").exists()
    assert (result.run_dir / "dataset_manifest.csv").exists()
    assert (result.run_dir / "dataset_summary.csv").exists()
    assert "dataset_key" in result.results_df.columns
    assert len(result.dataset_summary_df) > 0


def test_full_public_dataset_requires_explicit_opt_in():
    repo_root = Path(".").resolve()
    config = AnswerAccuracyRunConfig(
        repo_root=repo_root,
        model_config_path=repo_root / "configs" / "eval_core_demo.yaml",
        dataset_keys=["mmlu"],
        sample_size_per_dataset=None,
        download_if_missing=False,
    )

    try:
        load_answer_accuracy_tasks(config)
    except ValueError as exc:
        assert "Refusing to load the full public dataset split" in str(exc)
    else:
        raise AssertionError("Expected full public dataset guard to raise ValueError")


def test_run_refuses_more_than_max_tasks(tmp_path):
    repo_root = Path(".").resolve()
    config = AnswerAccuracyRunConfig(
        repo_root=repo_root,
        model_config_path=repo_root / "configs" / "eval_core_demo.yaml",
        dataset_keys=["answer_accuracy_sample"],
        sample_size_per_dataset=3,
        output_root=tmp_path,
        max_tasks_per_run=2,
    )

    try:
        run_answer_accuracy_workflow(config, show_progress=False)
    except ValueError as exc:
        assert "Refusing to evaluate 3 tasks" in str(exc)
    else:
        raise AssertionError("Expected max task guard to raise ValueError")
