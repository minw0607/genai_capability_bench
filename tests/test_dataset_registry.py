import json
from pathlib import Path

from genai_capability_bench.core.schemas import Capability
from genai_capability_bench.datasets import get_dataset_spec, list_dataset_specs, materialize_dataset
from genai_capability_bench.datasets.registry import _normalize_mmlu


def test_list_dataset_specs_includes_public_datasets():
    keys = {spec.key for spec in list_dataset_specs()}
    assert {"mmlu", "triviaqa", "natural_questions", "squad", "arc", "hotpotqa", "truthfulqa"} <= keys
    assert "curated_knowledge_v1" in keys


def test_materialize_local_answer_accuracy_sample():
    tasks, source = materialize_dataset(
        "answer_accuracy_sample",
        repo_root=Path("."),
        sample_size=3,
        seed=42,
    )
    assert source is not None
    assert len(tasks) == 3
    assert all(t.capability == Capability.ANSWER_ACCURACY for t in tasks)


def test_get_dataset_spec_rejects_unknown_key():
    try:
        get_dataset_spec("not_a_dataset")
    except KeyError as exc:
        assert "Unknown dataset key" in str(exc)
    else:
        raise AssertionError("Expected KeyError")


def test_mmlu_normalizer_uses_displayed_option_label_reference():
    task = _normalize_mmlu(
        {
            "question": "Which option is correct?",
            "choices": ["wrong", "also wrong", "correct", "nope"],
            "answer": 2,
            "subject": "demo_subject",
        },
        0,
    )

    assert task is not None
    assert task.expected_output == "correct"
    assert task.references == ["correct", "C", "C. correct", "C correct"]
    assert "2" not in task.references


def test_dataset_inventory_declares_scoring_profiles():
    trivia = get_dataset_spec("triviaqa")
    natural_questions = get_dataset_spec("natural_questions")
    mmlu = get_dataset_spec("mmlu")

    assert trivia.scoring_profile == "short_answer_qa"
    assert natural_questions.scoring_profile == "long_reference_qa"
    assert natural_questions.reference_shape == "passage_or_long_answer"
    assert mmlu.scoring_profile == "multiple_choice"


def test_materialized_tasks_include_dataset_metric_metadata():
    tasks, _ = materialize_dataset(
        "answer_accuracy_sample",
        repo_root=Path("."),
        sample_size=1,
        seed=42,
    )

    assert tasks[0].metadata["scoring_profile"] == "short_answer_qa"
    assert "primary_metrics" in tasks[0].metadata


def test_materialize_curated_knowledge_preserves_source_provenance_and_scoring():
    tasks, source = materialize_dataset(
        "curated_knowledge_v1",
        repo_root=Path("."),
        sample_size=10,
        seed=42,
    )

    assert source is not None
    assert source.name == "answer_accuracy_knowledge_v1.jsonl"
    assert len(tasks) == 10
    assert all(task.capability == Capability.ANSWER_ACCURACY for task in tasks)
    assert all(task.metadata["curated_dataset"] == "answer_accuracy_knowledge_v1" for task in tasks)
    assert all(task.metadata["dataset_key"] == "curated_knowledge_v1" for task in tasks)
    assert all(task.metadata.get("source_dataset") in {"arc", "mmlu", "triviaqa"} for task in tasks)
    assert all(task.metadata.get("source_task_id") for task in tasks)
    assert all(task.metadata.get("scoring_profile") in {"multiple_choice", "short_answer_qa"} for task in tasks)


def test_curated_knowledge_uses_source_balanced_stratified_sampling_by_default():
    tasks, _ = materialize_dataset(
        "curated_knowledge_v1",
        repo_root=Path("."),
        sample_size=90,
        seed=42,
    )

    source_counts = {}
    for task in tasks:
        source = task.metadata.get("source_dataset")
        source_counts[source] = source_counts.get(source, 0) + 1

    assert set(source_counts) == {"arc", "mmlu", "triviaqa"}
    assert min(source_counts.values()) >= 25
    assert max(source_counts.values()) <= 35


def test_curated_knowledge_manifest_tracks_full_compatible_sources():
    manifest_path = Path("datasets/curated/answer_accuracy_knowledge_v1_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["row_count"] >= 30_000
    assert manifest["source_counts"]["mmlu"] >= 14_000
    assert manifest["source_counts"]["triviaqa"] >= 17_000
    assert manifest["source_counts"]["arc"] >= 1_000
    assert manifest["skipped_sources"] == []
    assert manifest["compatible_source_scope"] == ["mmlu", "triviaqa", "arc"]
