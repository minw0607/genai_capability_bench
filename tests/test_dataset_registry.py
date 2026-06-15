from pathlib import Path

from genai_capability_bench.core.schemas import Capability
from genai_capability_bench.datasets import get_dataset_spec, list_dataset_specs, materialize_dataset
from genai_capability_bench.datasets.registry import _normalize_mmlu


def test_list_dataset_specs_includes_public_datasets():
    keys = {spec.key for spec in list_dataset_specs()}
    assert {"mmlu", "triviaqa", "natural_questions", "squad", "arc", "hotpotqa", "truthfulqa"} <= keys


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
    assert task.references == ["correct", "C"]
    assert "2" not in task.references
