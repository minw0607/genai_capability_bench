from pathlib import Path

from genai_capability_bench.core.schemas import Capability
from genai_capability_bench.datasets import get_dataset_spec, list_dataset_specs, materialize_dataset


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
