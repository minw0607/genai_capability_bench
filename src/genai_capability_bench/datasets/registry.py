"""Reusable dataset registry and normalization layer.

The loaders in this module normalize very different public benchmark schemas into
the repo's shared EvalTask format. The same layer can be reused by future
Truthfulness, Reasoning, RAG, Tool Use, and Agent notebooks.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from genai_capability_bench.core.runner import load_tasks
from genai_capability_bench.core.schemas import Capability, EvalTask


Normalizer = Callable[[dict[str, Any], int], EvalTask | None]


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset source metadata."""

    key: str
    display_name: str
    capability: Capability
    source_type: str  # "local" | "custom" | "huggingface"
    description: str
    local_path: str | None = None
    hf_path: str | None = None
    hf_config: str | None = None
    default_split: str = "validation"
    normalizer: str = "generic"
    requires_context: bool = False
    task_format: str = "open_qa"
    scoring_guidance: str = "reference_match"
    notes: str = ""


DATASET_SPECS: dict[str, DatasetSpec] = {
    "answer_accuracy_sample": DatasetSpec(
        key="answer_accuracy_sample",
        display_name="Answer Accuracy Sample",
        capability=Capability.ANSWER_ACCURACY,
        source_type="local",
        local_path="datasets/samples/answer_accuracy_sample.json",
        default_split="local",
        normalizer="already_normalized",
        description="Small local closed-book factual QA sample across common domains.",
        task_format="closed_book_short_answer",
        scoring_guidance="exact/contains/F1/TF-IDF reference matching",
    ),
    "core_demo_mixed": DatasetSpec(
        key="core_demo_mixed",
        display_name="Core Demo Mixed",
        capability=Capability.ANSWER_ACCURACY,
        source_type="local",
        local_path="datasets/samples/core_demo_tasks.json",
        default_split="local",
        normalizer="already_normalized",
        description="Tiny mixed-capability smoke-test dataset; filtered to answer accuracy in this notebook.",
        task_format="mixed_demo",
        scoring_guidance="capability-specific starter scoring",
    ),
    "mmlu": DatasetSpec(
        key="mmlu",
        display_name="MMLU",
        capability=Capability.ANSWER_ACCURACY,
        source_type="huggingface",
        hf_path="cais/mmlu",
        hf_config="all",
        default_split="test",
        normalizer="mmlu",
        description="Massive Multitask Language Understanding multiple-choice benchmark.",
        task_format="multiple_choice",
        scoring_guidance="option-text and answer-key matching; future grader should use exact option accuracy",
        notes="Large benchmark. Use small sample sizes first.",
    ),
    "triviaqa": DatasetSpec(
        key="triviaqa",
        display_name="TriviaQA",
        capability=Capability.ANSWER_ACCURACY,
        source_type="huggingface",
        hf_path="mandarjoshi/trivia_qa",
        hf_config="rc.nocontext",
        default_split="validation",
        normalizer="triviaqa",
        description="Open-domain trivia question answering benchmark.",
        task_format="open_domain_short_answer",
        scoring_guidance="alias/reference matching; embedding review useful for paraphrases",
        notes="Some configurations are large; normalized cache is recommended.",
    ),
    "natural_questions": DatasetSpec(
        key="natural_questions",
        display_name="Natural Questions",
        capability=Capability.ANSWER_ACCURACY,
        source_type="huggingface",
        hf_path="sentence-transformers/natural-questions",
        default_split="train",
        normalizer="natural_questions",
        description="Real user-style question-answer pairs derived from Natural Questions.",
        task_format="open_domain_short_answer",
        scoring_guidance="reference matching; answers may have multiple aliases",
    ),
    "squad": DatasetSpec(
        key="squad",
        display_name="SQuAD",
        capability=Capability.ANSWER_ACCURACY,
        source_type="huggingface",
        hf_path="rajpurkar/squad",
        default_split="validation",
        normalizer="squad",
        requires_context=True,
        description="Reading-comprehension QA over Wikipedia passages.",
        task_format="context_grounded_qa",
        scoring_guidance="reference matching plus future context-grounded scoring",
        notes="Context is stored in task metadata; better suited to future context/RAG workflows.",
    ),
    "arc": DatasetSpec(
        key="arc",
        display_name="ARC Challenge",
        capability=Capability.ANSWER_ACCURACY,
        source_type="huggingface",
        hf_path="ai2_arc",
        hf_config="ARC-Challenge",
        default_split="test",
        normalizer="arc",
        description="Grade-school science multiple-choice QA benchmark.",
        task_format="multiple_choice",
        scoring_guidance="option-text and answer-key matching; future grader should use exact option accuracy",
    ),
    "hotpotqa": DatasetSpec(
        key="hotpotqa",
        display_name="HotpotQA",
        capability=Capability.ANSWER_ACCURACY,
        source_type="huggingface",
        hf_path="hotpot_qa",
        hf_config="fullwiki",
        default_split="validation",
        normalizer="hotpotqa",
        requires_context=True,
        description="Diverse explainable multi-hop QA benchmark.",
        task_format="multi_hop_qa",
        scoring_guidance="reference matching; future scoring should separate reasoning and RAG/context use",
        notes="Also relevant to future reasoning and RAG capability notebooks.",
    ),
    "truthfulqa": DatasetSpec(
        key="truthfulqa",
        display_name="TruthfulQA",
        capability=Capability.TRUTHFULNESS,
        source_type="huggingface",
        hf_path="truthful_qa",
        hf_config="generation",
        default_split="validation",
        normalizer="truthfulqa",
        description="Truthfulness benchmark built around common misconceptions.",
        task_format="truthfulness_generation",
        scoring_guidance="correct-vs-incorrect references; primary evaluator should be Truthfulness",
        notes="Primary home is the Truthfulness notebook; selectable here with caveats.",
    ),
    "custom": DatasetSpec(
        key="custom",
        display_name="Custom Dataset",
        capability=Capability.ANSWER_ACCURACY,
        source_type="custom",
        default_split="custom",
        normalizer="already_normalized",
        description="User-provided JSON, JSONL, or CSV file already matching the EvalTask schema.",
        task_format="custom_normalized",
        scoring_guidance="depends on provided references and metadata",
    ),
}


NORMALIZERS: dict[str, Normalizer] = {}


def _normalizer(name: str):
    def wrap(fn: Normalizer) -> Normalizer:
        NORMALIZERS[name] = fn
        return fn

    return wrap


def list_dataset_specs(capability: Capability | None = None) -> list[DatasetSpec]:
    """Return registered datasets, optionally filtered by primary capability."""

    specs = list(DATASET_SPECS.values())
    if capability is not None:
        specs = [s for s in specs if s.capability == capability or s.key in {"custom"}]
    return specs


def get_dataset_spec(key: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_SPECS))
        raise KeyError(f"Unknown dataset key '{key}'. Valid options: {valid}") from exc


def materialize_dataset(
    key: str,
    repo_root: str | Path = ".",
    *,
    split: str | None = None,
    sample_size: int | None = None,
    seed: int = 42,
    download_if_missing: bool = True,
    cache_local_copy: bool = True,
    refresh_cache: bool = False,
    custom_path: str | Path | None = None,
) -> tuple[list[EvalTask], Path | None]:
    """Load, normalize, and optionally cache a dataset.

    Returns (tasks, cache_path). For local/custom datasets, cache_path may point
    to the source file. For Hugging Face datasets, cache_path points to the
    normalized local copy when caching is enabled.
    """

    repo_root = Path(repo_root)
    spec = get_dataset_spec(key)
    split = split or spec.default_split
    cache_path = _cache_path(repo_root, spec.key, split, sample_size)

    if spec.source_type == "local":
        source = repo_root / str(spec.local_path)
        tasks = load_tasks(source)
        tasks = [_enrich_choice_references(task) for task in tasks]
        return _sample(tasks, sample_size, seed), source

    if spec.source_type == "custom":
        if custom_path is None:
            raise ValueError("custom_path is required when key='custom'")
        source = Path(custom_path)
        tasks = load_tasks(source)
        tasks = [_enrich_choice_references(task) for task in tasks]
        return _sample(tasks, sample_size, seed), source

    if spec.source_type != "huggingface":
        raise ValueError(f"Unsupported source_type for {spec.key}: {spec.source_type}")

    if cache_path.exists() and not refresh_cache:
        tasks = [_enrich_choice_references(task) for task in load_tasks(cache_path)]
        return tasks, cache_path

    if not download_if_missing:
        raise FileNotFoundError(
            f"No cached dataset found at {cache_path}. Set download_if_missing=True to download it."
        )

    rows = _load_huggingface_rows(spec, split=split)
    rows = _sample_rows(rows, sample_size, seed)
    tasks = _normalize_rows(spec, rows)

    if cache_local_copy:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _write_tasks(cache_path, tasks)
        return tasks, cache_path

    return tasks, None


def dataset_options_table(capability: Capability | None = None) -> pd.DataFrame:
    """Small helper for notebook display."""

    rows = []
    for spec in list_dataset_specs(capability):
        rows.append(
            {
                "key": spec.key,
                "display_name": spec.display_name,
                "source": spec.source_type,
                "default_split": spec.default_split,
                "description": spec.description,
                "task_format": spec.task_format,
                "scoring_guidance": spec.scoring_guidance,
                "notes": spec.notes,
            }
        )
    return pd.DataFrame(rows)


def _cache_path(repo_root: Path, key: str, split: str, sample_size: int | None) -> Path:
    sample = "all" if sample_size is None else str(sample_size)
    return repo_root / "datasets" / "cache" / key / f"{split}_sample_{sample}.json"


def _load_huggingface_rows(spec: DatasetSpec, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Hugging Face dataset loading requires the optional 'datasets' package. "
            "Install with: pip install -e '.[hf]' or pip install datasets"
        ) from exc

    try:
        if spec.hf_config:
            ds = load_dataset(spec.hf_path, spec.hf_config, split=split)
        else:
            ds = load_dataset(spec.hf_path, split=split)
    except Exception as first_exc:
        # Some community datasets change configs/splits. Retry without config
        # before surfacing the original failure.
        if spec.hf_config:
            try:
                ds = load_dataset(spec.hf_path, split=split)
            except Exception:
                raise first_exc
        else:
            raise

    return [dict(row) for row in ds]


def _normalize_rows(spec: DatasetSpec, rows: list[dict[str, Any]]) -> list[EvalTask]:
    normalizer = NORMALIZERS.get(spec.normalizer)
    if normalizer is None:
        raise ValueError(f"No normalizer registered for {spec.normalizer}")
    tasks = []
    for idx, row in enumerate(rows):
        task = normalizer(row, idx)
        if task is not None:
            tasks.append(task)
    return tasks


def _sample(tasks: list[EvalTask], sample_size: int | None, seed: int) -> list[EvalTask]:
    if sample_size is None or sample_size >= len(tasks):
        return tasks
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(tasks)), sample_size))
    return [tasks[i] for i in indices]


def _sample_rows(rows: list[dict[str, Any]], sample_size: int | None, seed: int) -> list[dict[str, Any]]:
    if sample_size is None or sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), sample_size))
    return [rows[i] for i in indices]


def _write_tasks(path: Path, tasks: list[EvalTask]) -> None:
    rows = []
    for task in tasks:
        row = {
            "task_id": task.task_id,
            "capability": task.capability.value,
            "input_text": task.input_text,
            "expected_output": task.expected_output,
            "category": task.category,
            "subcategory": task.subcategory,
            "references": task.references,
            "incorrect_references": task.incorrect_references,
            "metadata": task.metadata,
        }
        rows.append(row)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _first_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        for item in value:
            out = _first_text(item)
            if out:
                return out
    if isinstance(value, dict):
        for key in ("text", "answer", "value"):
            out = _first_text(value.get(key))
            if out:
                return out
    return None


def _answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v).strip()]
    if isinstance(value, dict):
        for key in ("text", "normalized_aliases", "aliases", "value", "answer"):
            values = _answer_list(value.get(key))
            if values:
                return values
    return []


def _choice_text(choices: Any, answer_key: Any = None) -> tuple[str, str | None]:
    """Return formatted choices and expected answer text if possible."""

    labels: list[str] = []
    texts: list[str] = []
    if isinstance(choices, dict):
        labels = [str(x) for x in choices.get("label", [])]
        texts = [str(x) for x in choices.get("text", [])]
    elif isinstance(choices, list):
        texts = [str(x) for x in choices]
        labels = [chr(ord("A") + i) for i in range(len(texts))]

    formatted = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    answer = None
    if answer_key is not None:
        key = str(answer_key)
        if key.isdigit():
            idx = int(key)
            if 0 <= idx < len(texts):
                answer = texts[idx]
        elif key in labels:
            answer = texts[labels.index(key)]
        elif len(key) == 1 and key.upper() in labels:
            answer = texts[labels.index(key.upper())]
        else:
            answer = key
    return formatted, answer


def _choice_answer_label(choices: Any, answer_key: Any = None) -> str | None:
    """Return the displayed multiple-choice label for an answer key."""

    if answer_key is None:
        return None
    labels: list[str] = []
    if isinstance(choices, dict):
        labels = [str(x) for x in choices.get("label", [])]
    elif isinstance(choices, list):
        labels = [chr(ord("A") + i) for i in range(len(choices))]

    key = str(answer_key)
    if key.isdigit():
        idx = int(key)
        if 0 <= idx < len(labels):
            return labels[idx]
    if key in labels:
        return key
    if len(key) == 1 and key.upper() in labels:
        return key.upper()
    return None


def _choice_references(answer_text: str, choices: Any, answer_key: Any = None) -> list[str]:
    """Build references for multiple-choice tasks using answer text and displayed label."""

    refs = [answer_text]
    label = _choice_answer_label(choices, answer_key)
    if label:
        refs.append(label)
    elif answer_key is not None:
        refs.append(str(answer_key))
    return list(dict.fromkeys(refs))


def _enrich_choice_references(task: EvalTask) -> EvalTask:
    """Add displayed option labels to cached multiple-choice tasks when possible."""

    choices = task.metadata.get("choices") if isinstance(task.metadata, dict) else None
    answer_key = task.metadata.get("answer_key") if isinstance(task.metadata, dict) else None
    if not choices or answer_key is None or not task.expected_output:
        return task
    task.references = list(
        dict.fromkeys([*task.references, *_choice_references(task.expected_output, choices, answer_key)])
    )
    return task


@_normalizer("already_normalized")
def _already_normalized(row: dict[str, Any], idx: int) -> EvalTask | None:
    return EvalTask(
        task_id=str(row.get("task_id", f"custom_{idx}")),
        capability=Capability(row.get("capability", Capability.ANSWER_ACCURACY.value)),
        input_text=str(row["input_text"]),
        expected_output=row.get("expected_output"),
        category=row.get("category", "general"),
        subcategory=row.get("subcategory"),
        references=list(row.get("references") or []),
        incorrect_references=list(row.get("incorrect_references") or []),
        metadata=dict(row.get("metadata") or {}),
    )


@_normalizer("mmlu")
def _normalize_mmlu(row: dict[str, Any], idx: int) -> EvalTask | None:
    question = row.get("question")
    choices = row.get("choices")
    answer_key = row.get("answer")
    formatted, answer_text = _choice_text(choices, answer_key)
    if not question or not answer_text:
        return None
    subject = str(row.get("subject") or row.get("category") or "mmlu")
    input_text = f"{question}\n\nChoices:\n{formatted}\n\nAnswer with the correct option."
    return EvalTask(
        task_id=f"mmlu_{subject}_{idx}",
        capability=Capability.ANSWER_ACCURACY,
        input_text=input_text,
        expected_output=answer_text,
        category=subject,
        references=_choice_references(answer_text, choices, answer_key),
        metadata={"source_dataset": "mmlu", "choices": choices, "answer_key": answer_key},
    )


@_normalizer("triviaqa")
def _normalize_triviaqa(row: dict[str, Any], idx: int) -> EvalTask | None:
    question = row.get("question")
    answers = _answer_list(row.get("answer")) or _answer_list(row.get("answers"))
    if not question or not answers:
        return None
    return EvalTask(
        task_id=str(row.get("question_id") or row.get("id") or f"triviaqa_{idx}"),
        capability=Capability.ANSWER_ACCURACY,
        input_text=str(question),
        expected_output=answers[0],
        category="triviaqa",
        references=answers,
        metadata={"source_dataset": "triviaqa"},
    )


@_normalizer("natural_questions")
def _normalize_natural_questions(row: dict[str, Any], idx: int) -> EvalTask | None:
    question = row.get("query") or row.get("question") or row.get("question_text")
    answers = _answer_list(row.get("answer")) or _answer_list(row.get("answers"))
    if not question or not answers:
        return None
    return EvalTask(
        task_id=str(row.get("id") or f"natural_questions_{idx}"),
        capability=Capability.ANSWER_ACCURACY,
        input_text=str(question),
        expected_output=answers[0],
        category="natural_questions",
        references=answers,
        metadata={"source_dataset": "natural_questions"},
    )


@_normalizer("squad")
def _normalize_squad(row: dict[str, Any], idx: int) -> EvalTask | None:
    question = row.get("question")
    answers = _answer_list(row.get("answers"))
    if not question or not answers:
        return None
    return EvalTask(
        task_id=str(row.get("id") or f"squad_{idx}"),
        capability=Capability.ANSWER_ACCURACY,
        input_text=str(question),
        expected_output=answers[0],
        category="squad",
        references=answers,
        metadata={"source_dataset": "squad", "context": row.get("context"), "title": row.get("title")},
    )


@_normalizer("arc")
def _normalize_arc(row: dict[str, Any], idx: int) -> EvalTask | None:
    question = row.get("question")
    formatted, answer_text = _choice_text(row.get("choices"), row.get("answerKey"))
    if not question or not answer_text:
        return None
    input_text = f"{question}\n\nChoices:\n{formatted}\n\nAnswer with the correct option."
    return EvalTask(
        task_id=str(row.get("id") or f"arc_{idx}"),
        capability=Capability.ANSWER_ACCURACY,
        input_text=input_text,
        expected_output=answer_text,
        category="arc_science",
        references=_choice_references(answer_text, row.get("choices"), row.get("answerKey")),
        metadata={"source_dataset": "arc", "choices": row.get("choices"), "answer_key": row.get("answerKey")},
    )


@_normalizer("hotpotqa")
def _normalize_hotpotqa(row: dict[str, Any], idx: int) -> EvalTask | None:
    question = row.get("question")
    answer = _first_text(row.get("answer"))
    if not question or not answer:
        return None
    return EvalTask(
        task_id=str(row.get("id") or row.get("_id") or f"hotpotqa_{idx}"),
        capability=Capability.ANSWER_ACCURACY,
        input_text=str(question),
        expected_output=answer,
        category="hotpotqa",
        references=[answer],
        metadata={
            "source_dataset": "hotpotqa",
            "level": row.get("level"),
            "type": row.get("type"),
            "context": row.get("context"),
            "supporting_facts": row.get("supporting_facts"),
        },
    )


@_normalizer("truthfulqa")
def _normalize_truthfulqa(row: dict[str, Any], idx: int) -> EvalTask | None:
    question = row.get("question")
    correct = _answer_list(row.get("correct_answers")) or _answer_list(row.get("best_answer"))
    incorrect = _answer_list(row.get("incorrect_answers"))
    if not question or not correct:
        return None
    return EvalTask(
        task_id=str(row.get("id") or f"truthfulqa_{idx}"),
        capability=Capability.TRUTHFULNESS,
        input_text=str(question),
        expected_output=correct[0],
        category=str(row.get("category") or "truthfulqa"),
        references=correct,
        incorrect_references=incorrect,
        metadata={"source_dataset": "truthfulqa", "type": row.get("type")},
    )
