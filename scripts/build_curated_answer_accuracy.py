"""Build the curated closed-book answer-accuracy dataset.

The curated dataset is derived from normalized public benchmark caches without
editing source questions or reference answers. It keeps a uniform EvalTask shape
while preserving source provenance in metadata.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from genai_capability_bench.core.runner import load_tasks
from genai_capability_bench.core.schemas import EvalTask
from genai_capability_bench.datasets.registry import _enrich_choice_references, _enrich_task_metadata, get_dataset_spec


CURATED_VERSION = "answer_accuracy_knowledge_v1"
SOURCE_FILES = [
    (
        "mmlu",
        [
            "datasets/cache/mmlu/test_sample_all.json",
            "datasets/cache/mmlu/test_sample_100.json",
            "datasets/cache/mmlu/test_sample_10.json",
        ],
    ),
    (
        "triviaqa",
        [
            "datasets/cache/triviaqa/validation_sample_all.json",
            "datasets/cache/triviaqa/validation_sample_100.json",
            "datasets/cache/triviaqa/validation_sample_25.json",
        ],
    ),
    (
        "arc",
        [
            "datasets/cache/arc/test_sample_all.json",
            "datasets/cache/arc/test_sample_100.json",
        ],
    ),
]

SOURCE_DECISIONS = {
    "mmlu": "Included: broad closed-book multiple-choice knowledge coverage.",
    "triviaqa": "Included: concise open-domain factual QA aliases.",
    "arc": "Included when cached: science exam-style multiple-choice QA.",
    "natural_questions": (
        "Available as an out-of-box benchmark, but excluded from curated v1 because the current "
        "cache uses long-reference answers that are less compatible with concise closed-book QA."
    ),
}


DOMAIN_MAP = {
    "anatomy": ("health_medicine", "anatomy"),
    "clinical_knowledge": ("health_medicine", "clinical_knowledge"),
    "college_medicine": ("health_medicine", "college_medicine"),
    "professional_medicine": ("health_medicine", "professional_medicine"),
    "medical_genetics": ("health_medicine", "medical_genetics"),
    "nutrition": ("health_medicine", "nutrition"),
    "human_aging": ("health_medicine", "human_aging"),
    "business_ethics": ("business", "business_ethics"),
    "econometrics": ("business", "econometrics"),
    "professional_accounting": ("business", "professional_accounting"),
    "high_school_macroeconomics": ("business", "macroeconomics"),
    "high_school_microeconomics": ("business", "microeconomics"),
    "college_mathematics": ("math", "college_mathematics"),
    "elementary_mathematics": ("math", "elementary_mathematics"),
    "high_school_mathematics": ("math", "high_school_mathematics"),
    "high_school_statistics": ("math", "statistics"),
    "formal_logic": ("reasoning_logic", "formal_logic"),
    "college_physics": ("science", "college_physics"),
    "conceptual_physics": ("science", "conceptual_physics"),
    "high_school_physics": ("science", "high_school_physics"),
    "high_school_chemistry": ("science", "high_school_chemistry"),
    "electrical_engineering": ("engineering_technology", "electrical_engineering"),
    "computer_security": ("engineering_technology", "computer_security"),
    "high_school_computer_science": ("engineering_technology", "computer_science"),
    "machine_learning": ("engineering_technology", "machine_learning"),
    "professional_law": ("law_government", "professional_law"),
    "high_school_government_and_politics": ("law_government", "government_and_politics"),
    "us_foreign_policy": ("law_government", "us_foreign_policy"),
    "security_studies": ("law_government", "security_studies"),
    "high_school_us_history": ("history", "us_history"),
    "high_school_world_history": ("history", "world_history"),
    "high_school_european_history": ("history", "european_history"),
    "prehistory": ("history", "prehistory"),
    "global_facts": ("geography_global_facts", "global_facts"),
    "high_school_geography": ("geography_global_facts", "geography"),
    "world_religions": ("humanities_culture", "world_religions"),
    "philosophy": ("humanities_culture", "philosophy"),
    "moral_disputes": ("humanities_culture", "moral_disputes"),
    "moral_scenarios": ("humanities_culture", "moral_scenarios"),
    "sociology": ("social_science", "sociology"),
    "professional_psychology": ("social_science", "professional_psychology"),
    "miscellaneous": ("general_knowledge", "miscellaneous"),
    "triviaqa": ("general_knowledge", "triviaqa"),
    "arc_science": ("science", "arc_science"),
}


def build_curated_dataset(repo_root: Path) -> tuple[Path, Path]:
    """Build curated JSONL and manifest files."""

    rows: list[dict[str, Any]] = []
    skipped_sources: list[str] = []
    included_sources: list[dict[str, Any]] = []
    for source_dataset, candidate_paths in SOURCE_FILES:
        rel_path = _first_existing_path(repo_root, candidate_paths)
        if rel_path is None:
            skipped_sources.append(source_dataset)
            continue
        path = repo_root / rel_path
        tasks = load_tasks(path)
        rows.extend(_curated_rows(source_dataset, tasks, rel_path))
        included_sources.append({"source_dataset": source_dataset, "source_cache_path": rel_path, "rows": len(tasks)})

    out_dir = repo_root / "datasets" / "curated"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / f"{CURATED_VERSION}.jsonl"
    manifest_path = out_dir / f"{CURATED_VERSION}_manifest.json"
    dataset_path.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(_manifest(rows, included_sources, skipped_sources), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return dataset_path, manifest_path


def _first_existing_path(repo_root: Path, candidate_paths: list[str]) -> str | None:
    for rel_path in candidate_paths:
        if (repo_root / rel_path).exists():
            return rel_path
    return None


def _curated_rows(source_dataset: str, tasks: list[EvalTask], source_path: str) -> list[dict[str, Any]]:
    rows = []
    source_spec = get_dataset_spec(source_dataset)
    for task in tasks:
        task = _enrich_task_metadata(_enrich_choice_references(task), source_spec)
        category, subcategory = _taxonomy(task)
        metadata = dict(task.metadata)
        metadata.update(
            {
                "curated_dataset": CURATED_VERSION,
                "curated_category": category,
                "curated_subcategory": subcategory,
                "source_dataset": source_dataset,
                "source_task_id": task.task_id,
                "source_category": task.category,
                "source_subcategory": task.subcategory,
                "source_cache_path": source_path,
                "normalization_version": CURATED_VERSION,
                "integrity_note": "Question, expected_output, and source references copied from normalized source task.",
            }
        )
        rows.append(
            {
                "task_id": f"{CURATED_VERSION}_{task.task_id}",
                "capability": task.capability.value,
                "input_text": task.input_text,
                "expected_output": task.expected_output,
                "category": category,
                "subcategory": subcategory,
                "references": task.references,
                "incorrect_references": task.incorrect_references,
                "metadata": metadata,
            }
        )
    return rows


def _taxonomy(task: EvalTask) -> tuple[str, str]:
    source_category = str(task.category or "general_knowledge")
    return DOMAIN_MAP.get(source_category, ("general_knowledge", source_category))


def _manifest(
    rows: list[dict[str, Any]],
    included_sources: list[dict[str, Any]],
    skipped_sources: list[str],
) -> dict[str, Any]:
    source_counts = Counter(row["metadata"]["source_dataset"] for row in rows)
    category_counts = Counter(row["category"] for row in rows)
    subcategory_counts = Counter(
        f"{row['category']}::{row['subcategory']}" for row in rows if row.get("subcategory")
    )
    return {
        "dataset_key": CURATED_VERSION,
        "display_name": "Curated Knowledge QA v1",
        "row_count": len(rows),
        "capability": "answer_accuracy",
        "selection_policy": (
            "Use the largest available compatible normalized cache for each closed-book answer-accuracy source. "
            "No row sampling is applied during curation; notebook run modes sample safely at evaluation time."
        ),
        "compatible_source_scope": ["mmlu", "triviaqa", "arc"],
        "included_sources": included_sources,
        "source_counts": dict(sorted(source_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "subcategory_counts": dict(sorted(subcategory_counts.items())),
        "skipped_sources": skipped_sources,
        "source_decisions": SOURCE_DECISIONS,
        "integrity_policy": (
            "Curated rows copy normalized source question text, expected_output, and references. "
            "Only taxonomy/provenance metadata is added."
        ),
        "compatible_columns": [
            "task_id",
            "capability",
            "input_text",
            "expected_output",
            "category",
            "subcategory",
            "references",
            "incorrect_references",
            "metadata",
        ],
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path, manifest_path = build_curated_dataset(repo_root)
    print(f"Curated dataset written: {dataset_path}")
    print(f"Curated manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
