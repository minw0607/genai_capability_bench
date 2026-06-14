"""Experiment runner for capability evaluations."""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv

from genai_capability_bench.capabilities.registry import get_evaluator
from genai_capability_bench.clients.factory import create_client
from genai_capability_bench.core.schemas import Capability, EvalTask, ModelSpec, RunMetadata
from genai_capability_bench.reporting.tables import summarize_results


def load_config(path: str | Path) -> dict[str, Any]:
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return _expand_env_vars(config)


def _expand_env_vars(value):
    """Recursively expand ${ENV_VAR} placeholders in config values."""

    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_tasks(path: str | Path) -> list[EvalTask]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".csv":
        rows = pd.read_csv(path).to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported task file type: {path.suffix}")

    tasks = []
    for row in rows:
        tasks.append(
            EvalTask(
                task_id=str(row["task_id"]),
                capability=Capability(row["capability"]),
                input_text=str(row["input_text"]),
                expected_output=row.get("expected_output"),
                category=row.get("category", "general"),
                subcategory=row.get("subcategory"),
                references=list(row.get("references") or []),
                incorrect_references=list(row.get("incorrect_references") or []),
                metadata=dict(row.get("metadata") or {}),
            )
        )
    return tasks


def parse_models(config: dict[str, Any]) -> list[ModelSpec]:
    models = []
    for row in config.get("models", []):
        models.append(
            ModelSpec(
                name=row["name"],
                provider=row.get("provider", "mock"),
                model=row.get("model", "mock-model"),
                api_version=row.get("api_version"),
                temperature=row.get("temperature", 0.0),
                max_tokens=int(row.get("max_tokens", 1000)),
                metadata=dict(row.get("metadata") or {}),
            )
        )
    return models


def run_from_config(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    run_id = config.get("run_id") or uuid.uuid4().hex[:12]
    output_dir = Path(config.get("output_dir", "outputs/runs")) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    models = parse_models(config)
    tasks = load_tasks(config["dataset"])
    thresholds = config.get("pass_thresholds", {})

    clients = {model.name: create_client(model) for model in models}
    results = []
    for model in models:
        for task in tasks:
            evaluator = get_evaluator(
                task.capability,
                pass_threshold=float(thresholds.get(task.capability.value, config.get("default_pass_threshold", 0.7))),
            )
            result = evaluator.evaluate_task(run_id, task, model, clients[model.name])
            results.append(result)

    metadata = RunMetadata.create(
        run_id=run_id,
        config_path=str(config_path),
        models=models,
        capabilities=sorted({task.capability for task in tasks}, key=lambda c: c.value),
        notes=config.get("notes", ""),
    )
    result_rows = [r.to_dict() for r in results]
    summary = summarize_results(result_rows)

    (output_dir / "metadata.json").write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "results.json").write_text(json.dumps(result_rows, indent=2), encoding="utf-8")
    pd.DataFrame(result_rows).to_csv(output_dir / "results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    print(f"Run complete: {run_id}")
    print(f"Artifacts: {output_dir}")
    print(summary.to_string(index=False))
    return output_dir


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m genai_capability_bench.core.runner <config.yaml>")
    run_from_config(sys.argv[1])


if __name__ == "__main__":
    main()
