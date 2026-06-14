from pathlib import Path

from genai_capability_bench.core.runner import run_from_config


def test_core_demo_runner_creates_outputs():
    output_dir = run_from_config("configs/eval_core_demo.yaml")
    assert Path(output_dir, "results.json").exists()
    assert Path(output_dir, "summary.csv").exists()

