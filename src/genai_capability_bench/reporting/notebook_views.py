"""Notebook-friendly display table builders."""

from __future__ import annotations

import os
import pandas as pd

from genai_capability_bench.core.schemas import Capability
from genai_capability_bench.datasets.registry import list_dataset_specs


def dataset_catalog_table(capability: Capability | None = None) -> pd.DataFrame:
    """Return a concise dataset catalog for notebooks."""

    rows = []
    for spec in list_dataset_specs(capability):
        rows.append(
            {
                "Dataset Key": spec.key,
                "Name": spec.display_name,
                "Source": spec.source_type,
                "Default Split": spec.default_split,
                "Primary Capability": spec.capability.value,
                "Task Format": spec.task_format,
                "Scoring Guidance": spec.scoring_guidance,
                "Context Needed": "Yes" if spec.requires_context else "No",
                "Description": spec.description,
                "Notes": spec.notes,
            }
        )
    return pd.DataFrame(rows)


def model_config_table(models) -> pd.DataFrame:
    """Return model configuration display table."""

    return pd.DataFrame(
        [
            {
                "Name": m.name,
                "Provider": m.provider,
                "Model / Deployment": m.model,
                "API Version": m.api_version or "",
                "Temperature": m.temperature,
                "Max Tokens": m.max_tokens,
            }
            for m in models
        ]
    )


def provider_environment_table() -> pd.DataFrame:
    """Display provider context without leaking secrets."""

    base_url = os.environ.get("OPENAI_BASE_URL", "")
    api_version = os.environ.get("OPENAI_API_VERSION", "")
    apim_header = os.environ.get("OPENAI_APIM_HEADER_NAME", "")
    return pd.DataFrame(
        [
            {"Field": "Provider mode", "Value": "Azure OpenAI" if api_version else "OpenAI-compatible"},
            {"Field": "Base URL", "Value": _redact_url(base_url)},
            {"Field": "API version", "Value": api_version or "(blank / non-Azure mode)"},
            {"Field": "APIM header configured", "Value": "Yes" if apim_header else "No"},
            {"Field": "API key present", "Value": "Yes" if os.environ.get("OPENAI_API_KEY") else "No"},
        ]
    )


def judge_config_table() -> pd.DataFrame:
    """Display optional judge model context."""

    return pd.DataFrame(
        [
            {"Field": "Judge model", "Value": os.environ.get("OPENAI_JUDGE_MODEL", "(not set)")},
            {"Field": "Judge purpose", "Value": "Optional review for ambiguous, low-score, or metric-disagreement cases"},
            {"Field": "Default mode", "Value": "Disabled unless ENABLE_JUDGE_REVIEW=True"},
        ]
    )


def dataset_preset_table() -> pd.DataFrame:
    """Plain-language dataset presets for general audiences."""

    return pd.DataFrame(
        [
            {
                "Preset": "local_smoke",
                "Datasets": "answer_accuracy_sample",
                "When to use": "First run, no internet, no API cost, workflow validation.",
            },
            {
                "Preset": "broad_knowledge",
                "Datasets": "mmlu",
                "When to use": "Broad subject benchmark across STEM, law, medicine, humanities, business.",
            },
            {
                "Preset": "open_domain_qa",
                "Datasets": "triviaqa, natural_questions",
                "When to use": "Open-domain factual QA with real or trivia-style questions.",
            },
            {
                "Preset": "science_exam",
                "Datasets": "arc",
                "When to use": "Science multiple-choice QA and exam-style factual reasoning.",
            },
            {
                "Preset": "context_grounded",
                "Datasets": "squad",
                "When to use": "Reading comprehension over supplied context; better for future context/RAG workflows.",
            },
            {
                "Preset": "multi_hop",
                "Datasets": "hotpotqa",
                "When to use": "Multi-hop questions; useful but should be interpreted alongside reasoning/RAG metrics.",
            },
            {
                "Preset": "custom_golden_set",
                "Datasets": "custom",
                "When to use": "Enterprise/domain-specific golden sets.",
            },
        ]
    )


def html_summary_report(run, target_label: str, judge_enabled: bool = False) -> str:
    """Build an in-notebook HTML executive report."""

    flagged = int((run.diagnostics_df["review_priority"] != "Looks good").sum()) if not run.diagnostics_df.empty else 0
    total = len(run.results_df)
    avg_score = float(run.results_df["score"].mean()) if total else 0.0
    pass_rate = float(run.results_df["passed"].mean()) if total else 0.0
    dataset_rows = "".join(
        f"<tr><td>{r.dataset_key}</td><td>{r.split}</td><td>{r.answer_accuracy_tasks}</td><td>{r.categories}</td></tr>"
        for r in run.dataset_manifest_df.itertuples(index=False)
    )
    summary_rows = "".join(
        f"<tr><td>{r.dataset_key}</td><td>{r.category}</td><td>{r.n}</td><td>{r.avg_score:.2f}</td><td>{r.pass_rate:.2f}</td></tr>"
        for r in run.dataset_summary_df.itertuples(index=False)
    )
    return f"""
    <div style="border:1px solid #d0d7de;border-radius:8px;padding:18px;margin:12px 0;font-family:Arial, sans-serif;">
      <h2 style="margin-top:0;">Answer Accuracy Executive Report</h2>
      <p><strong>Run ID:</strong> <code>{run.run_id}</code></p>
      <p><strong>Target model:</strong> {target_label}</p>
      <p><strong>Judge review:</strong> {"Enabled" if judge_enabled else "Disabled"}</p>
      <div style="display:flex;gap:12px;flex-wrap:wrap;margin:14px 0;">
        <div style="background:#eff6ff;padding:12px;border-radius:8px;min-width:140px;"><strong>{total}</strong><br/>Responses</div>
        <div style="background:#ecfdf5;padding:12px;border-radius:8px;min-width:140px;"><strong>{avg_score:.2f}</strong><br/>Average Score</div>
        <div style="background:#fefce8;padding:12px;border-radius:8px;min-width:140px;"><strong>{pass_rate:.0%}</strong><br/>Pass Rate</div>
        <div style="background:#fff7ed;padding:12px;border-radius:8px;min-width:140px;"><strong>{flagged}</strong><br/>Flagged Cases</div>
      </div>
      <h3>Dataset Manifest</h3>
      <table style="border-collapse:collapse;width:100%;">
        <tr><th align="left">Dataset</th><th align="left">Split</th><th align="right">Tasks</th><th align="left">Categories</th></tr>
        {dataset_rows}
      </table>
      <h3>Dataset / Category Summary</h3>
      <table style="border-collapse:collapse;width:100%;">
        <tr><th align="left">Dataset</th><th align="left">Category</th><th align="right">N</th><th align="right">Avg Score</th><th align="right">Pass Rate</th></tr>
        {summary_rows}
      </table>
      <p style="margin-top:14px;"><strong>Interpretation:</strong> {run.summary_text}</p>
    </div>
    """


def _redact_url(url: str) -> str:
    if not url:
        return "(not set)"
    if len(url) <= 36:
        return url
    return url[:28] + "..." + url[-8:]
