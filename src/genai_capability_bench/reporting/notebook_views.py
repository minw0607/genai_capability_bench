"""Notebook-friendly display table builders."""

from __future__ import annotations

import os
import pandas as pd
from dotenv import load_dotenv

from genai_capability_bench.core.schemas import Capability, ModelSpec
from genai_capability_bench.datasets.registry import get_dataset_spec, list_dataset_specs
from genai_capability_bench.metrics.registry import metric_standards_table, scoring_profiles_table
from genai_capability_bench.reporting.model_labels import model_public_label
from genai_capability_bench.reporting.ratings import assess_answer_accuracy_run, dataset_rollup


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
                "Answer Type": spec.answer_type,
                "Reference Shape": spec.reference_shape,
                "Scoring Profile": spec.scoring_profile,
                "Primary Metrics": ", ".join(spec.primary_metrics),
                "Secondary Metrics": ", ".join(spec.secondary_metrics),
                "Scoring Guidance": spec.scoring_guidance,
                "Context Needed": "Yes" if spec.requires_context else "No",
                "Description": spec.description,
                "Caveats": spec.caveats,
                "Notes": spec.notes,
            }
        )
    return pd.DataFrame(rows)


def metric_standards_display_table() -> pd.DataFrame:
    """Return repo-wide metric standards for notebooks."""

    return metric_standards_table()[
        ["key", "display_name", "role", "definition", "best_for", "limitations", "source"]
    ]


def scoring_profiles_display_table() -> pd.DataFrame:
    """Return repo-wide scoring profiles for notebooks."""

    df = scoring_profiles_table().copy()
    for col in ["primary_metrics", "secondary_metrics", "diagnostic_metrics"]:
        df[col] = df[col].apply(lambda values: ", ".join(values))
    return df


def model_config_table(models) -> pd.DataFrame:
    """Return model configuration display table."""

    return pd.DataFrame(
        [
            {
                "Report Label": model_public_label(m),
                "Provider": m.provider,
                "Raw Deployment Visible in Artifacts": "Yes, in machine-readable result files only",
                "API Version": m.api_version or "",
                "Temperature": m.temperature,
                "Token Parameter": m.token_parameter or "(omitted)",
                "Token Limit": m.max_tokens if m.max_tokens is not None else "(omitted)",
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

    judge_model = os.environ.get("OPENAI_JUDGE_MODEL", "")
    return pd.DataFrame(
        [
            {"Field": "Judge model", "Value": _public_model_name(judge_model) if judge_model else "(not set)"},
            {"Field": "Judge purpose", "Value": "Optional review for ambiguous, low-score, or metric-disagreement cases"},
            {"Field": "Default mode", "Value": "Disabled unless ENABLE_JUDGE_REVIEW=True"},
        ]
    )


def embedding_config_table() -> pd.DataFrame:
    """Display optional embedding model context."""

    load_dotenv()
    embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL") or os.environ.get("EMBEDDING_MODEL", "")
    embedding_value = "Configured; provider-specific deployment hidden" if embedding_model else "(not set)"
    return pd.DataFrame(
        [
            {"Field": "Embedding model", "Value": embedding_value},
            {
                "Field": "Default notebook use",
                "Value": "Local TF-IDF semantic similarity unless USE_API_EMBEDDINGS=True.",
            },
            {
                "Field": "API embedding mode",
                "Value": "Uses the configured OpenAI-compatible embedding deployment for semantic cosine similarity.",
            },
        ]
    )


def dataset_preset_table() -> pd.DataFrame:
    """Plain-language active dataset presets for the closed-book answer notebook."""

    return pd.DataFrame(
        [
            {
                "Preset": "local_smoke",
                "Datasets": "answer_accuracy_sample",
                "Suggested sample": "10 (full local sample)",
                "Scale ceiling": "Full local sample shipped with repo",
                "When to use": "First run, no internet, no API cost, workflow validation.",
            },
            {
                "Preset": "curated_knowledge",
                "Datasets": "curated_knowledge_v1",
                "Suggested sample": "25-100 first; 200+ for stronger reporting",
                "Scale ceiling": "Full curated local dataset, budget permitting",
                "When to use": "Recommended default: broad source-preserving closed-book knowledge benchmark.",
            },
            {
                "Preset": "knowledge_portfolio",
                "Datasets": "mmlu, triviaqa, natural_questions, arc",
                "Suggested sample": "25-100 per dataset first; scale only after scoring review",
                "Scale ceiling": "Full selected benchmark splits",
                "When to use": "Portfolio-style closed-book answer accuracy across multiple dataset shapes.",
            },
            {
                "Preset": "broad_knowledge",
                "Datasets": "mmlu",
                "Suggested sample": "25-100 for first pass; scale by subject after validation",
                "Scale ceiling": "Full selected benchmark split",
                "When to use": "Broad subject benchmark across STEM, law, medicine, humanities, business.",
            },
            {
                "Preset": "open_domain_qa",
                "Datasets": "triviaqa, natural_questions",
                "Suggested sample": "25-100 per dataset first; larger runs can be expensive",
                "Scale ceiling": "Full selected benchmark splits",
                "When to use": "Open-domain factual QA with real or trivia-style questions.",
            },
            {
                "Preset": "science_exam",
                "Datasets": "arc",
                "Suggested sample": "25-100 first; full split after grader validation",
                "Scale ceiling": "Full selected benchmark split",
                "When to use": "Science multiple-choice QA and exam-style factual reasoning.",
            },
            {
                "Preset": "custom_golden_set",
                "Datasets": "custom",
                "Suggested sample": "All curated items if small; stratified sample if large",
                "Scale ceiling": "Provided custom file",
                "When to use": "Enterprise/domain-specific golden sets.",
            },
        ]
    )


def selected_dataset_plan_table(
    dataset_keys: list[str],
    *,
    preset_name: str,
    sample_size_per_dataset: int | None,
    sample_strategy: str | None = None,
    dataset_splits: dict[str, str | None] | None = None,
    selected_categories: list[str] | str = "ALL",
) -> pd.DataFrame:
    """Show the active dataset plan before loading benchmark data."""

    rows = []
    dataset_splits = dataset_splits or {}
    sample_text = "Full selected split / full file" if sample_size_per_dataset is None else str(sample_size_per_dataset)
    category_text = "All normalized categories" if selected_categories == "ALL" else ", ".join(selected_categories)
    for key in dataset_keys:
        spec = get_dataset_spec(key)
        resolved_strategy = sample_strategy or spec.default_sample_strategy
        rows.append(
            {
                "Preset": preset_name,
                "Dataset Key": key,
                "Benchmark Role": spec.description,
                "Source": spec.source_type,
                "Split": dataset_splits.get(key) or spec.default_split,
                "Requested Sample": sample_text,
                "Sampling Strategy": resolved_strategy,
                "Category Scope": category_text,
                "Task Format": spec.task_format,
                "Scoring Profile": spec.scoring_profile,
                "Reference Shape": spec.reference_shape,
                "Context Required": "Yes" if spec.requires_context else "No",
                "Scoring Caveat": spec.scoring_guidance,
            }
        )
    return pd.DataFrame(rows)


def sample_size_guidance_table() -> pd.DataFrame:
    """Professional sample-size guidance for benchmark planning."""

    return pd.DataFrame(
        [
            {
                "Run Type": "Smoke test",
                "Sample Size": "5-10 total",
                "Purpose": "Validate credentials, dataset loading, scoring, artifacts.",
            },
            {
                "Run Type": "Notebook demo",
                "Sample Size": "10-25 per selected dataset",
                "Purpose": "Fast, readable demonstration with visible diagnostics.",
            },
            {
                "Run Type": "Exploratory benchmark",
                "Sample Size": "50-100 per selected dataset",
                "Purpose": "Compare models and identify weak categories before a larger run.",
            },
            {
                "Run Type": "Evaluation report",
                "Sample Size": "200+ or full split, depending on cost and dataset size",
                "Purpose": "Produce stronger evidence for model selection or governance review.",
            },
            {
                "Run Type": "Custom golden set",
                "Sample Size": "Prefer all curated examples if the set is intentionally small",
                "Purpose": "Measure business/domain capability on high-value questions.",
            },
        ]
    )


def html_summary_report(run, target_label: str, judge_enabled: bool = False) -> str:
    """Build an in-notebook HTML executive report."""

    flagged = int((run.diagnostics_df["review_priority"] != "Looks good").sum()) if not run.diagnostics_df.empty else 0
    total = len(run.results_df)
    avg_score = float(run.results_df["score"].mean()) if total else 0.0
    pass_rate = float(run.results_df["passed"].mean()) if total else 0.0
    flag_rate = flagged / total if total else 0.0
    assessment = assess_answer_accuracy_run(
        results_df=run.results_df,
        diagnostics_df=run.diagnostics_df,
        dataset_manifest_df=run.dataset_manifest_df,
    )
    reliability_notes = getattr(run, "reliability_notes", []) or []
    reliability_rows = "".join(f"<li style='margin:4px 0;'>{note}</li>" for note in reliability_notes)
    portfolio_df = dataset_rollup(run.dataset_summary_df)
    portfolio_rows = "".join(
        f"<tr style='background:{'#FAFAFA' if i % 2 == 0 else '#FFFFFF'};border-bottom:1px solid #ECEFF1;'>"
        f"<td style='padding:7px 10px;font-weight:600;'>{r.dataset_key}</td>"
        f"<td style='padding:7px 10px;text-align:right;'>{r.n}</td>"
        f"<td style='padding:7px 10px;text-align:right;'>{r.categories}</td>"
        f"<td style='padding:7px 10px;text-align:right;font-weight:700;'>{r.avg_score:.2f}</td>"
        f"<td style='padding:7px 10px;text-align:right;'>{r.pass_rate:.0%}</td>"
        f"<td style='padding:7px 10px;text-align:center;'>{_capability_badge(_capability_rating(float(r.pass_rate), float(r.avg_score)))}</td></tr>"
        for i, r in enumerate(portfolio_df.itertuples(index=False))
    )
    dataset_rows = "".join(
        f"<tr style='background:{'#FAFAFA' if i % 2 == 0 else '#FFFFFF'};border-bottom:1px solid #ECEFF1;'>"
        f"<td style='padding:7px 10px;font-weight:600;'>{r.dataset_key}</td>"
        f"<td style='padding:7px 10px;'>{r.split}</td>"
        f"<td style='padding:7px 10px;text-align:right;'>{r.answer_accuracy_tasks}</td>"
        f"<td style='padding:7px 10px;'>{r.scoring_profile}</td>"
        f"<td style='padding:7px 10px;'>{r.reference_shape}</td></tr>"
        for i, r in enumerate(run.dataset_manifest_df.itertuples(index=False))
    )
    performance_rows = "".join(
        f"<tr style='background:{'#FAFAFA' if i % 2 == 0 else '#FFFFFF'};border-bottom:1px solid #ECEFF1;'>"
        f"<td style='padding:7px 10px;font-weight:600;'>{r.dataset_key}</td>"
        f"<td style='padding:7px 10px;'>{r.category}</td>"
        f"<td style='padding:7px 10px;text-align:right;'>{r.n}</td>"
        f"<td style='padding:7px 10px;text-align:right;font-weight:700;'>{r.avg_score:.2f}</td>"
        f"<td style='padding:7px 10px;text-align:right;'>{r.pass_rate:.0%}</td>"
        f"<td style='padding:7px 10px;text-align:center;'>{_capability_badge(_capability_rating(float(r.pass_rate), float(r.avg_score)))}</td></tr>"
        for i, r in enumerate(run.dataset_summary_df.itertuples(index=False))
    )
    finding_rows = "".join(_finding_card(level, title, body) for level, title, body in _html_finding_cards(run))
    artifact_rows = "".join(
        f"<tr style='border-bottom:1px solid #ECEFF1;'>"
        f"<td style='padding:6px 10px;font-weight:600;'>{category}</td>"
        f"<td style='padding:6px 10px;'>{description}</td>"
        f"<td style='padding:6px 10px;'><code>{path.name}</code></td></tr>"
        for category, description, path in _report_artifacts(getattr(run, "artifact_paths", {}))
        if path is not None
    )
    capability_dark = _capability_color(assessment.capability_rating)
    reliability_dark = _reliability_color(assessment.reliability_rating)
    reliability_light = _reliability_light_color(assessment.reliability_rating)
    high_priority = _priority_count(run, "High review priority")
    near_threshold = _priority_count(run, "Near threshold")
    judge_failures = _judge_failure_count_html(run)
    generated = pd.Timestamp.now().strftime("%Y-%m-%d")
    return f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:940px;margin:0 auto 18px;border:1px solid #CFD8DC;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.08);background:white;color:#263238;">
      <div style="background:#263238;color:white;padding:20px 28px 16px;">
        <div style="font-size:11px;letter-spacing:1.5px;text-transform:uppercase;opacity:.72;margin-bottom:4px;">Confidential — Capability Evaluation</div>
        <div style="font-size:22px;font-weight:700;letter-spacing:-.3px;">Answer Accuracy Assessment — Executive Summary</div>
        <div style="margin-top:8px;font-size:12px;opacity:.78;">Target: <strong>{target_label}</strong> &nbsp;|&nbsp; Datasets: <strong>{_dataset_label(run)}</strong> &nbsp;|&nbsp; n = <strong>{total} responses</strong> &nbsp;|&nbsp; Date: <strong>{generated}</strong></div>
      </div>
      <div style="background:{reliability_light};border-left:6px solid {reliability_dark};padding:14px 24px;display:grid;gap:10px;">
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
          <div style="background:{capability_dark};color:white;font-size:13px;font-weight:700;padding:6px 16px;border-radius:4px;white-space:nowrap;letter-spacing:.5px;">CAPABILITY: {assessment.capability_rating.upper()}</div>
          <div style="background:{reliability_dark};color:white;font-size:13px;font-weight:700;padding:6px 16px;border-radius:4px;white-space:nowrap;letter-spacing:.5px;">EVALUATION RELIABILITY: {assessment.reliability_rating.upper()}</div>
          <div style="background:#455A64;color:white;font-size:13px;font-weight:700;padding:6px 16px;border-radius:4px;white-space:nowrap;letter-spacing:.5px;">POSTURE: {assessment.review_posture.upper()}</div>
        </div>
        <div style="font-size:13.5px;color:#333;line-height:1.6;">{_executive_narrative_from_assessment(assessment, pass_rate, avg_score, flag_rate, reliability_notes, judge_failures)}</div>
      </div>
      <div style="padding:20px 28px;">
        <div style="margin:18px 0 10px;">
          {_section_title('Testing Scope')}
          <div style="display:flex;background:#FAFAFA;border:1px solid #ECEFF1;border-radius:6px;overflow:hidden;margin-bottom:8px;flex-wrap:wrap;">
            {_metric_tile(len(run.dataset_manifest_df), 'Datasets')}
            {_metric_tile(total, 'Responses')}
            {_metric_tile(f'{avg_score:.2f}', 'Avg Score')}
            {_metric_tile(f'{pass_rate:.0%}', 'Pass Rate', capability_dark)}
            {_metric_tile(high_priority, 'High-Priority Flags', '#C62828' if high_priority else '#263238')}
            {_metric_tile(near_threshold, 'Near Threshold')}
          </div>
          <div style="font-size:12px;color:#546E7A;line-height:1.6;">Run provenance: {_html_provenance(run)}. Judge review: {'enabled' if judge_enabled else 'disabled'}.</div>
        </div>
        <div style="margin:18px 0 10px;">
          {_section_title('Key Findings')}
          <div style="display:grid;gap:8px;">{finding_rows}</div>
        </div>
        <div style="margin:18px 0 10px;">
          {_section_title('Dataset Portfolio')}
          <table style="width:100%;border-collapse:collapse;font-size:12px;">
            <tr style="background:#37474F;color:white;"><th style="padding:7px 10px;text-align:left;">Dataset</th><th style="padding:7px 10px;text-align:right;">N</th><th style="padding:7px 10px;text-align:right;">Categories</th><th style="padding:7px 10px;text-align:right;">Avg Score</th><th style="padding:7px 10px;text-align:right;">Pass Rate</th><th style="padding:7px 10px;text-align:center;">Capability</th></tr>
            {portfolio_rows}
          </table>
        </div>
        <div style="margin:18px 0 10px;">
          {_section_title('Category Results')}
          <table style="width:100%;border-collapse:collapse;font-size:12px;">
            <tr style="background:#37474F;color:white;"><th style="padding:7px 10px;text-align:left;">Dataset</th><th style="padding:7px 10px;text-align:left;">Category</th><th style="padding:7px 10px;text-align:right;">N</th><th style="padding:7px 10px;text-align:right;">Avg Score</th><th style="padding:7px 10px;text-align:right;">Pass Rate</th><th style="padding:7px 10px;text-align:center;">Capability</th></tr>
            {performance_rows}
          </table>
        </div>
        <div style="margin:18px 0 10px;">
          {_section_title('Evidence Base')}
          <table style="width:100%;border-collapse:collapse;font-size:12px;">
            <tr style="background:#37474F;color:white;"><th style="padding:7px 10px;text-align:left;">Dataset</th><th style="padding:7px 10px;text-align:left;">Split</th><th style="padding:7px 10px;text-align:right;">Tasks</th><th style="padding:7px 10px;text-align:left;">Scoring Profile</th><th style="padding:7px 10px;text-align:left;">Reference Shape</th></tr>
            {dataset_rows}
          </table>
        </div>
        <div style="margin:18px 0 10px;">
          {_section_title('Evaluation Methodology')}
          <div style="display:grid;gap:6px;">
            {_methodology_row('Scoring', 'Each dataset uses its registered scoring profile. The pass/fail decision is based on the profile primary score against the configured threshold.')}
            {_methodology_row('Metrics', 'Exact match, token F1, ROUGE-L, BLEU, and semantic similarity are retained as supporting signals rather than treated as interchangeable truth.')}
            {_methodology_row('Judge Review', 'LLM judge review is an adjudication aid for flagged cases. It should be calibrated and monitored rather than used as an unqualified replacement for deterministic metrics.')}
            {_methodology_row('Checkpointing', 'Compatible checkpoints allow repeatable reporting without rerunning expensive target-model calls; reports identify when replay is used.')}
          </div>
        </div>
        <div style="margin:18px 0 10px;">
          {_section_title('Governance Alignment')}
          <div style="background:#E8EAF6;border-left:4px solid #3949AB;border-radius:0 6px 6px 0;padding:12px 16px;">
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;">
              {_governance_badge('NIST AI RMF')}
              {_governance_badge('NIST AI 600-1')}
              {_governance_badge('EU AI Act')}
              {_governance_badge('ISO/IEC 42001')}
            </div>
            <div style="font-size:13px;color:#263238;line-height:1.7;">The evidence supports capability monitoring, data-quality review, and model-performance documentation. For curated benchmark runs, source/category stratification and flagged-case review are particularly relevant to measurement validity: before results are used for model selection or governance evidence, the benchmark should distinguish model errors from source-label, reference-shape, and metric-fit limitations.</div>
          </div>
        </div>
        <div style="margin:18px 0 10px;">
          {_section_title('Recommended Actions')}
          <ol style="margin:0;padding-left:20px;display:grid;gap:8px;">
            {_action_item('Review source-label and metric false-negative candidates', 'Inspect judge-rescued, near-threshold, and high-overlap failures before treating deterministic pass rate as final model quality.')}
            {_action_item('Use judge review selectively for metric false negatives', 'Prioritize near-threshold and reference-shape-warning cases, and confirm provider parameters before relying on judge-assisted findings.')}
            {_action_item('Interpret curated slices separately', 'Report source and broad-category performance alongside the headline score so MMLU, TriviaQA, and ARC evidence are not collapsed into one opaque number.')}
            {_action_item('Retest with a fresh run when methodology changes', 'Disable auto-resume or bump the scoring method version after changing metrics, prompts, judge settings, or dataset normalization.')}
          </ol>
        </div>
        <div style="margin:18px 0 10px;">
          {_section_title('Analytical Note')}
          <div style="background:#FFF8E1;border:1px solid #FFE082;border-radius:6px;padding:10px 14px;font-size:12.5px;color:#37474F;line-height:1.6;"><strong>Measurement caveat:</strong> Curated benchmark scores combine multiple source datasets and task formats. Treat the headline score as directional evidence, then use source/category slices and flagged-case diagnostics for final interpretation.</div>
        </div>
        <div style="margin:18px 0 10px;">
          {_section_title('Saved Artifacts')}
          <table style="width:100%;border-collapse:collapse;font-size:12px;">
            <tr style="background:#37474F;color:white;"><th style="padding:7px 10px;text-align:left;">Category</th><th style="padding:7px 10px;text-align:left;">Purpose</th><th style="padding:7px 10px;text-align:left;">File</th></tr>
            {artifact_rows}
          </table>
        </div>
      </div>
      <div style="background:#ECEFF1;padding:10px 28px;font-size:11px;color:#607D8B;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;">
        <span>Generated {generated} · GenAI Capability Bench · Answer Accuracy</span>
        <span>Metrics computed deterministically · Judge review optional</span>
      </div>
    </div>
    """


def _section_title(title: str) -> str:
    return (
        "<div style=\"font-size:13px;font-weight:700;color:#37474F;text-transform:uppercase;"
        "letter-spacing:.8px;border-bottom:2px solid #ECEFF1;padding-bottom:5px;margin-bottom:10px;\">"
        f"{title}</div>"
    )


def _metric_tile(value, label: str, color: str = "#263238") -> str:
    return (
        "<div style=\"text-align:center;padding:10px 16px;border-right:1px solid #ECEFF1;min-width:110px;\">"
        f"<div style=\"font-size:25px;font-weight:800;color:{color};\">{value}</div>"
        f"<div style=\"font-size:11px;color:#607D8B;margin-top:2px;text-transform:uppercase;letter-spacing:.5px;\">{label}</div>"
        "</div>"
    )


def _finding_card(level: str, title: str, body: str) -> str:
    dark = _risk_dark_color(level)
    light = _risk_light_color(level)
    text_color = "#333" if level == "Low" else "white"
    return (
        f"<div style=\"background:{light};border-left:4px solid {dark};border-radius:0 6px 6px 0;padding:10px 14px;\">"
        "<div style=\"display:flex;align-items:center;gap:8px;margin-bottom:4px;\">"
        f"<span style=\"display:inline-block;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700;"
        f"letter-spacing:.5px;background:{dark};color:{text_color};\">{level.upper()}</span>"
        f"<span style=\"font-size:13px;font-weight:700;color:#263238;\">{title}</span></div>"
        f"<div style=\"font-size:12.5px;color:#37474F;line-height:1.6;\">{body}</div></div>"
    )


def _methodology_row(label: str, body: str) -> str:
    return (
        "<div style=\"display:flex;gap:10px;align-items:flex-start;padding:8px 12px;background:#FAFAFA;"
        "border-radius:6px;border-left:3px solid #455A64;\">"
        f"<div style=\"min-width:95px;font-size:11px;font-weight:700;color:#455A64;text-transform:uppercase;"
        f"letter-spacing:.5px;padding-top:1px;\">{label}</div>"
        f"<div style=\"font-size:12.5px;color:#546E7A;line-height:1.55;\">{body}</div></div>"
    )


def _governance_badge(label: str) -> str:
    return (
        "<span style=\"background:#3949AB;color:white;font-size:11px;font-weight:700;"
        f"padding:3px 10px;border-radius:4px;\">{label}</span>"
    )


def _action_item(title: str, body: str) -> str:
    return (
        "<li style=\"padding:8px 0 8px 6px;\">"
        f"<div style=\"font-size:13px;font-weight:700;color:#263238;\">{title}</div>"
        f"<div style=\"font-size:12.5px;color:#546E7A;margin-top:3px;line-height:1.6;\">{body}</div></li>"
    )


def _risk_badge(level: str) -> str:
    dark = _risk_dark_color(level)
    text_color = "#333" if level == "Low" else "white"
    return (
        f"<span style=\"display:inline-block;padding:2px 10px;border-radius:4px;font-size:11px;"
        f"font-weight:700;letter-spacing:.5px;background:{dark};color:{text_color};\">{level.upper()}</span>"
    )


def _capability_badge(level: str) -> str:
    dark = _capability_color(level)
    return (
        f"<span style=\"display:inline-block;padding:2px 10px;border-radius:4px;font-size:11px;"
        f"font-weight:700;letter-spacing:.5px;background:{dark};color:white;\">{level.upper()}</span>"
    )


def _capability_color(level: str) -> str:
    return {
        "Strong": "#166534",
        "Moderate-Strong": "#0f766e",
        "Mixed": "#b45309",
        "Weak": "#b91c1c",
    }.get(level, "#607D8B")


def _reliability_color(level: str) -> str:
    return {"High": "#166534", "Medium": "#b45309", "Low": "#b91c1c"}.get(level, "#607D8B")


def _reliability_light_color(level: str) -> str:
    return {"High": "#ecfdf5", "Medium": "#fffbeb", "Low": "#fff1f2"}.get(level, "#ECEFF1")


def _capability_rating(pass_rate: float, avg_score: float) -> str:
    if pass_rate >= 0.85 and avg_score >= 0.80:
        return "Strong"
    if pass_rate >= 0.70 and avg_score >= 0.65:
        return "Moderate-Strong"
    if pass_rate >= 0.50 and avg_score >= 0.50:
        return "Mixed"
    return "Weak"


def _risk_dark_color(level: str) -> str:
    return {"High": "#C62828", "Medium": "#E65100", "Low": "#F9A825"}.get(level, "#607D8B")


def _risk_light_color(level: str) -> str:
    return {"High": "#FFEBEE", "Medium": "#FFF3E0", "Low": "#FFFDE7"}.get(level, "#ECEFF1")


def _dataset_rating(pass_rate: float, avg_score: float) -> str:
    if pass_rate < 0.60 or avg_score < 0.60:
        return "High"
    if pass_rate < 0.80 or avg_score < 0.75:
        return "Medium"
    return "Low"


def _priority_count(run, priority: str) -> int:
    if "review_priority" not in run.diagnostics_df:
        return 0
    return int((run.diagnostics_df["review_priority"] == priority).sum())


def _dataset_label(run) -> str:
    if run.dataset_manifest_df.empty or "dataset_key" not in run.dataset_manifest_df:
        return "selected benchmark datasets"
    return ", ".join(run.dataset_manifest_df["dataset_key"].astype(str).tolist())


def _report_artifacts(artifact_paths: dict) -> list[tuple[str, str, object]]:
    catalog = [
        ("Executive reporting", "Styled HTML report for leadership review", "executive_summary_html"),
        ("Technical reporting", "Markdown memo for source control and text review", "technical_report_md"),
        ("Raw outputs", "Per-question model responses and normalized scores", "raw_results_csv"),
        ("Diagnostics", "Flag reasons, metric details, and review-priority fields", "diagnostics_csv"),
        ("Dataset performance", "Dataset/category-level score and pass-rate summary", "dataset_summary_csv"),
        ("Dataset manifest", "Dataset source, split, reference shape, and scoring profile", "dataset_manifest_csv"),
        ("Checkpoint", "Resumable JSONL checkpoint for interrupted or replayed runs", "checkpoint_jsonl"),
    ]
    return [(category, description, artifact_paths[key]) for category, description, key in catalog if key in artifact_paths]


def _executive_narrative(
    risk_rating: str,
    pass_rate: float,
    avg_score: float,
    flag_rate: float,
    reliability_notes: list[str],
    judge_failures: int,
) -> str:
    if risk_rating == "High":
        base = (
            f"The run shows a material interpretation risk: pass rate is {pass_rate:.0%}, average score is "
            f"{avg_score:.2f}, and {flag_rate:.0%} of cases require review. The result should not be treated "
            "as a final model-quality conclusion until dataset/reference calibration issues are resolved."
        )
    elif risk_rating == "Medium":
        base = (
            f"The run provides mixed evidence: pass rate is {pass_rate:.0%} with an average score of {avg_score:.2f}. "
            "The model shows useful capability signal, but flagged cases should be reviewed before model selection."
        )
    else:
        base = (
            f"The run provides directionally strong evidence for this scope, with pass rate {pass_rate:.0%} "
            f"and average score {avg_score:.2f}. Continue expanding coverage before production reliance."
        )
    if reliability_notes:
        base += " Key caveat: " + reliability_notes[-1]
    if judge_failures:
        base += f" Judge review also failed for {judge_failures} attempted case(s), so judge-assisted findings are incomplete."
    return base


def _executive_narrative_from_assessment(
    assessment,
    pass_rate: float,
    avg_score: float,
    flag_rate: float,
    reliability_notes: list[str],
    judge_failures: int,
) -> str:
    base = (
        f"The run shows {assessment.capability_rating.lower()} answer-accuracy capability for the selected scope "
        f"(pass rate {pass_rate:.0%}, average score {avg_score:.2f}). Measurement reliability is "
        f"{assessment.reliability_rating.lower()}: {assessment.reliability_rationale}"
    )
    if flag_rate > 0:
        base += f" {flag_rate:.0%} of cases were flagged for review."
    if reliability_notes:
        base += " Additional caveat: " + reliability_notes[-1]
    if judge_failures:
        base += f" Judge review failed for {judge_failures} attempted case(s), so judge-assisted findings are incomplete."
    return base


def _html_finding_cards(run) -> list[tuple[str, str, str]]:
    findings: list[tuple[str, str, str]] = []
    if not run.dataset_summary_df.empty:
        strongest = run.dataset_summary_df.sort_values("avg_score", ascending=False).iloc[0]
        weakest = run.dataset_summary_df.sort_values("avg_score", ascending=True).iloc[0]
        findings.append(
            (
                "Medium" if float(strongest.pass_rate) < 0.80 else "Low",
                f"{strongest.dataset_key} provides the cleanest positive signal",
                (
                    f"{strongest.dataset_key} achieved an average score of {strongest.avg_score:.2f} "
                    f"and pass rate of {strongest.pass_rate:.0%}. This slice is more interpretable because "
                    "the reference shape is closer to concise answer generation."
                ),
            )
        )
        findings.append(
            (
                "High",
                f"{weakest.dataset_key} drives the headline weakness",
                (
                    f"{weakest.dataset_key} scored {weakest.avg_score:.2f} on average with a {weakest.pass_rate:.0%} "
                    "pass rate. This should be treated as a calibration warning when the dataset uses long "
                    "passage-style references but the model produces concise answers."
                ),
            )
        )
    high_priority = _priority_count(run, "High review priority")
    if high_priority:
        findings.append(
            (
                "High" if high_priority / max(len(run.diagnostics_df), 1) > 0.30 else "Medium",
                "Manual review burden is material",
                (
                    f"The diagnostics flagged {high_priority} high-priority cases. This is large enough to affect "
                    "review cost, triage workflow, and confidence in a single aggregate score."
                ),
            )
        )
    contains_only = int(run.diagnostics_df.get("contains_only_credit", pd.Series(dtype=bool)).fillna(False).sum())
    if contains_only:
        findings.append(
            (
                "Medium",
                "Simple contains-match would over-credit some answers",
                (
                    f"{contains_only} case(s) contain a reference string but still fail the profile score. "
                    "This supports keeping contains-match as a diagnostic, not a primary scoring rule."
                ),
            )
        )
    judge_failures = _judge_failure_count_html(run)
    if judge_failures:
        findings.append(
            (
                "Medium",
                "Judge review configuration needs correction",
                (
                    f"Judge review was attempted for {judge_failures} flagged case(s), but no valid score was returned. "
                    "Provider parameter compatibility should be resolved before using judge-assisted conclusions."
                ),
            )
        )
    return findings or [("Low", "No major diagnostic issue detected", "No findings were triggered by the current diagnostic rules.")]


def _judge_failure_count_html(run) -> int:
    if "judge_score" not in run.diagnostics_df or "judge_reason" not in run.diagnostics_df:
        return 0
    judge_scores = pd.to_numeric(run.diagnostics_df["judge_score"], errors="coerce")
    reasons = run.diagnostics_df["judge_reason"].fillna("").astype(str)
    return int((judge_scores.isna() & reasons.str.startswith("Judge review failed:")).sum())


def _redact_url(url: str) -> str:
    if not url:
        return "(not set)"
    if len(url) <= 36:
        return url
    return url[:28] + "..." + url[-8:]


def _public_model_name(raw: str) -> str:
    return model_public_label(
        ModelSpec(
            name=raw,
            provider="openai_compatible",
            model=raw,
            api_version=os.environ.get("OPENAI_API_VERSION"),
        )
    )


def _html_risk_rating(run, pass_rate: float, avg_score: float, flag_rate: float) -> tuple[str, str, str]:
    manifest = run.dataset_manifest_df
    has_reference_warning = (
        "reference_shape" in manifest
        and (manifest["reference_shape"] == "passage_or_long_answer").any()
    )
    judge_rescues = 0
    judge_failures = 0
    if "judge_score" in run.diagnostics_df and "passed" in run.diagnostics_df:
        judge_scores = pd.to_numeric(run.diagnostics_df["judge_score"], errors="coerce")
        judge_rescues = int(((judge_scores >= 0.7) & (~run.diagnostics_df["passed"].astype(bool))).sum())
        if "judge_reason" in run.diagnostics_df:
            reasons = run.diagnostics_df["judge_reason"].fillna("").astype(str)
            judge_failures = int((judge_scores.isna() & reasons.str.startswith("Judge review failed:")).sum())
    if pass_rate < 0.60 or flag_rate > 0.40 or has_reference_warning:
        rating, color = "High", "#fff1f2"
    elif pass_rate < 0.80 or avg_score < 0.75 or flag_rate > 0.20 or judge_rescues or judge_failures:
        rating, color = "Medium", "#fffbeb"
    else:
        rating, color = "Low", "#ecfdf5"
    reasons = [f"pass rate {pass_rate:.0%}", f"average score {avg_score:.2f}", f"review flag rate {flag_rate:.0%}"]
    if has_reference_warning:
        reasons.append("long-reference dataset calibration warning")
    if judge_rescues:
        reasons.append(f"{judge_rescues} judge-identified possible metric false negative(s)")
    if judge_failures:
        reasons.append(f"judge review failed for {judge_failures} attempted case(s)")
    return rating, color, "; ".join(reasons) + "."


def _html_provenance(run) -> str:
    source = getattr(run, "resumed_from_checkpoint", None)
    if source:
        return f"Checkpoint replay from previous compatible run <code>{_checkpoint_run_label(source)}</code>"
    return "Fresh target-model evaluation"


def _checkpoint_run_label(path) -> str:
    parts = getattr(path, "parts", ())
    if parts:
        return parts[-2] if len(parts) >= 2 else str(path)
    text = str(path)
    return text.rstrip("/").split("/")[-2] if "/" in text.rstrip("/") else text


def _html_findings(run) -> list[str]:
    findings = []
    if not run.dataset_summary_df.empty:
        strongest = run.dataset_summary_df.sort_values("avg_score", ascending=False).iloc[0]
        weakest = run.dataset_summary_df.sort_values("avg_score", ascending=True).iloc[0]
        findings.append(
            f"Strongest slice: {strongest.dataset_key} / {strongest.category} "
            f"with average score {strongest.avg_score:.2f} and pass rate {strongest.pass_rate:.0%}."
        )
        findings.append(
            f"Weakest slice: {weakest.dataset_key} / {weakest.category} "
            f"with average score {weakest.avg_score:.2f} and pass rate {weakest.pass_rate:.0%}."
        )
    if "review_priority" in run.diagnostics_df:
        counts = run.diagnostics_df["review_priority"].value_counts()
        findings.append("Diagnostic review distribution: " + ", ".join(f"{k}: {v}" for k, v in counts.items()) + ".")
    if "judge_score" in run.diagnostics_df:
        judge_scores = pd.to_numeric(run.diagnostics_df["judge_score"], errors="coerce")
        reviewed = int(judge_scores.notna().sum())
        rescues = int(((judge_scores >= 0.7) & (~run.diagnostics_df["passed"].astype(bool))).sum())
        if reviewed:
            findings.append(
                f"LLM judge reviewed {reviewed} flagged case(s); {rescues} deterministic failure(s) were judged likely correct."
            )
        if "judge_reason" in run.diagnostics_df:
            reasons = run.diagnostics_df["judge_reason"].fillna("").astype(str)
            failures = int((judge_scores.isna() & reasons.str.startswith("Judge review failed:")).sum())
            if failures:
                findings.append(
                    f"LLM judge was attempted for {failures} flagged case(s), but no valid score was returned; check provider parameter compatibility."
                )
    return findings or ["No findings available."]


def _html_conclusion(risk_rating: str, pass_rate: float, flagged: int, total: int, reliability_notes: list[str]) -> str:
    if risk_rating == "Low":
        return "The benchmark evidence is directionally strong for this selected scope. Continue with broader samples and regression tracking before production use."
    if risk_rating == "Medium":
        return f"The run shows mixed evidence: pass rate is {pass_rate:.0%}, with {flagged} of {total} cases requiring review. Inspect diagnostics before model selection."
    notes = " ".join(reliability_notes)
    return (
        f"Interpret cautiously: pass rate is {pass_rate:.0%}, with {flagged} of {total} cases flagged for review. "
        f"{notes} Resolve dataset/reference calibration issues before drawing final capability conclusions."
    )
