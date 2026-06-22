"""Metric standards and scoring profiles shared across notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from genai_capability_bench.metrics.generation import best_reference_generation_scores
from genai_capability_bench.metrics.lexical import best_reference_score, contains_match, exact_match, token_f1
from genai_capability_bench.metrics.semantic import best_tfidf_similarity

MetricRole = Literal["primary", "secondary", "diagnostic", "judge"]


@dataclass(frozen=True)
class MetricSpec:
    """Human-readable metric standard."""

    key: str
    display_name: str
    role: MetricRole
    definition: str
    best_for: str
    limitations: str
    source: str


@dataclass(frozen=True)
class ScoringProfile:
    """Dataset/task scoring profile."""

    key: str
    display_name: str
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...]
    diagnostic_metrics: tuple[str, ...]
    scoring_formula: str
    recommended_for: str
    caveat: str


METRIC_SPECS: dict[str, MetricSpec] = {
    "exact_match": MetricSpec(
        key="exact_match",
        display_name="Exact Match",
        role="primary",
        definition="Normalized prediction exactly matches at least one reference answer.",
        best_for="Short factual answers, dates, IDs, option labels, and names.",
        limitations="Too strict for paraphrases and concise answers when references are long passages.",
        source="SQuAD-style QA evaluation.",
    ),
    "token_f1": MetricSpec(
        key="token_f1",
        display_name="Token F1",
        role="primary",
        definition="Harmonic mean of token precision and recall against the best reference.",
        best_for="Short answer QA with minor wording variation.",
        limitations="Lexical overlap only; misses semantic equivalence.",
        source="SQuAD-style QA evaluation.",
    ),
    "contains_match": MetricSpec(
        key="contains_match",
        display_name="Contains Match",
        role="diagnostic",
        definition="Reference answer appears inside the normalized model answer.",
        best_for="Finding answers embedded in longer responses.",
        limitations="Can over-credit answers that contain the right phrase while adding wrong content.",
        source="Practical QA diagnostic.",
    ),
    "semantic_similarity": MetricSpec(
        key="semantic_similarity",
        display_name="Semantic Similarity",
        role="secondary",
        definition="Cosine similarity over local TF-IDF vectors in the current lightweight implementation.",
        best_for="Offline approximation of semantic overlap when wording differs.",
        limitations="TF-IDF is not contextual; future provider embeddings/BERTScore should replace it for production.",
        source="Local deterministic semantic proxy.",
    ),
    "bleu": MetricSpec(
        key="bleu",
        display_name="BLEU",
        role="secondary",
        definition="N-gram precision with brevity penalty against reference text.",
        best_for="Machine translation-like generation where reference wording is expected.",
        limitations="Poor primary metric for factual QA because correct paraphrases can score low.",
        source="Papineni et al. 2002.",
    ),
    "rouge_1": MetricSpec(
        key="rouge_1",
        display_name="ROUGE-1",
        role="secondary",
        definition="Unigram recall overlap with reference text.",
        best_for="Longer answers or summaries where recall against reference content matters.",
        limitations="Lexical and recall-heavy; can reward verbose overlap.",
        source="Lin 2004.",
    ),
    "rouge_l": MetricSpec(
        key="rouge_l",
        display_name="ROUGE-L",
        role="secondary",
        definition="Longest-common-subsequence F-measure between prediction and reference.",
        best_for="Longer free-form answers and reference-passage comparisons.",
        limitations="Still lexical; not reliable alone for factual correctness.",
        source="Lin 2004.",
    ),
    "llm_judge_correctness": MetricSpec(
        key="llm_judge_correctness",
        display_name="LLM Judge Correctness",
        role="judge",
        definition="Rubric-based model review of answer correctness, used for ambiguous or low-confidence cases.",
        best_for="Open-ended answers where deterministic metrics disagree or references are incomplete.",
        limitations="Adds cost and can be biased; should be calibrated and sampled.",
        source="G-Eval / model-graded evaluation practice.",
    ),
}


SCORING_PROFILES: dict[str, ScoringProfile] = {
    "short_answer_qa": ScoringProfile(
        key="short_answer_qa",
        display_name="Short-Answer QA",
        primary_metrics=("exact_match", "token_f1"),
        secondary_metrics=("semantic_similarity", "bleu", "rouge_l"),
        diagnostic_metrics=("contains_match",),
        scoring_formula="max(exact_match, 0.65 * token_f1 + 0.35 * semantic_similarity)",
        recommended_for="Closed-book and open-domain QA with concise aliases.",
        caveat="Contains match is diagnostic only to avoid over-crediting long wrong answers.",
    ),
    "multiple_choice": ScoringProfile(
        key="multiple_choice",
        display_name="Multiple Choice",
        primary_metrics=("exact_match",),
        secondary_metrics=("token_f1",),
        diagnostic_metrics=("contains_match",),
        scoring_formula="exact_match against answer text or displayed option label",
        recommended_for="MMLU, ARC, and option-label tasks.",
        caveat="A dedicated option parser should be preferred when models return explanations.",
    ),
    "long_reference_qa": ScoringProfile(
        key="long_reference_qa",
        display_name="Long-Reference QA",
        primary_metrics=("rouge_l", "semantic_similarity"),
        secondary_metrics=("token_f1", "rouge_1", "bleu"),
        diagnostic_metrics=("contains_match",),
        scoring_formula=(
            "max(0.55 * rouge_l + 0.45 * semantic_similarity, "
            "0.50 * token_f1 + 0.50 * semantic_similarity)"
        ),
        recommended_for="Questions where references are long passages rather than concise answers.",
        caveat="Low scores may indicate reference-shape mismatch, not necessarily wrong concise answers.",
    ),
    "truthfulness_generation": ScoringProfile(
        key="truthfulness_generation",
        display_name="Truthfulness Generation",
        primary_metrics=("llm_judge_correctness",),
        secondary_metrics=("semantic_similarity", "token_f1"),
        diagnostic_metrics=("contains_match",),
        scoring_formula="judge rubric score, with deterministic metrics as supporting evidence",
        recommended_for="TruthfulQA-style misconception-resistant generation.",
        caveat="Correct-vs-incorrect reference comparison and calibrated judge rubrics are needed.",
    ),
}


def metric_standards_table() -> pd.DataFrame:
    """Return the repo-wide metric standard table."""

    return pd.DataFrame([spec.__dict__ for spec in METRIC_SPECS.values()])


def scoring_profiles_table() -> pd.DataFrame:
    """Return repo-wide scoring profile definitions."""

    return pd.DataFrame([profile.__dict__ for profile in SCORING_PROFILES.values()])


def evaluate_reference_metrics(prediction: str, references: list[str], scoring_profile: str) -> dict[str, float | str]:
    """Compute common reference metrics and profile-specific primary score."""

    references = references or []
    exact = best_reference_score(prediction, references, exact_match)
    contains = best_reference_score(prediction, references, contains_match)
    f1 = best_reference_score(prediction, references, token_f1)
    semantic = best_tfidf_similarity(prediction, references)
    generation = best_reference_generation_scores(prediction, references)

    if scoring_profile == "multiple_choice":
        primary = exact
    elif scoring_profile == "long_reference_qa":
        primary = max(0.55 * generation["rouge_l"] + 0.45 * semantic, 0.50 * f1 + 0.50 * semantic)
    else:
        primary = max(exact, 0.65 * f1 + 0.35 * semantic)

    return {
        "scoring_profile": scoring_profile,
        "primary_score": float(primary),
        "exact_match": float(exact),
        "contains_match": float(contains),
        "token_f1": float(f1),
        "semantic_similarity": float(semantic),
        "tfidf_similarity": float(semantic),
        **{k: float(v) for k, v in generation.items()},
    }
