from genai_capability_bench.metrics.generation import bleu_score, rouge_l
from genai_capability_bench.metrics.lexical import contains_match, exact_match, token_f1
from genai_capability_bench.metrics.registry import evaluate_reference_metrics, metric_standards_table
from genai_capability_bench.metrics.semantic import tfidf_similarity


def test_exact_match_normalizes_articles_and_case():
    assert exact_match("The George Washington!", "george washington") == 1.0


def test_contains_match():
    assert contains_match("The answer is carbon dioxide.", "carbon dioxide") == 1.0


def test_token_f1_partial_overlap():
    assert 0.0 < token_f1("George Washington", "Washington") <= 1.0


def test_tfidf_similarity_handles_single_character_answers():
    assert tfidf_similarity("C", "C") == 1.0


def test_tfidf_similarity_handles_punctuation_only_text():
    assert tfidf_similarity("...", "...") == 0.0


def test_rouge_l_rewards_sequence_overlap():
    assert rouge_l("George Washington", "George Washington") == 1.0


def test_bleu_is_available_as_secondary_metric():
    assert bleu_score("the cat sat", "the cat sat") > 0.0


def test_metric_registry_computes_profile_primary_score_without_contains_credit():
    metrics = evaluate_reference_metrics(
        "APR stands for Annual Percentage Rate",
        ["Annual Percentage Rate"],
        "short_answer_qa",
    )

    assert metrics["contains_match"] == 1.0
    assert metrics["exact_match"] == 0.0
    assert metrics["primary_score"] < 1.0


def test_metric_registry_defaults_semantic_similarity_to_tfidf():
    metrics = evaluate_reference_metrics("George Washington", ["Washington"], "short_answer_qa")

    assert metrics["semantic_similarity_method"] == "tfidf"
    assert metrics["tfidf_similarity"] is not None


def test_metric_registry_can_fallback_from_optional_api_embeddings():
    metrics = evaluate_reference_metrics(
        "George Washington",
        ["Washington"],
        "short_answer_qa",
        semantic_similarity_mode="api_embeddings_if_configured",
    )

    assert metrics["semantic_similarity_method"] in {"api_embeddings", "tfidf", "tfidf_fallback"}


def test_multiple_choice_accepts_label_plus_answer_text():
    metrics = evaluate_reference_metrics(
        "D. stamina.",
        ["stamina.", "D", "D. stamina.", "D stamina."],
        "multiple_choice",
    )

    assert metrics["primary_score"] == 1.0
    assert metrics["exact_match"] == 1.0


def test_metric_standards_table_includes_bleu_and_rouge():
    keys = set(metric_standards_table()["key"])
    assert {"bleu", "rouge_l", "semantic_similarity"} <= keys
