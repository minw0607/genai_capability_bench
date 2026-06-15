from genai_capability_bench.metrics.lexical import contains_match, exact_match, token_f1
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
