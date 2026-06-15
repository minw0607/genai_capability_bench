from genai_capability_bench.core.runner import config_float


def test_config_float_uses_default_for_unresolved_env_placeholder():
    assert config_float("${EVAL_PASS_THRESHOLD}", 0.7) == 0.7


def test_config_float_parses_numeric_strings():
    assert config_float("0.85", 0.7) == 0.85

