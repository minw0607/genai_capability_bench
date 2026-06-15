from genai_capability_bench.core.runner import config_float, config_optional_int, config_optional_str, load_config


def test_config_float_uses_default_for_unresolved_env_placeholder():
    assert config_float("${EVAL_PASS_THRESHOLD}", 0.7) == 0.7


def test_config_float_parses_numeric_strings():
    assert config_float("0.85", 0.7) == 0.85


def test_optional_int_can_omit_parameter():
    assert config_optional_int("omit", 1000) is None
    assert config_optional_int("${OPENAI_MAX_TOKENS}", 1000) == 1000
    assert config_optional_int("${OPENAI_MAX_TOKENS}", None) is None


def test_optional_str_can_omit_parameter_name():
    assert config_optional_str("omit", "max_tokens") is None
    assert config_optional_str("${OPENAI_TOKEN_PARAMETER}", "max_tokens") == "max_tokens"
    assert config_optional_str("${OPENAI_TOKEN_PARAMETER}", None) is None


def test_load_config_supports_legacy_model_env_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_GENERATION_MODEL", "")
    monkeypatch.setenv("OPENAI_JUDGE_MODEL", "")
    monkeypatch.setenv("AGENT_MODEL", "legacy-generation-model")
    monkeypatch.setenv("JUDGE_MODEL", "legacy-judge-model")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model: ${OPENAI_GENERATION_MODEL}\njudge: ${OPENAI_JUDGE_MODEL}\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["model"] == "legacy-generation-model"
    assert config["judge"] == "legacy-judge-model"
