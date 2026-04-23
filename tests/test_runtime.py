from src.reflexion_lab.mock_runtime import (
    OpenAIConfig,
    parse_json_object,
    parse_openai_response,
    runtime_config_from_env,
)


def test_build_runtime_from_env_reads_openai_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-5-nano")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "240")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "4")

    config = runtime_config_from_env()

    assert config == OpenAIConfig(
        api_key="test-key",
        model_name="gpt-5-nano",
        timeout_seconds=240,
        max_retries=4,
    )


def test_parse_openai_usage_counts_tokens():
    payload = {
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }

    result = parse_openai_response(payload, latency_ms=42)

    assert result.content == "hello"
    assert result.token_count == 18
    assert result.latency_ms == 42


def test_parse_json_object_extracts_embedded_json():
    text = "Here is the result:\n```json\n{\"score\": 1, \"reason\": \"ok\"}\n```"

    payload = parse_json_object(text)

    assert payload == {"score": 1, "reason": "ok"}
