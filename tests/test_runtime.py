from src.reflexion_lab.mock_runtime import (
    OpenRouterConfig,
    parse_json_object,
    parse_openrouter_message,
    runtime_config_from_env,
)


def test_build_runtime_from_env_reads_openrouter_settings(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    config = runtime_config_from_env()

    assert config == OpenRouterConfig(
        api_key="test-key",
        model="openai/gpt-oss-20b:free",
        base_url="https://openrouter.ai/api/v1",
    )


def test_parse_openrouter_usage_counts_tokens():
    payload = {
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }

    result = parse_openrouter_message(payload, latency_ms=42)

    assert result.content == "hello"
    assert result.token_count == 18
    assert result.latency_ms == 42


def test_parse_json_object_extracts_embedded_json():
    text = "Here is the result:\n```json\n{\"score\": 1, \"reason\": \"ok\"}\n```"

    payload = parse_json_object(text)

    assert payload == {"score": 1, "reason": "ok"}
