from pathlib import Path

import pytest
import requests

from provise.models.vlm import (
    DEFAULT_EVAL_VLM_MODEL,
    OpenAICompatibleVisionLanguageModel,
    create_eval_vlm,
)
from provise.models.openai_compatible import DEFAULT_OPENROUTER_API_BASE


class _Response:
    def __init__(self, status_code, content="ok", text=""):
        self.status_code = status_code
        self.text = text
        self._content = content

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_eval_vlm_defaults_to_openrouter_base(monkeypatch):
    monkeypatch.delenv("PROVISE_API_KEY", raising=False)
    monkeypatch.delenv("PROVISE_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_BASE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    model = OpenAICompatibleVisionLanguageModel(DEFAULT_EVAL_VLM_MODEL)
    model.load_model()

    assert model.api_key == "test-key"
    assert model.api_base == DEFAULT_OPENROUTER_API_BASE


def test_eval_vlm_explicit_openrouter_provider_overrides_proxy_config(monkeypatch):
    monkeypatch.setenv("PROVISE_API_KEY", "proxy-key")
    monkeypatch.setenv("PROVISE_API_BASE", "https://proxy.example/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.delenv("OPENROUTER_API_BASE", raising=False)

    model = OpenAICompatibleVisionLanguageModel(
        "moonshotai/kimi-k3", provider="openrouter"
    )
    model.load_model()

    assert model.api_key == "router-key"
    assert model.api_base == DEFAULT_OPENROUTER_API_BASE


def test_create_eval_vlm_uses_default_model(monkeypatch):
    monkeypatch.delenv("PROVISE_PARSER_MODEL", raising=False)
    monkeypatch.delenv("PROVISE_PARSER_MAX_TOKENS", raising=False)

    model = create_eval_vlm()

    assert isinstance(model, OpenAICompatibleVisionLanguageModel)
    assert model.model_name == DEFAULT_EVAL_VLM_MODEL
    assert model.max_tokens == 512


def test_eval_vlm_retries_transient_http_error(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    model = OpenAICompatibleVisionLanguageModel("test-model", max_retries=2, retry_backoff=0)
    model.api_key = "test-key"
    model.api_base = "https://example.test/v1"
    responses = iter([_Response(502, text="temporary"), _Response(200, content="parsed")])
    monkeypatch.setattr(model, "_post_with_proxy_fallback", lambda payload: next(responses))

    assert model.predict(str(image_path), "question") == "parsed"


def test_eval_vlm_supports_request_controls(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    model = OpenAICompatibleVisionLanguageModel(
        "test-model",
        system_prompt="Return structured output.",
        response_format={"type": "json_object"},
        temperature=None,
        reasoning={"enabled": False},
    )
    model.api_key = "test-key"
    model.api_base = "https://example.test/v1"
    payloads = []

    def request(payload):
        payloads.append(payload)
        return _Response(200, content="parsed")

    monkeypatch.setattr(model, "_post_with_proxy_fallback", request)

    assert model.predict(str(image_path), "question") == "parsed"
    assert payloads[0]["messages"][0] == {
        "role": "system",
        "content": "Return structured output.",
    }
    assert payloads[0]["response_format"] == {"type": "json_object"}
    assert payloads[0]["reasoning"] == {"enabled": False}
    assert "temperature" not in payloads[0]


def test_eval_vlm_retries_empty_success_response(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    model = OpenAICompatibleVisionLanguageModel(
        "test-model", max_retries=1, retry_backoff=0
    )
    model.api_key = "test-key"
    model.api_base = "https://example.test/v1"
    responses = iter([_Response(200, content=None), _Response(200, content="parsed")])
    monkeypatch.setattr(model, "_post_with_proxy_fallback", lambda payload: next(responses))

    assert model.predict(str(image_path), "question") == "parsed"


def test_eval_vlm_raises_after_transient_retries_are_exhausted(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    model = OpenAICompatibleVisionLanguageModel("test-model", max_retries=2, retry_backoff=0)
    model.api_key = "test-key"
    model.api_base = "https://example.test/v1"
    calls = []

    def fail(payload):
        calls.append(payload)
        return _Response(503, text="still unavailable")

    monkeypatch.setattr(model, "_post_with_proxy_fallback", fail)

    with pytest.raises(RuntimeError, match="API error 503"):
        model.predict(str(image_path), "question")
    assert len(calls) == 3


def test_eval_vlm_retries_transient_request_exception(monkeypatch, tmp_path: Path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    model = OpenAICompatibleVisionLanguageModel("test-model", max_retries=2, retry_backoff=0)
    model.api_key = "test-key"
    model.api_base = "https://example.test/v1"
    outcomes = iter(
        [
            requests.exceptions.ReadTimeout("temporary timeout"),
            _Response(200, content="parsed"),
        ]
    )

    def request(_payload):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(model, "_post_with_proxy_fallback", request)

    assert model.predict(str(image_path), "question") == "parsed"


def test_eval_vlm_raises_after_request_exceptions_are_exhausted(
    monkeypatch, tmp_path: Path
):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    model = OpenAICompatibleVisionLanguageModel("test-model", max_retries=2, retry_backoff=0)
    model.api_key = "test-key"
    model.api_base = "https://example.test/v1"
    calls = []

    def fail(payload):
        calls.append(payload)
        raise requests.exceptions.ChunkedEncodingError("response stream ended")

    monkeypatch.setattr(model, "_post_with_proxy_fallback", fail)

    with pytest.raises(RuntimeError, match="failed after 3 attempt"):
        model.predict(str(image_path), "question")
    assert len(calls) == 3
