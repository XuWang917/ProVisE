import base64
import io

from PIL import Image

from provise.models.generative import (
    OpenAICompatibleImageGenerationModel,
    create_model,
)
from provise.models.openai_compatible import DEFAULT_OPENROUTER_API_BASE


def test_openai_compatible_model_defaults_to_openrouter_base(monkeypatch):
    monkeypatch.delenv("PROVISE_API_KEY", raising=False)
    monkeypatch.delenv("PROVISE_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_BASE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    model = OpenAICompatibleImageGenerationModel(
        "bytedance-seed/seedream-4.5", modalities=["image"]
    )
    model.load_model()

    assert model.api_key == "test-key"
    assert model.api_base == DEFAULT_OPENROUTER_API_BASE


def test_create_model_builds_seedream_openai_compatible_backend():
    model = create_model("seedream")

    assert isinstance(model, OpenAICompatibleImageGenerationModel)
    assert model.model_name == "bytedance-seed/seedream-4.5"
    assert model.modalities == ["image"]
    assert model.provider == "openrouter"


def test_create_model_builds_nanobanana2_openai_compatible_alias():
    model = create_model("nanobanana2")

    assert isinstance(model, OpenAICompatibleImageGenerationModel)
    assert model.model_name == "google/gemini-3.1-flash-image-preview"
    assert model.modalities == ["image", "text"]
    assert model.provider == "openrouter"


def test_create_model_builds_gpt_image_2_openai_compatible_backend():
    model = create_model("gpt-image-2")

    assert isinstance(model, OpenAICompatibleImageGenerationModel)
    assert model.model_name == "gpt-image-2"


def test_openai_compatible_generate_saves_image_and_builds_chat_payload(tmp_path):
    source_path = tmp_path / "source.png"
    output_path = tmp_path / "generated.png"
    Image.new("RGB", (80, 60), (240, 240, 240)).save(source_path)

    returned = Image.new("RGB", (32, 32), (255, 0, 0))
    buffer = io.BytesIO()
    returned.save(buffer, format="PNG")
    returned_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    captured = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "images": [
                                {"image_url": {"url": f"data:image/png;base64,{returned_b64}"}}
                            ]
                        }
                    }
                ]
            }

    model = OpenAICompatibleImageGenerationModel(
        "bytedance-seed/seedream-4.5", modalities=["image"]
    )
    model.api_key = "test-key"
    model.api_base = DEFAULT_OPENROUTER_API_BASE

    def _fake_post(payload):
        captured["payload"] = payload
        return _FakeResponse()

    model._post_with_proxy_fallback = _fake_post

    ok = model.generate(str(source_path), "draw one red square", str(output_path))

    assert ok is True
    assert output_path.exists()
    assert Image.open(output_path).size == (80, 60)
    assert captured["payload"]["model"] == "bytedance-seed/seedream-4.5"
    assert captured["payload"]["modalities"] == ["image"]
    assert captured["payload"]["max_tokens"] == 1024
    assert captured["payload"]["messages"][1]["content"][0]["type"] == "text"
    assert captured["payload"]["messages"][1]["content"][1]["type"] == "image_url"
