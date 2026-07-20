import numpy as np

from provise.models import vlm as vlm_module
from provise.protocols.depth import (
    DenseDepthABProtocol,
    coordinates_from_item,
    normalize_depth_label,
    parse_depth_coordinates,
)


def test_normalize_depth_label_variants():
    assert normalize_depth_label("A") == "(A)"
    assert normalize_depth_label("(A)") == "(A)"
    assert normalize_depth_label("B") == "(B)"
    assert normalize_depth_label("(B)") == "(B)"


def test_parse_depth_coordinates_json_like_response():
    assert parse_depth_coordinates("[[0.1, 0.2], [0.3, 0.4]]") == [[0.1, 0.2], [0.3, 0.4]]
    assert parse_depth_coordinates("```json\n[[.10, .20], [3e-1, 4e-1]]\n```") == [
        [0.1, 0.2],
        [0.3, 0.4],
    ]


def test_parse_depth_coordinates_rejects_invalid_response():
    assert parse_depth_coordinates("[[1.2, 0.2], [0.3, 0.4]]") == []
    assert parse_depth_coordinates("A is closer") == []


def test_coordinates_from_item_prefers_evaluation_coordinates():
    item = {
        "evaluation": {"coordinates": [[0.2, 0.3], [0.7, 0.8]]},
        "metadata": {"coordinates": [[0.1, 0.1], [0.9, 0.9]]},
    }

    assert coordinates_from_item(item) == [[0.2, 0.3], [0.7, 0.8]]


def test_coordinates_from_item_reads_huggingface_metadata_json():
    item = {
        "metadata_json": '{"coordinates":[[0.2,0.3],[0.7,0.8]],"source":"BLINK"}'
    }

    assert coordinates_from_item(item) == [[0.2, 0.3], [0.7, 0.8]]


def test_coordinates_from_item_reads_nested_metadata_json():
    item = {
        "metadata": {
            "metadata_json": '{"coordinates":[[0.2,0.3],[0.7,0.8]]}'
        }
    }

    assert coordinates_from_item(item) == [[0.2, 0.3], [0.7, 0.8]]


def test_parse_with_coords_samples_generated_depth_map():
    image = np.zeros((10, 10), dtype=np.uint8)
    image[:, 2] = 200
    image[:, 8] = 50
    item = {"evaluation": {"coordinates": [[0.2, 0.5], [0.8, 0.5]]}}
    protocol = DenseDepthABProtocol({"kernel_size": 1})

    parsed = protocol._parse_with_coords(image, item)

    assert parsed.parse_success is True
    assert parsed.prediction == "(A)"
    assert parsed.extra["coordinate_source"] == "benchmark"
    assert parsed.extra["depth_a"] == 200.0
    assert parsed.extra["depth_b"] == 50.0
    assert parsed.extra["depth_delta"] == 150.0


def test_parse_with_coords_rejects_indistinguishable_depth_values():
    image = np.full((10, 10), 80, dtype=np.uint8)
    item = {"evaluation": {"coordinates": [[0.2, 0.5], [0.8, 0.5]]}}
    protocol = DenseDepthABProtocol({"kernel_size": 1})

    parsed = protocol._parse_with_coords(image, item)

    assert parsed.parse_success is False
    assert parsed.prediction == ""
    assert parsed.extra["error"] == "indistinguishable depth values"
    assert parsed.extra["depth_delta"] == 0.0


def test_parse_with_coords_infers_missing_coordinates(tmp_path):
    source_path = tmp_path / "source.png"
    source_path.write_bytes(b"not used by fake vlm")
    image = np.zeros((10, 10), dtype=np.uint8)
    image[:, 2] = 50
    image[:, 8] = 200
    item = {
        "image_path": source_path.name,
        "evaluation": {"coordinates": []},
        "metadata": {"coordinates": []},
    }
    protocol = DenseDepthABProtocol({"kernel_size": 1})

    class FakeVLM:
        def predict(self, image_path, prompt):
            assert image_path == str(source_path.resolve())
            assert "[[x_A, y_A], [x_B, y_B]]" in prompt
            return "[[0.2, 0.5], [0.8, 0.5]]"

    protocol._get_eval_vlm = lambda: FakeVLM()

    parsed = protocol._parse_with_coords(image, item, str(tmp_path))

    assert parsed.parse_success is True
    assert parsed.prediction == "(B)"
    assert parsed.extra["coordinate_source"] == "vlm"
    assert parsed.extra["coordinates"] == [[0.2, 0.5], [0.8, 0.5]]


def test_depth_parser_uses_configured_eval_vlm_from_environment(monkeypatch):
    captured = {}

    class FakeVLM:
        model_name = "gpt-5.4"

        def load_model(self):
            return None

    def fake_create_eval_vlm(*, timeout, max_tokens, model_name):
        captured.update(timeout=timeout, max_tokens=max_tokens, model_name=model_name)
        return FakeVLM()

    monkeypatch.setenv("PROVISE_PARSER_MODEL", "gpt-5.4")
    monkeypatch.setattr(vlm_module, "create_eval_vlm", fake_create_eval_vlm)
    DenseDepthABProtocol._eval_vlm = None
    DenseDepthABProtocol._eval_vlm_key = None
    try:
        protocol = DenseDepthABProtocol({"vlm_model": "", "vlm_timeout": 30, "vlm_max_tokens": 64})
        vlm = protocol._get_eval_vlm()
    finally:
        DenseDepthABProtocol._eval_vlm = None
        DenseDepthABProtocol._eval_vlm_key = None

    assert vlm.model_name == "gpt-5.4"
    assert captured == {"timeout": 30, "max_tokens": 64, "model_name": "gpt-5.4"}
