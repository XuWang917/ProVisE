import json

import cv2
import numpy as np
import pytest

from provise.parser_ops import (
    DEFAULT_REGISTRY,
    ParserContext,
    ParserOpSpec,
    ParserPlanError,
    ParserRegistry,
    ParserValue,
    cyan_point_marker_edit_pipeline,
    cyan_point_marker_pipeline,
    green_choice_count_board_pipeline,
    green_instance_marker_count_pipeline,
    marked_object_choice_pipeline,
    relation_zone_boolean_pipeline,
)
from provise.protocol_agent.builder import load_protocol_catalog, protocol_inventory
from provise.protocols import create_protocol
from provise.parser_ops.operators import _resolve_clip_model_name
from provise.evaluation.runner import load_protocol_pool, resolve_task_config


def _write_image(path, circles=()):
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    for x, y, radius in circles:
        cv2.circle(image, (x, y), radius, (255, 255, 0), -1)
    cv2.imwrite(str(path), image)


def _execute_point_pipeline(path, **config):
    return DEFAULT_REGISTRY.execute(
        cyan_point_marker_pipeline(),
        ParserContext(
            generated_path=str(path),
            item={},
            benchmark_root="",
            protocol_config={"min_pixels": 5, **config},
        ),
    )


def test_parser_ops_reject_unknown_operator():
    pipeline = {
        "steps": [{"id": "answer", "op": "run_arbitrary_python"}],
        "output": "answer",
    }

    with pytest.raises(ParserPlanError, match="Unknown parser operator"):
        DEFAULT_REGISTRY.compile(pipeline)


def test_parser_ops_reject_missing_input_reference():
    pipeline = {
        "steps": [
            {
                "id": "mask",
                "op": "hsv_color_mask",
                "inputs": ["missing_image"],
                "params": {"lower": [80, 70, 70], "upper": [100, 255, 255]},
            }
        ],
        "output": "mask",
    }

    with pytest.raises(ParserPlanError, match="missing or forward input"):
        DEFAULT_REGISTRY.compile(pipeline)


def test_parser_ops_reject_static_input_kind_mismatch():
    pipeline = {
        "steps": [
            {"id": "image", "op": "load_generated"},
            {"id": "centroid", "op": "component_centroid", "inputs": ["image"]},
        ],
        "output": "centroid",
    }

    with pytest.raises(ParserPlanError, match="expected_kind|expects.*kind|received 'image_bgr'"):
        DEFAULT_REGISTRY.compile(pipeline)


def test_parser_ops_reject_unlisted_parameter():
    pipeline = {
        "steps": [{"id": "image", "op": "load_generated", "params": {"shell": "rm -rf /"}}],
        "output": "image",
    }

    with pytest.raises(ParserPlanError, match="unsupported params"):
        DEFAULT_REGISTRY.compile(pipeline)


def test_parser_ops_reject_invalid_literal_parameters_at_compile_time():
    pipeline = {
        "steps": [
            {"id": "image", "op": "load_generated"},
            {
                "id": "mask",
                "op": "hsv_color_mask",
                "inputs": ["image"],
                "params": {"lower": [-1, 0, 0], "upper": [10, 255, 255]},
            },
        ],
        "output": "mask",
    }

    with pytest.raises(ParserPlanError, match="invalid params"):
        DEFAULT_REGISTRY.compile(pipeline)


def test_parser_ops_detect_runtime_output_kind_mismatch():
    registry = ParserRegistry()
    registry.register(
        ParserOpSpec(
            name="bad_contract",
            input_kinds=(),
            output_kind="expected_kind",
            function=lambda context, inputs, params: ParserValue("wrong_kind", 1),
        )
    )

    result = registry.execute(
        {"steps": [{"id": "value", "op": "bad_contract"}], "output": "value"},
        ParserContext("unused.png", {}, ""),
    )

    assert not result.success
    assert result.error_type == "operator_kind_mismatch"


def test_point_marker_pipeline_recovers_normalized_point(tmp_path):
    path = tmp_path / "marker.png"
    _write_image(path, [(100, 50, 6)])

    result = _execute_point_pipeline(path)

    assert result.success
    assert result.output is not None
    assert result.output.kind == "normalized_point"
    assert abs(result.prediction[0] - 0.5) < 0.02
    assert abs(result.prediction[1] - 0.5) < 0.02
    json.dumps(result.diagnostics)


def test_point_marker_pipeline_reports_missing_marker(tmp_path):
    path = tmp_path / "empty.png"
    _write_image(path)

    result = _execute_point_pipeline(path)

    assert not result.success
    assert result.error_type == "no_component"
    assert result.error == "cyan point marker not found"


def test_point_marker_pipeline_rejects_ambiguous_markers(tmp_path):
    path = tmp_path / "ambiguous.png"
    _write_image(path, [(50, 50, 7), (150, 50, 7)])

    result = _execute_point_pipeline(path)

    assert not result.success
    assert result.error_type == "ambiguous_components"
    assert len(result.diagnostics["steps"]["marker"]["candidate_scores"]) == 2


def test_point_marker_pipeline_ignores_smaller_cyan_distractor(tmp_path):
    path = tmp_path / "distractor.png"
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    cv2.circle(image, (50, 50), 8, (255, 255, 0), -1)
    image[10:16, 170:182] = (255, 255, 0)
    cv2.imwrite(str(path), image)

    result = _execute_point_pipeline(path)

    assert result.success
    assert abs(result.prediction[0] - 0.25) < 0.02
    assert result.diagnostics["steps"]["marker"]["component_count"] == 2


def test_source_aware_pipeline_removes_larger_preexisting_cyan_region(tmp_path):
    source_path = tmp_path / "source.png"
    generated_path = tmp_path / "generated.png"
    source = np.full((100, 200, 3), 255, dtype=np.uint8)
    source[55:95, 130:190] = (255, 255, 0)
    generated = source.copy()
    cv2.circle(generated, (40, 25), 7, (255, 255, 0), -1)
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(generated_path), generated)

    result = DEFAULT_REGISTRY.execute(
        cyan_point_marker_edit_pipeline(),
        ParserContext(
            generated_path=str(generated_path),
            item={},
            benchmark_root=str(tmp_path),
            source_paths=(str(source_path),),
            protocol_config={"min_pixels": 5},
        ),
    )

    assert result.success
    assert abs(result.prediction[0] - 0.2) < 0.02
    assert abs(result.prediction[1] - 0.25) < 0.02
    assert result.diagnostics["steps"]["mask"]["candidate_pixel_count"] > 2000
    assert result.diagnostics["steps"]["mask"]["pixel_count"] < 300


def test_source_aware_instance_count_ignores_natural_green_and_counts_new_markers(tmp_path):
    source_path = tmp_path / "green_source.png"
    generated_path = tmp_path / "green_generated.png"
    source = np.full((120, 200, 3), 255, dtype=np.uint8)
    source[55:115, 5:90] = (0, 180, 0)
    generated = source.copy()
    cv2.circle(generated, (130, 35), 8, (0, 255, 0), -1)
    cv2.circle(generated, (170, 75), 8, (0, 255, 0), -1)
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(generated_path), generated)

    result = DEFAULT_REGISTRY.execute(
        green_instance_marker_count_pipeline(),
        ParserContext(
            generated_path=str(generated_path),
            item={},
            benchmark_root=str(tmp_path),
            source_paths=(str(source_path),),
            protocol_config={"min_area": 30},
        ),
    )

    assert result.success
    assert result.prediction == 2
    assert result.diagnostics["steps"]["count"]["component_count"] == 2


def test_source_aware_instance_count_returns_zero_when_no_marker_was_added(tmp_path):
    source_path = tmp_path / "unchanged_source.png"
    generated_path = tmp_path / "unchanged_generated.png"
    source = np.full((120, 200, 3), 255, dtype=np.uint8)
    source[30:100, 20:180] = (0, 180, 0)
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(generated_path), source)

    result = DEFAULT_REGISTRY.execute(
        green_instance_marker_count_pipeline(),
        ParserContext(
            generated_path=str(generated_path),
            item={},
            benchmark_root=str(tmp_path),
            source_paths=(str(source_path),),
            protocol_config={"min_area": 30},
        ),
    )

    assert result.success
    assert result.prediction == 0


def test_choice_count_board_maps_visible_marker_count_to_numeric_choice(tmp_path):
    generated_path = tmp_path / "count_board.png"
    image = np.full((180, 360, 3), 255, dtype=np.uint8)
    for x in (80, 180, 280):
        cv2.circle(image, (x, 135), 14, (0, 255, 0), -1)
    cv2.imwrite(str(generated_path), image)
    item = {
        "answer": "B",
        "choices": [
            {"label": "A", "text": "1"},
            {"label": "B", "text": "3"},
            {"label": "C", "text": "four"},
        ],
    }

    result = DEFAULT_REGISTRY.execute(
        green_choice_count_board_pipeline(),
        ParserContext(str(generated_path), item, str(tmp_path)),
    )

    assert result.success, result.error
    assert result.prediction == "B"
    assert result.diagnostics["steps"]["count"]["component_count"] == 3
    assert result.diagnostics["steps"]["choice"]["count"] == 3


def test_marked_object_choice_uses_deterministic_argmax_without_abstention():
    pipeline = marked_object_choice_pipeline()
    choice = next(step for step in pipeline["steps"] if step["id"] == "choice")

    assert choice["params"]["min_margin"]["default"] == 0.0


@pytest.mark.parametrize(
    ("subject_center", "expected"),
    [((100, 50), True), ((20, 50), False)],
)
def test_relation_zone_pipeline_recovers_boolean_from_spatial_membership(
    tmp_path, subject_center, expected
):
    source_path = tmp_path / "source.png"
    generated_path = tmp_path / "generated.png"
    source = np.full((100, 200, 3), 255, dtype=np.uint8)
    generated = source.copy()
    cv2.rectangle(generated, (60, 15), (150, 85), (0, 255, 0), -1)
    cv2.circle(generated, subject_center, 7, (255, 0, 255), -1)
    cv2.circle(generated, (175, 50), 7, (255, 255, 0), -1)
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(generated_path), generated)

    result = DEFAULT_REGISTRY.execute(
        relation_zone_boolean_pipeline(),
        ParserContext(
            generated_path=str(generated_path),
            item={},
            benchmark_root=str(tmp_path),
            source_paths=(str(source_path),),
        ),
    )

    assert result.success, result.error
    assert result.output.kind == "boolean"
    assert result.prediction is expected
    assert result.diagnostics["steps"]["reference_component"]["status"] == "ok"


def test_relation_zone_pipeline_rejects_missing_target_region(tmp_path):
    source_path = tmp_path / "source.png"
    generated_path = tmp_path / "generated.png"
    source = np.full((100, 200, 3), 255, dtype=np.uint8)
    generated = source.copy()
    cv2.circle(generated, (100, 50), 7, (255, 0, 255), -1)
    cv2.circle(generated, (175, 50), 7, (255, 255, 0), -1)
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(generated_path), generated)

    result = DEFAULT_REGISTRY.execute(
        relation_zone_boolean_pipeline(),
        ParserContext(
            generated_path=str(generated_path),
            item={},
            benchmark_root=str(tmp_path),
            source_paths=(str(source_path),),
        ),
    )

    assert not result.success
    assert result.error_type == "target_region_missing"


def test_protocol_spec_loads_and_executes_parser_ops_pipeline(tmp_path):
    protocol_pool = load_protocol_pool("configs/protocol_specs")

    protocol_name, config, prompt = resolve_task_config(
        "localization",
        {
            "protocol": "agentic_point_marker",
            "prompt_variant": "cyan_point_marker",
            "protocol_config": {"min_pixels": 5},
        },
        protocol_pool,
    )

    assert protocol_name == "agentic_point_marker"
    assert config["parser_pipeline"]["output"] == "point"
    assert config["min_pixels"] == 5
    assert "solid cyan circular dot" in prompt

    source_path = tmp_path / "source.png"
    generated_path = tmp_path / "yaml_marker.png"
    _write_image(source_path)
    _write_image(generated_path, [(100, 50, 6)])
    item = {"answer": [0.5, 0.5], "image_path": source_path.name}
    parsed = create_protocol(protocol_name, config).parse(
        str(generated_path), item, str(tmp_path)
    )
    assert parsed.parse_success
    assert abs(parsed.prediction[0] - 0.5) < 0.02


def test_dynamic_parser_ops_protocol_accepts_an_inline_generated_prompt():
    protocol_pool = load_protocol_pool("configs/protocol_specs")
    pipeline = cyan_point_marker_pipeline()

    protocol_name, config, prompt = resolve_task_config(
        "localization",
        {
            "protocol": "agentic_parser_ops_protocol",
            "prompt_variant": "generated",
            "prompt": "For {question}, add one cyan point.",
            "protocol_config": {"parser_pipeline": pipeline},
        },
        protocol_pool,
    )

    assert protocol_name == "agentic_parser_ops_protocol"
    assert config["parser_pipeline"] == pipeline
    assert prompt == "For {question}, add one cyan point."


def test_agent_protocol_inventory_exposes_visual_contract_and_parser_ops():
    inventory = protocol_inventory(load_protocol_catalog("configs/protocol_specs"))
    point_marker = next(row for row in inventory if row["protocol"] == "agentic_point_marker")

    assert point_marker["visual_contract"]["primitive"] == "solid_circle"
    assert point_marker["parser_ops"]["output_kind"] == "normalized_point"
    assert all(row["protocol"] != "agentic_parser_ops_protocol" for row in inventory)


def test_parser_ops_recovers_normalized_bbox(tmp_path):
    path = tmp_path / "box.png"
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 20), (159, 79), (0, 0, 255), -1)
    cv2.imwrite(str(path), image)
    pipeline = {
        "steps": [
            {"id": "image", "op": "load_generated"},
            {
                "id": "mask",
                "op": "hsv_color_mask",
                "inputs": ["image"],
                "params": {"lower": [0, 180, 180], "upper": [10, 255, 255]},
            },
            {"id": "box", "op": "mask_bbox", "inputs": ["mask"]},
            {"id": "answer", "op": "normalize_bbox", "inputs": ["box", "image"]},
        ],
        "output": "answer",
    }

    result = DEFAULT_REGISTRY.execute(pipeline, ParserContext(str(path), {}, ""))

    assert result.success, result.error
    assert result.output.kind == "normalized_bbox"
    assert np.allclose(result.prediction, [40 / 199, 20 / 99, 160 / 199, 80 / 99])


def test_parser_ops_recovers_path_endpoints(tmp_path):
    path = tmp_path / "path.png"
    image = np.full((120, 240, 3), 255, dtype=np.uint8)
    cv2.line(image, (24, 60), (215, 60), (0, 0, 255), 5)
    cv2.imwrite(str(path), image)
    pipeline = {
        "steps": [
            {"id": "image", "op": "load_generated"},
            {
                "id": "mask",
                "op": "hsv_color_mask",
                "inputs": ["image"],
                "params": {"lower": [0, 180, 180], "upper": [10, 255, 255]},
            },
            {"id": "skeleton", "op": "skeletonize_mask", "inputs": ["mask"]},
            {"id": "ends", "op": "mask_endpoints", "inputs": ["skeleton"]},
            {
                "id": "answer",
                "op": "normalize_polyline",
                "inputs": ["ends", "image"],
            },
        ],
        "output": "answer",
    }

    result = DEFAULT_REGISTRY.execute(pipeline, ParserContext(str(path), {}, ""))

    assert result.success, result.error
    xs = sorted(point[0] for point in result.prediction)
    assert xs[0] < 0.15
    assert xs[1] > 0.85


def test_parser_ops_recovers_line_angle(tmp_path):
    path = tmp_path / "line.png"
    image = np.full((160, 240, 3), 255, dtype=np.uint8)
    cv2.line(image, (30, 120), (210, 40), (255, 0, 0), 5)
    cv2.imwrite(str(path), image)
    pipeline = {
        "steps": [
            {"id": "image", "op": "load_generated"},
            {
                "id": "mask",
                "op": "hsv_color_mask",
                "inputs": ["image"],
                "params": {"lower": [110, 150, 150], "upper": [130, 255, 255]},
            },
            {
                "id": "lines",
                "op": "hough_lines",
                "inputs": ["mask"],
                "params": {"threshold": 20, "min_line_length": 80, "max_line_gap": 10},
            },
            {"id": "line", "op": "select_longest_line", "inputs": ["lines"]},
            {"id": "answer", "op": "line_angle", "inputs": ["line"]},
        ],
        "output": "answer",
    }

    result = DEFAULT_REGISTRY.execute(pipeline, ParserContext(str(path), {}, ""))

    assert result.success, result.error
    assert -30.0 < result.prediction < -15.0


def test_clip_model_resolution_prefers_explicit_then_environment(monkeypatch):
    monkeypatch.setenv("PROVISE_CLIP_MODEL", "/models/provise-clip")
    assert _resolve_clip_model_name({"model": "/models/explicit"}) == "/models/explicit"
    assert _resolve_clip_model_name({}) == "/models/provise-clip"
