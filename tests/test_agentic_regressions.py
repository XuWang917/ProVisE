import io
import json

import cv2
import numpy as np

from provise.evaluation.metrics import score_prediction
from provise.benchmark.ingestion import (
    metric_evidence_supports,
    normalize_mapping_metric,
)
from provise.parser_ops import (
    DEFAULT_REGISTRY,
    ParserContext,
    dual_anchor_relation_pipeline,
    grounded_dimension_pipeline,
    marked_object_choice_pipeline,
)
from provise.protocol_agent.builder import (
    AgenticProtocolBuilder,
    sanitize_revision_diagnostics,
)
from provise.reporting import ProgressReporter, format_elapsed


def _numeric_item(image_name="source.png"):
    return {
        "id": "distance_1",
        "task": "distance",
        "question": "What is the distance between the box and chair?",
        "answer": 40.0,
        "answer_type": "number",
        "choices": [],
        "image_path": image_name,
        "evaluation": {
            "metric": "qspatial_ratio",
            "metric_config": {"delta": 2.0},
        },
        "metadata": {"answer_unit": "cm"},
    }


def test_progress_reporter_writes_jsonl(tmp_path):
    path = tmp_path / "progress.jsonl"
    reporter = ProgressReporter(path, enabled=False, heartbeat_seconds=0.01)
    reporter.emit("started", event="test_started", stage=1, total_stages=2)

    row = json.loads(path.read_text().strip())
    assert row["event"] == "test_started"
    assert row["stage"] == 1


def test_progress_reporter_refreshes_heartbeats_on_one_terminal_line():
    class TTYBuffer(io.StringIO):
        def isatty(self):
            return True

    stream = TTYBuffer()
    reporter = ProgressReporter(stream=stream, color=False)

    reporter.emit("Inspecting benchmark", event="benchmark_ingestion_started")
    reporter.emit(
        "Inspecting benchmark (10s elapsed)", event="benchmark_ingestion_heartbeat"
    )
    reporter.emit(
        "Inspecting benchmark (20s elapsed)", event="benchmark_ingestion_heartbeat"
    )
    reporter.emit("Inspecting benchmark done", event="benchmark_ingestion_completed")

    output = stream.getvalue()
    assert output.count("\n") == 2
    assert "\rInspecting benchmark (10s elapsed)" in output
    assert "\rInspecting benchmark (20s elapsed)" in output
    assert output.endswith("Inspecting benchmark done\n")


def test_waiting_starts_at_zero_on_the_task_line():
    class TTYBuffer(io.StringIO):
        def isatty(self):
            return True

    stream = TTYBuffer()
    reporter = ProgressReporter(stream=stream, color=False)

    with reporter.waiting("Generating image", event="image_generation", task="left"):
        pass

    output = stream.getvalue()
    assert "\r[Task: left] Generating image (0s)" in output
    assert output.endswith("[Task: left] Generating image done (0s)\n")
    assert output.count("\n") == 1


def test_elapsed_time_uses_minutes_after_sixty_seconds():
    assert format_elapsed(0) == "0s"
    assert format_elapsed(59.9) == "59s"
    assert format_elapsed(60) == "1m"
    assert format_elapsed(61) == "1m 01s"


def test_concise_progress_hides_internal_events_but_keeps_jsonl(tmp_path):
    stream = io.StringIO()
    path = tmp_path / "progress.jsonl"
    reporter = ProgressReporter(path, stream=stream, concise=True)

    reporter.emit("Loaded 100 samples", event="unified_samples_loaded")
    reporter.emit(
        "Agent decision: build",
        event="task_agent_decision",
        task="spatial_task",
    )
    reporter.emit(
        "Run finished",
        event="run_completed",
        stage=6,
        total_stages=6,
    )

    output = stream.getvalue()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert "Loaded 100 samples" not in output
    assert "Agent decision: build" in output
    assert "[6/6] Run finished" in output
    assert [row["event"] for row in rows] == [
        "unified_samples_loaded",
        "task_agent_decision",
        "run_completed",
    ]


def test_progress_reporter_colors_statuses_only_on_terminal(monkeypatch, tmp_path):
    class TTYBuffer(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("CLICOLOR", raising=False)
    stream = TTYBuffer()
    path = tmp_path / "progress.jsonl"
    reporter = ProgressReporter(path, stream=stream, concise=False, color=True)

    reporter.emit("PASSED", event="task_workflow_completed", status="completed", task="left")
    reporter.emit("FAILED", event="task_workflow_completed", status="failed", task="right")

    output = stream.getvalue()
    assert "\033[32mPASSED\033[0m" in output
    assert "\033[31mFAILED\033[0m" in output
    assert "\033[" not in path.read_text(encoding="utf-8")


def test_no_color_environment_disables_ansi(monkeypatch):
    class TTYBuffer(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setenv("NO_COLOR", "1")
    stream = TTYBuffer()
    reporter = ProgressReporter(stream=stream, concise=False, color=True)

    reporter.emit("FAILED", event="task_workflow_completed", status="failed", task="left")

    assert "\033[" not in stream.getvalue()


def test_qspatial_ratio_metric_normalizes_units():
    result = score_prediction(
        "qspatial_ratio",
        {"value": 0.5, "unit": "m"},
        _numeric_item(),
        "",
        {"delta": 2.0},
    )

    assert result.is_correct
    assert result.extra["prediction_cm"] == 50.0
    assert result.extra["ground_truth_cm"] == 40.0


def test_builder_compiles_grounded_dimension_contract():
    response = {
        "task": "distance",
        "decision": "build",
        "build_mode": "recipe",
        "confidence": "high",
        "reason": "A grounded dimension annotation preserves the metric estimate.",
        "visual_contract": {
            "recipe": "grounded_dimension",
            "mode": "edit_source",
            "primitives": ["two object outlines", "dimension line", "numeric unit label"],
            "parameters": {"anchor_count": 2, "unit": "cm"},
        },
        "readout": {"recipe": "grounded_dimension"},
    }
    result = AgenticProtocolBuilder(
        [_numeric_item()],
        benchmark_name="toy",
        data_file="toy.jsonl",
        benchmark_root="assets",
    ).build(raw_response=json.dumps(response))

    cfg = result.benchmark_config["tasks"]["distance"]
    route = result.manifest["route_rows"][0]
    assert cfg["protocol"] == "agentic_parser_ops_protocol"
    assert cfg["protocol_config"]["parser_output_kind"] == "scalar_measurement"
    assert cfg["formal_evaluation"] is True
    assert route["decision"] == "build"
    assert route["build_mode"] == "recipe"
    assert route["recipe"] == "grounded_dimension"
    assert "{answer}" not in cfg["prompt"]


def test_builder_compiles_multi_image_choice_count_board_contract():
    item = {
        "id": "count_1",
        "task": "scene_count",
        "question": "How many chairs are visible across the room views?",
        "answer": "B",
        "answer_type": "choice",
        "choices": [
            {"label": "A", "text": "1"},
            {"label": "B", "text": "3"},
            {"label": "C", "text": "5"},
        ],
        "input": {
            "type": "multi_image",
            "media": [
                {"type": "image", "path": "view_1.png", "role": "view"},
                {"type": "image", "path": "view_2.png", "role": "view"},
            ],
        },
        "evaluation": {"metric": "accuracy"},
    }
    response = {
        "task": "scene_count",
        "decision": "build",
        "build_mode": "recipe",
        "confidence": "high",
        "reason": "A visible counted evidence board preserves the numeric choice contract.",
        "visual_contract": {
            "recipe": "choice_count_board",
            "mode": "reference_synthesis",
            "primitives": ["one target tile and one marker per distinct counted chair"],
            "parameters": {},
        },
        "readout": {"recipe": "choice_count_board"},
    }

    result = AgenticProtocolBuilder(
        [item],
        benchmark_name="toy",
        data_file="toy.jsonl",
        benchmark_root="assets",
    ).build(raw_response=json.dumps(response))

    cfg = result.benchmark_config["tasks"]["scene_count"]
    route = result.manifest["route_rows"][0]
    assert cfg["protocol"] == "agentic_parser_ops_protocol"
    assert cfg["input"]["mode"] == "metadata_images"
    assert cfg["protocol_config"]["parser_output_kind"] == "choice_label"
    assert cfg["protocol_config"]["parser_pipeline"]["output"] == "choice"
    assert cfg["formal_evaluation"] is True
    assert route["recipe"] == "choice_count_board"


def test_builder_rejects_dual_anchor_relation_for_boolean_choices():
    item = {
        "id": "feasibility_1",
        "task": "feasibility",
        "question": "Can the woman stand on the man's left side?",
        "answer": "B",
        "answer_type": "choice",
        "choices": [
            {"label": "A", "text": "no"},
            {"label": "B", "text": "yes"},
        ],
        "image_path": "scene.png",
        "evaluation": {"metric": "accuracy"},
    }
    response = {
        "task": "feasibility",
        "decision": "build",
        "build_mode": "recipe",
        "confidence": "high",
        "reason": "Mark subject and reference anchors.",
        "visual_contract": {
            "recipe": "dual_anchor_relation",
            "mode": "edit_source",
            "primitives": ["subject anchor", "reference anchor"],
            "parameters": {"relation_mode": "horizontal"},
        },
        "readout": {"recipe": "dual_anchor_relation"},
    }

    result = AgenticProtocolBuilder(
        [item], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    ).build(raw_response=json.dumps(response))

    assert result.benchmark_config["tasks"] == {}
    route = result.manifest["route_rows"][0]
    assert route["source"] == "contract_compiler"
    assert "binary_boolean" in route["reason"]


def test_recipe_parameter_description_is_safely_normalized():
    response = {
        "task": "distance",
        "decision": "build",
        "build_mode": "recipe",
        "confidence": "high",
        "reason": "A grounded dimension annotation preserves the metric estimate.",
        "visual_contract": {
            "recipe": "grounded_dimension",
            "mode": "edit_source",
            "primitives": ["two object outlines", "dimension line", "numeric unit label"],
            "parameters": {"anchor_count": "2 for distance between objects", "unit": "cm"},
        },
        "readout": {"recipe": "grounded_dimension"},
    }

    result = AgenticProtocolBuilder(
        [_numeric_item()], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    ).build(raw_response=json.dumps(response))

    config = result.benchmark_config["tasks"]["distance"]
    assert config["protocol_config"]["anchor_count"] == 2


def test_invalid_recipe_parameter_becomes_a_task_diagnostic_instead_of_crashing():
    response = {
        "task": "distance",
        "decision": "build",
        "build_mode": "recipe",
        "confidence": "high",
        "reason": "A grounded dimension annotation preserves the metric estimate.",
        "visual_contract": {
            "recipe": "grounded_dimension",
            "mode": "edit_source",
            "primitives": ["dimension line"],
            "parameters": {"anchor_count": "all objects"},
        },
        "readout": {"recipe": "grounded_dimension"},
    }

    result = AgenticProtocolBuilder(
        [_numeric_item()], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    ).build(raw_response=json.dumps(response))

    assert result.benchmark_config["tasks"] == {}
    route = result.manifest["route_rows"][0]
    assert route["source"] == "contract_compiler"
    assert "anchor_count" in route["reason"]


def test_builder_rejects_unregistered_direct_pipeline():
    response = {
        "task": "distance",
        "decision": "build",
        "build_mode": "parser_ops",
        "confidence": "high",
        "reason": "Custom geometry readout.",
        "visual_contract": {
            "mode": "edit_source",
            "primitives": ["dimension line"],
        },
        "generation_prompt": "For {question}, draw a dimension line.",
        "readout": {
            "output_kind": "scalar_measurement",
            "pipeline": {
                "steps": [{"id": "answer", "op": "execute_python"}],
                "output": "answer",
            },
        },
    }
    result = AgenticProtocolBuilder(
        [_numeric_item()],
        benchmark_name="toy",
        data_file="toy.jsonl",
        benchmark_root="assets",
    ).build(raw_response=json.dumps(response))

    assert result.benchmark_config["tasks"] == {}
    assert result.manifest["route_rows"][0]["source"] == "contract_compiler"


def test_grounded_dimension_round_trip_with_local_ocr(monkeypatch, tmp_path):
    source_path = tmp_path / "source.png"
    generated_path = tmp_path / "generated.png"
    source = np.full((180, 320, 3), 245, dtype=np.uint8)
    generated = source.copy()
    cv2.rectangle(generated, (25, 55), (95, 135), (255, 0, 255), 7)
    cv2.rectangle(generated, (225, 55), (295, 135), (255, 255, 0), 7)
    cv2.line(generated, (100, 95), (220, 95), (0, 255, 255), 6)
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(generated_path), generated)
    monkeypatch.setattr(
        "provise.parser_ops.operators._run_local_ocr",
        lambda image: [("42 cm", 0.99, [])],
    )

    result = DEFAULT_REGISTRY.execute(
        grounded_dimension_pipeline(anchor_count=2),
        ParserContext(
            str(generated_path),
            _numeric_item(source_path.name),
            str(tmp_path),
            source_paths=(str(source_path),),
            protocol_config={"unit": "cm"},
        ),
    )

    assert result.success, result.error
    assert result.prediction == {"value": 42.0, "unit": "cm"}
    assert result.diagnostics["steps"]["grounded_measurement"]["grounded"]


def test_grounded_dimension_prefers_full_image_ocr_with_explicit_unit(
    monkeypatch, tmp_path
):
    source_path = tmp_path / "source.png"
    generated_path = tmp_path / "generated.png"
    source = np.full((600, 1000, 3), 245, dtype=np.uint8)
    generated = source.copy()
    cv2.rectangle(generated, (50, 180), (220, 420), (255, 0, 255), 9)
    cv2.rectangle(generated, (780, 180), (950, 420), (255, 255, 0), 9)
    cv2.line(generated, (230, 300), (770, 300), (0, 255, 255), 8)
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(generated_path), generated)
    calls = []

    def fake_ocr(image):
        calls.append(image.shape[:2])
        if len(calls) == 1:
            return [("1", 0.99, [])]
        return [("15cm", 0.95, [])]

    monkeypatch.setattr("provise.parser_ops.operators._run_local_ocr", fake_ocr)
    result = DEFAULT_REGISTRY.execute(
        grounded_dimension_pipeline(anchor_count=2),
        ParserContext(
            str(generated_path),
            _numeric_item(source_path.name),
            str(tmp_path),
            source_paths=(str(source_path),),
            protocol_config={"unit": "cm"},
        ),
    )

    assert result.success, result.error
    assert result.prediction == {"value": 15.0, "unit": "cm"}
    assert result.diagnostics["steps"]["measurement"]["ocr_source"] == "full_generated_image"
    assert len(calls) == 2


def test_grounded_dimension_recovers_number_next_to_split_unit_token(
    monkeypatch, tmp_path
):
    source_path = tmp_path / "source.png"
    generated_path = tmp_path / "generated.png"
    source = np.full((600, 1000, 3), 245, dtype=np.uint8)
    generated = source.copy()
    cv2.rectangle(generated, (50, 180), (220, 420), (255, 0, 255), 9)
    cv2.rectangle(generated, (780, 180), (950, 420), (255, 255, 0), 9)
    cv2.line(generated, (230, 300), (770, 300), (0, 255, 255), 8)
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(generated_path), generated)
    calls = []

    def fake_ocr(image):
        calls.append(image.shape[:2])
        if len(calls) == 1:
            return []
        if len(calls) == 2:
            return [
                ("605", 0.99, [[20, 20], [60, 20], [60, 40], [20, 40]]),
                ("cm", 0.99, [[700, 300], [750, 300], [750, 330], [700, 330]]),
            ]
        return [("7 cm", 0.95, [])]

    monkeypatch.setattr("provise.parser_ops.operators._run_local_ocr", fake_ocr)
    result = DEFAULT_REGISTRY.execute(
        grounded_dimension_pipeline(anchor_count=2),
        ParserContext(
            str(generated_path),
            _numeric_item(source_path.name),
            str(tmp_path),
            source_paths=(str(source_path),),
            protocol_config={"unit": "cm"},
        ),
    )

    assert result.success, result.error
    assert result.prediction == {"value": 7.0, "unit": "cm"}
    assert result.diagnostics["steps"]["measurement"]["ocr_source"] == "unit_neighborhood"
    assert len(calls) == 3


def test_grounded_dimension_uses_geometric_line_fallback(monkeypatch, tmp_path):
    source_path = tmp_path / "source.png"
    generated_path = tmp_path / "generated.png"
    source = np.full((600, 1000, 3), 245, dtype=np.uint8)
    generated = source.copy()
    cv2.rectangle(generated, (50, 180), (220, 420), (255, 0, 255), 9)
    cv2.rectangle(generated, (780, 180), (950, 420), (255, 255, 0), 9)
    cv2.line(generated, (230, 300), (770, 300), (0, 255, 255), 8)
    cv2.rectangle(generated, (450, 380), (550, 480), (0, 255, 255), -1)
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(generated_path), generated)
    monkeypatch.setattr(
        "provise.parser_ops.operators._run_local_ocr",
        lambda image: [("42 cm", 0.99, [])],
    )

    result = DEFAULT_REGISTRY.execute(
        grounded_dimension_pipeline(anchor_count=2),
        ParserContext(
            str(generated_path),
            _numeric_item(source_path.name),
            str(tmp_path),
            source_paths=(str(source_path),),
            protocol_config={"unit": "cm"},
        ),
    )

    assert result.success, result.error
    diagnostics = result.diagnostics["steps"]["grounded_measurement"]
    assert diagnostics["line_evidence"] == "source_aware_hough"
    assert diagnostics["hough_longest_line"] >= 500


def test_grounded_dimension_rejects_yellow_block_without_line(monkeypatch, tmp_path):
    source_path = tmp_path / "source.png"
    generated_path = tmp_path / "generated.png"
    source = np.full((600, 1000, 3), 245, dtype=np.uint8)
    generated = source.copy()
    cv2.rectangle(generated, (50, 180), (220, 420), (255, 0, 255), 9)
    cv2.rectangle(generated, (780, 180), (950, 420), (255, 255, 0), 9)
    cv2.rectangle(generated, (450, 250), (550, 350), (0, 255, 255), -1)
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(generated_path), generated)
    monkeypatch.setattr(
        "provise.parser_ops.operators._run_local_ocr",
        lambda image: [("42 cm", 0.99, [])],
    )

    result = DEFAULT_REGISTRY.execute(
        grounded_dimension_pipeline(anchor_count=2),
        ParserContext(
            str(generated_path),
            _numeric_item(source_path.name),
            str(tmp_path),
            source_paths=(str(source_path),),
            protocol_config={"unit": "cm"},
        ),
    )

    assert not result.success
    assert result.error_type == "invalid_dimension_line"


def test_dimension_label_crop_expands_across_the_minor_axis(tmp_path):
    path = tmp_path / "vertical_dimension.png"
    image = np.full((800, 600, 3), 245, dtype=np.uint8)
    cv2.line(image, (300, 180), (300, 620), (0, 255, 255), 12)
    cv2.putText(image, "46 cm", (340, 410), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
    cv2.imwrite(str(path), image)
    pipeline = {
        "steps": [
            {"id": "image", "op": "load_generated"},
            {
                "id": "mask",
                "op": "hsv_color_mask",
                "inputs": ["image"],
                "params": {"lower": [18, 80, 100], "upper": [40, 255, 255]},
            },
            {"id": "components", "op": "connected_components", "inputs": ["mask"]},
            {"id": "line", "op": "select_largest", "inputs": ["components"]},
            {
                "id": "crop",
                "op": "crop_dimension_label",
                "inputs": ["image", "line"],
            },
        ],
        "output": "crop",
    }

    result = DEFAULT_REGISTRY.execute(pipeline, ParserContext(str(path), {}, ""))

    assert result.success, result.error
    diagnostics = result.diagnostics["steps"]["crop"]
    assert diagnostics["dimension_orientation"] == "vertical"
    assert diagnostics["width"] >= 512
    assert result.prediction.shape[1] >= 512


def test_dual_anchor_relation_maps_to_choice(tmp_path):
    source_path = tmp_path / "source.png"
    generated_path = tmp_path / "generated.png"
    source = np.full((150, 300, 3), 245, dtype=np.uint8)
    generated = source.copy()
    cv2.rectangle(generated, (20, 45), (90, 120), (255, 0, 255), 7)
    cv2.rectangle(generated, (210, 45), (280, 120), (255, 255, 0), 7)
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(generated_path), generated)
    item = {
        "answer": "A",
        "image_path": source_path.name,
        "choices": [
            {"label": "A", "text": "The cup is left of the chair."},
            {"label": "B", "text": "The cup is right of the chair."},
        ],
    }

    result = DEFAULT_REGISTRY.execute(
        dual_anchor_relation_pipeline(mode="horizontal", map_to_choices=True),
        ParserContext(
            str(generated_path),
            item,
            str(tmp_path),
            source_paths=(str(source_path),),
        ),
    )

    assert result.success, result.error
    assert result.prediction == "A"


def test_marked_object_choice_crops_source_evidence_and_maps_with_fixed_clip(
    monkeypatch, tmp_path
):
    source_path = tmp_path / "source.png"
    generated_path = tmp_path / "generated.png"
    source = np.full((180, 300, 3), 245, dtype=np.uint8)
    cv2.rectangle(source, (30, 45), (115, 145), (60, 60, 60), -1)
    cv2.circle(source, (230, 95), 45, (130, 130, 130), -1)
    generated = source.copy()
    cv2.circle(generated, (230, 95), 52, (255, 0, 255), 8)
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(generated_path), generated)

    class FakeClip:
        def encode(self, values, normalize_embeddings=True):
            assert len(values) == 3
            return np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    monkeypatch.setattr(
        "provise.parser_ops.operators._load_clip_model", lambda model_name: FakeClip()
    )
    item = {
        "answer": "B",
        "image_path": source_path.name,
        "choices": [
            {"label": "A", "text": "box"},
            {"label": "B", "text": "ball"},
        ],
    }

    result = DEFAULT_REGISTRY.execute(
        marked_object_choice_pipeline(),
        ParserContext(
            str(generated_path),
            item,
            str(tmp_path),
            source_paths=(str(source_path),),
        ),
    )

    assert result.success, result.error
    assert result.prediction == "B"
    assert result.diagnostics["steps"]["object_crop"]["width"] >= 96


def test_metric_evidence_downgrades_an_unsupported_claim():
    mapping = {"evaluation": {"metric": "bbox_iou"}}
    evidence = {
        "metric:evaluate.py": {
            "excerpt": "correct = prediction == answer; accuracy = correct / total"
        }
    }

    normalize_mapping_metric(mapping, evidence)

    assert mapping["evaluation"]["metric"] == "unverified"
    assert metric_evidence_supports(
        "bbox_iou", "Compute bounding box IoU as intersection over union."
    )
    assert metric_evidence_supports(
        "point_distance", "The Euclidean distance must be below the distance threshold."
    )
    assert metric_evidence_supports(
        "angle_error", "Report angular error and apply a degree threshold."
    )
    assert metric_evidence_supports(
        "point_in_mask", "Point-in-mask checking against the ground-truth region."
    )


def test_unverified_metric_keeps_smoke_protocol_but_blocks_formal_scoring():
    item = _numeric_item()
    item["evaluation"]["metric"] = "unverified"
    response = {
        "task": "distance",
        "decision": "build",
        "build_mode": "recipe",
        "confidence": "high",
        "reason": "A visible dimension annotation exposes the estimate.",
        "visual_contract": {
            "recipe": "grounded_dimension",
            "mode": "edit_source",
            "primitives": ["two object outlines", "dimension line", "measurement label"],
            "parameters": {"anchor_count": 2, "unit": "cm"},
        },
        "readout": {"recipe": "grounded_dimension"},
    }

    result = AgenticProtocolBuilder(
        [item], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    ).build(raw_response=json.dumps(response))

    config = result.benchmark_config["tasks"]["distance"]
    assert config["protocol"] == "agentic_parser_ops_protocol"
    assert config["formal_evaluation"] is False


def test_task_level_vlm_fallback_is_available_without_an_opt_in_flag():
    item = {
        "id": "spatial_1",
        "task": "occluded_layout",
        "question": "Which hidden layout is spatially consistent?",
        "answer": "A",
        "answer_type": "choice",
        "choices": [
            {"label": "A", "text": "The red object remains behind the wall."},
            {"label": "B", "text": "The red object moves in front of the wall."},
        ],
        "image_path": "source.png",
        "evaluation": {"metric": "accuracy"},
    }
    response = {
        "task": "occluded_layout",
        "decision": "fallback",
        "confidence": "high",
        "reason": "No deterministic operator can recover the occluded 3D layout.",
        "fallback": {
            "generation_prompt": (
                "For {question}, edit the source to show the inferred hidden object layout with "
                "visible object contours and one depth-order arrow. Do not write an option label."
            ),
            "parse_prompt": (
                "Read only the visible object contours and depth-order arrow in the generated image; "
                "return the matching choice as JSON."
            ),
            "visual_evidence": "object contours and a depth-order arrow",
            "invalid_conditions": [
                "the hidden object is not visualized",
                "the depth-order arrow is missing or ambiguous",
            ],
        },
    }

    result = AgenticProtocolBuilder(
        [item], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    ).build(raw_response=json.dumps(response))

    config = result.benchmark_config["tasks"]["occluded_layout"]
    route = result.manifest["route_rows"][0]
    assert config["protocol"] == "agentic_vlm_protocol"
    assert config["prompt"].endswith("Candidate answers:\n{choices}")
    assert config["protocol_config"]["include_source_images"] is False
    assert route["parser_backend"] == "vlm_fallback"
    assert route["parser_inputs"] == ["generated_response"]
    assert route["active"] is True


def test_revision_diagnostics_never_expose_correctness_or_ground_truth():
    sanitized = sanitize_revision_diagnostics(
        {
            "failed_checks": ["parse_success_rate"],
            "score": 0.0,
            "ground_truth": "A",
            "nested": {"is_correct": False, "error_type": "marker_missing"},
        }
    )

    assert sanitized == {
        "failed_checks": ["parse_success_rate"],
        "nested": {"error_type": "marker_missing"},
    }
