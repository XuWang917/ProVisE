import json

import cv2
import numpy as np

from provise.protocols import create_protocol, list_protocols
from provise.protocols.agentic_vlm import measure_visual_change
from provise.protocols.fallback import parse_fallback_json_response


def test_agentic_protocols_are_registered():
    names = set(list_protocols())

    assert "agentic_point_marker" in names
    assert "generic_vlm_fallback" in names


def test_agentic_point_marker_parser_recovers_normalized_centroid(tmp_path):
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:] = (255, 255, 255)
    cv2.circle(image, (100, 50), 5, (255, 255, 0), -1)
    path = tmp_path / "point.png"
    cv2.imwrite(str(path), image)
    protocol = create_protocol("agentic_point_marker", {"min_pixels": 5})

    parsed = protocol.parse(str(path), {"answer": [0.5, 0.5]}, "")

    assert parsed.parse_success
    assert abs(parsed.prediction[0] - 0.5) < 0.02
    assert abs(parsed.prediction[1] - 0.5) < 0.02


def test_agentic_point_marker_ignores_smaller_cyan_scene_region(tmp_path):
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    cv2.circle(image, (50, 50), 8, (255, 255, 0), -1)
    image[10:16, 170:182] = (255, 255, 0)
    path = tmp_path / "point_with_distractor.png"
    cv2.imwrite(str(path), image)
    protocol = create_protocol("agentic_point_marker", {"min_pixels": 5})

    parsed = protocol.parse(str(path), {"answer": [0.25, 0.5]}, "")

    assert parsed.parse_success
    assert abs(parsed.prediction[0] - 0.25) < 0.02
    assert abs(parsed.prediction[1] - 0.5) < 0.02
    assert parsed.extra["component_count"] == 2


def test_agentic_point_marker_scores_normalized_point_against_mask(tmp_path):
    target_mask = np.zeros((100, 200), dtype=np.uint8)
    target_mask[30:71, 80:121] = 255
    mask_path = tmp_path / "target_mask.png"
    cv2.imwrite(str(mask_path), target_mask)

    protocol = create_protocol("agentic_point_marker", {"min_pixels": 5})
    item = {
        "answer": "target region",
        "evaluation": {"mask_path": mask_path.name},
    }

    inside_image = np.full((100, 200, 3), 255, dtype=np.uint8)
    cv2.circle(inside_image, (100, 50), 5, (255, 255, 0), -1)
    inside_path = tmp_path / "inside.png"
    cv2.imwrite(str(inside_path), inside_image)
    inside_parsed = protocol.parse(str(inside_path), item, str(tmp_path))
    inside_score = protocol.score(inside_parsed, item, str(tmp_path))

    outside_image = np.full((100, 200, 3), 255, dtype=np.uint8)
    cv2.circle(outside_image, (20, 20), 5, (255, 255, 0), -1)
    outside_path = tmp_path / "outside.png"
    cv2.imwrite(str(outside_path), outside_image)
    outside_parsed = protocol.parse(str(outside_path), item, str(tmp_path))
    outside_score = protocol.score(outside_parsed, item, str(tmp_path))

    assert inside_score.is_correct
    assert inside_score.score == 1.0
    assert inside_score.extra["mask_pixel"] == 255
    assert not outside_score.is_correct
    assert outside_score.score == 0.0
    assert outside_score.extra["mask_pixel"] == 0


def test_fallback_parser_extracts_json_prediction():
    payload = """```json
    {"status": "valid", "prediction": "B", "evidence": "magenta marker in option B", "confidence": "high"}
    ```"""

    parsed = parse_fallback_json_response(payload)

    assert parsed["status"] == "valid"
    assert parsed["prediction"] == "B"
    assert parsed["confidence"] == "high"


def test_agentic_fallback_preserves_structured_point_prediction():
    protocol = create_protocol(
        "agentic_vlm_protocol",
        {
            "metric": "point_distance",
            "metric_config": {"threshold": 0.05},
            "mock_parse_response": json.dumps(
                {
                    "status": "valid",
                    "prediction": [0.5, 0.25],
                    "evidence": "a visible point marker",
                    "confidence": "high",
                }
            ),
        },
    )
    item = {
        "answer": [0.5, 0.25],
        "answer_type": "points",
        "evaluation": {"metric": "point_distance", "point": [0.5, 0.25]},
    }

    parsed = protocol.parse("unused.png", item, ".")
    score = protocol.score(parsed, item, ".")

    assert parsed.parse_success
    assert parsed.prediction == [0.5, 0.25]
    assert score.is_correct


def test_agentic_noncompliant_output_never_receives_answer_credit():
    protocol = create_protocol(
        "agentic_vlm_protocol",
        {
            "metric": "accuracy",
            "mock_parse_response": json.dumps(
                {
                    "status": "ambiguous",
                    "prediction": "A",
                    "evidence": "The required spatial placement is not clearly visible.",
                    "confidence": "low",
                }
            ),
        },
    )
    item = {
        "answer": "A",
        "answer_type": "choice",
        "evaluation": {"metric": "accuracy"},
    }

    parsed = protocol.parse("unused.png", item, ".")
    score = protocol.score(parsed, item, ".")

    assert not parsed.parse_success
    assert parsed.extra["model_protocol_noncompliance"] is True
    assert score.score == 0.0
    assert not score.is_correct


def test_agentic_fallback_readout_hides_source_question():
    protocol = create_protocol(
        "agentic_vlm_protocol",
        {
            "parse_prompt": "Read the visible relation arrow.",
            "visual_evidence": "relation arrow",
            "parser_observation": "arrow direction",
        },
    )
    item = {
        "question": "SECRET SOURCE QUESTION",
        "answer_type": "choice",
        "choices": [
            {"label": "A", "text": "left"},
            {"label": "B", "text": "right"},
        ],
    }

    prompt = protocol._parse_prompt(item)

    assert "SECRET SOURCE QUESTION" not in prompt
    assert "A. left" in prompt
    assert "relation arrow" in prompt


def test_fallback_prompt_includes_expected_answer_format():
    protocol = create_protocol("generic_vlm_fallback")
    item = {
        "question": "Which option is closest?",
        "answer_type": "choice",
        "choices": [{"label": "A", "text": "chair"}, {"label": "B", "text": "table"}],
    }

    prompt = protocol.render_prompt(
        "Instruction: {question}\nFormat: {answer_format}\nChoices: {choices}",
        item,
        "",
    )

    assert "one candidate option label from: A, B" in prompt


def test_fallback_prompt_supports_image_embedded_choice_labels():
    protocol = create_protocol("generic_vlm_fallback")
    item = {
        "question": "Which option printed in the image is correct?",
        "answer_type": "choice",
        "choices": [],
    }

    prompt = protocol.render_prompt(
        "Instruction: {question}\nFormat: {answer_format}\nChoices: {choices}",
        item,
        "",
    )

    assert "one option label printed in the image, such as A, B, C, or D" in prompt


def test_visual_change_gate_rejects_copy_and_accepts_spatial_edit(tmp_path):
    source = np.full((100, 100, 3), 255, dtype=np.uint8)
    source_path = tmp_path / "source.png"
    copy_path = tmp_path / "copy.png"
    edit_path = tmp_path / "edit.png"
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(copy_path), source)
    edited = source.copy()
    cv2.rectangle(edited, (20, 20), (50, 50), (0, 0, 255), 4)
    cv2.imwrite(str(edit_path), edited)

    copied = measure_visual_change(str(source_path), str(copy_path))
    changed = measure_visual_change(str(source_path), str(edit_path))

    assert copied["sufficient"] is False
    assert copied["visual_change_fraction"] == 0.0
    assert changed["sufficient"] is True
    assert changed["visual_change_fraction"] > changed["visual_change_min_fraction"]


def test_agentic_vlm_parser_does_not_call_vlm_for_unchanged_output(tmp_path):
    source = np.full((64, 64, 3), 255, dtype=np.uint8)
    source_path = tmp_path / "source.png"
    generated_path = tmp_path / "generated.png"
    cv2.imwrite(str(source_path), source)
    cv2.imwrite(str(generated_path), source)
    protocol = create_protocol(
        "agentic_vlm_protocol",
        {"parse_prompt": "Read the visible spatial edit and return its choice."},
    )
    item = {
        "question": "Is the cup left of the plate?",
        "answer": "A",
        "answer_type": "choice",
        "choices": [{"label": "A", "text": "yes"}, {"label": "B", "text": "no"}],
        "input": {
            "type": "image",
            "media": [{"type": "image", "path": source_path.name, "role": "primary"}],
        },
    }

    parsed = protocol.parse(str(generated_path), item, str(tmp_path))

    assert parsed.parse_success is False
    assert parsed.extra["error_type"] == "insufficient_visual_change"
    assert parsed.extra["visual_change_fraction"] == 0.0
