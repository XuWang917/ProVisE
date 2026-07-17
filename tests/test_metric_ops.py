from pathlib import Path

from PIL import Image

from provise.evaluation.metrics import score_prediction


def test_bbox_metric_accepts_structured_vlm_prediction():
    item = {
        "answer": [0.1, 0.2, 0.8, 0.9],
        "evaluation": {"metric": "bbox_iou"},
    }

    score = score_prediction(
        "bbox_iou",
        {"bbox": [0.1, 0.2, 0.8, 0.9]},
        item,
        ".",
        {"threshold": 0.5},
    )

    assert score.is_correct
    assert score.score == 1.0


def test_bbox_metric_bridges_normalized_prediction_to_pixel_ground_truth():
    item = {
        "answer": [10, 20, 80, 90],
        "evaluation": {"metric": "bbox_iou"},
        "metadata": {"width": 100, "height": 100},
    }

    score = score_prediction(
        "bbox_iou",
        {"bbox": [0.1, 0.2, 0.8, 0.9]},
        item,
        ".",
        {"threshold": 0.5},
    )

    assert score.is_correct
    assert score.score == 1.0


def test_bbox_metric_scores_matching_no_target_sentinels():
    item = {
        "answer": [0, 0, 0, 0],
        "evaluation": {"metric": "bbox_iou"},
    }

    score = score_prediction("bbox_iou", [0, 0, 0, 0], item, ".")

    assert score.is_correct
    assert score.score == 1.0


def test_smape_matches_capture_formula_and_alias():
    item = {
        "answer": 28,
        "evaluation": {"metric": "symmetric_mean_absolute_percentage_error"},
    }

    score = score_prediction(
        "symmetric_mean_absolute_percentage_error",
        20,
        item,
        ".",
    )

    assert not score.is_correct
    assert abs(score.extra["smape_percent"] - (8 / 48 * 100)) < 1e-9
    assert abs(score.score - (1 - 8 / 48)) < 1e-9


def test_point_in_mask_supports_any_and_mean_aggregation(tmp_path: Path):
    mask = Image.new("L", (10, 10), 0)
    mask.paste(255, (0, 0, 5, 10))
    mask.save(tmp_path / "mask.png")
    item = {
        "answer": None,
        "evaluation": {"metric": "point_in_mask", "mask_path": "mask.png"},
    }
    points = [[0.2, 0.5], [0.8, 0.5]]

    any_score = score_prediction(
        "point_in_mask",
        points,
        item,
        str(tmp_path),
        {"aggregation": "any"},
    )
    mean_score = score_prediction(
        "point_in_mask",
        points,
        item,
        str(tmp_path),
        {"aggregation": "mean", "correct_threshold": 0.5},
    )

    assert any_score.score == 1.0
    assert any_score.is_correct
    assert mean_score.score == 0.5
    assert mean_score.is_correct
    assert mean_score.extra["hit_rate"] == 0.5


def test_point_metric_parses_structured_literal_ground_truth():
    item = {
        "answer": "[0.4, 0.6]",
        "evaluation": {"metric": "point_distance"},
    }

    score = score_prediction("point_distance", [0.4, 0.6], item, ".", {"threshold": 0.01})

    assert score.is_correct
    assert score.extra["distance"] == 0.0


def test_dfd_metric_bridges_normalized_vlm_path_to_pixel_ground_truth(tmp_path: Path):
    Image.new("RGB", (200, 100), "white").save(tmp_path / "scene.png")
    item = {
        "answer": "[(20, 20), (100, 50), (180, 80)]",
        "input": {
            "type": "image",
            "media": [{"type": "image", "path": "scene.png", "role": "primary"}],
        },
        "evaluation": {"metric": "dfd"},
    }
    prediction = [[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]]

    score = score_prediction("dfd", prediction, item, tmp_path, {"threshold": 0.01})

    assert score.is_correct
    assert score.extra["dfd"] == 0.0


def test_state_similarity_accepts_a_recovered_choice_label():
    item = {
        "answer": "B",
        "choices": [{"label": "A", "text": "first"}, {"label": "B", "text": "second"}],
        "evaluation": {"metric": "state_similarity"},
    }

    score = score_prediction("state_similarity", "B", item, Path("."))

    assert score.is_correct
    assert score.score == 1.0


def test_accuracy_maps_default_choice_label_to_string_choice_text():
    item = {
        "answer": "the right",
        "choices": ["the left", "the right"],
        "evaluation": {"metric": "accuracy"},
    }

    score = score_prediction("accuracy", "B", item, Path("."))

    assert score.is_correct
    assert score.score == 1.0
    assert score.extra["prediction"] == "b"
    assert score.extra["ground_truth"] == "b"


def test_accuracy_maps_choice_label_to_binary_choice_text():
    item = {
        "answer": "no",
        "choices": ["no", "yes"],
        "evaluation": {"metric": "accuracy"},
    }

    score = score_prediction("accuracy", "A. no", item, Path("."))

    assert score.is_correct
    assert score.score == 1.0
