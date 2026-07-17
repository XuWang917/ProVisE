from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import cv2
import numpy as np


METRIC_ALIASES = {
    "acc": "accuracy",
    "exact": "exact_match",
    "exact_count": "exact_count",
    "numeric_error": "numeric_absolute_error",
    "q_spatial_success": "qspatial_ratio",
    "q-spatial": "qspatial_ratio",
    "iou": "mask_iou",
    "precision": "mask_precision",
    "symmetric_mean_absolute_percentage_error": "smape",
}

METRIC_OUTPUT_KINDS: Dict[str, frozenset[str]] = {
    "accuracy": frozenset({"text", "label", "choice_label", "boolean", "integer_count"}),
    "exact_match": frozenset({"text", "label", "choice_label", "boolean", "integer_count"}),
    "exact_count": frozenset({"integer_count"}),
    "smape": frozenset({"integer_count", "scalar", "scalar_measurement"}),
    "qspatial_ratio": frozenset({"scalar", "scalar_measurement"}),
    "numeric_absolute_error": frozenset({"scalar", "scalar_measurement"}),
    "numeric_relative_error": frozenset({"scalar", "scalar_measurement"}),
    "point_in_mask": frozenset({"normalized_point", "normalized_points"}),
    "point_distance": frozenset({"normalized_point", "normalized_points"}),
    "bbox_iou": frozenset({"normalized_bbox"}),
    "angle_error": frozenset({"scalar"}),
    "mask_precision": frozenset({"binary_mask", "mask_path"}),
    "mask_iou": frozenset({"binary_mask", "mask_path"}),
    "mra": frozenset({"scalar", "scalar_measurement"}),
    "dfd": frozenset({"normalized_polyline"}),
    "state_similarity": frozenset({"image_path", "choice_label"}),
    "unverified": frozenset(
        {
            "text",
            "label",
            "choice_label",
            "boolean",
            "integer_count",
            "scalar",
            "scalar_measurement",
            "normalized_point",
            "normalized_points",
            "normalized_bbox",
            "binary_mask",
            "mask_path",
            "normalized_polyline",
            "image_path",
        }
    ),
}

UNIT_TO_CENTIMETERS = {
    "m": 100.0,
    "meter": 100.0,
    "meters": 100.0,
    "metre": 100.0,
    "metres": 100.0,
    "cm": 1.0,
    "centimeter": 1.0,
    "centimeters": 1.0,
    "centimetre": 1.0,
    "centimetres": 1.0,
    "mm": 0.1,
    "millimeter": 0.1,
    "millimeters": 0.1,
    "ft": 30.48,
    "foot": 30.48,
    "feet": 30.48,
    "in": 2.54,
    "inch": 2.54,
    "inches": 2.54,
}


@dataclass(frozen=True)
class MetricScore:
    score: float
    is_correct: bool
    extra: Dict[str, Any] = field(default_factory=dict)


def normalize_metric_name(value: Any) -> str:
    name = str(value or "unverified").strip().lower().replace("-", "_").replace(" ", "_")
    return METRIC_ALIASES.get(name, name)


def metric_accepts_output(metric: str, output_kind: str) -> bool:
    accepted = METRIC_OUTPUT_KINDS.get(normalize_metric_name(metric))
    return bool(accepted and str(output_kind) in accepted)


def score_prediction(
    metric: str,
    prediction: Any,
    item: Mapping[str, Any],
    benchmark_root: str | Path,
    config: Mapping[str, Any] | None = None,
) -> MetricScore:
    metric_name = normalize_metric_name(metric)
    cfg = dict(config or {})
    ground_truth = item.get("answer")
    if metric_name in {"accuracy", "exact_match", "exact_count"}:
        pred = _normalize_label(prediction, item)
        gt = _normalize_label(ground_truth, item)
        correct = pred == gt
        return MetricScore(
            1.0 if correct else 0.0,
            correct,
            {"prediction": pred, "ground_truth": gt, "metric": metric_name},
        )

    if metric_name == "qspatial_ratio":
        pred_value, pred_unit = _measurement(prediction, cfg.get("unit"))
        gt_unit = (
            (item.get("metadata") or {}).get("answer_unit")
            or (item.get("evaluation") or {}).get("unit")
            or cfg.get("ground_truth_unit")
        )
        gt_value, gt_unit = _measurement(ground_truth, gt_unit)
        pred_cm = _to_centimeters(pred_value, pred_unit)
        gt_cm = _to_centimeters(gt_value, gt_unit)
        if pred_cm is None or gt_cm is None or pred_cm <= 0 or gt_cm <= 0:
            return MetricScore(
                0.0,
                False,
                {
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "metric": metric_name,
                    "error": "invalid_numeric_measurement",
                },
            )
        ratio = max(pred_cm / gt_cm, gt_cm / pred_cm)
        delta = float(cfg.get("delta", 2.0))
        correct = ratio < delta
        score = max(0.0, min(1.0, 1.0 - (ratio - 1.0) / max(delta - 1.0, 1e-9)))
        return MetricScore(
            score,
            correct,
            {
                "prediction": pred_cm,
                "ground_truth": gt_cm,
                "prediction_cm": pred_cm,
                "ground_truth_cm": gt_cm,
                "ratio": ratio,
                "delta": delta,
                "metric": metric_name,
            },
        )

    if metric_name in {"numeric_absolute_error", "numeric_relative_error", "mra"}:
        pred_value, _ = _measurement(prediction, cfg.get("unit"))
        gt_value, _ = _measurement(ground_truth, cfg.get("ground_truth_unit"))
        if pred_value is None or gt_value is None:
            return MetricScore(0.0, False, {"error": "non_numeric", "metric": metric_name})
        absolute_error = abs(pred_value - gt_value)
        relative_error = absolute_error / max(abs(gt_value), float(cfg.get("epsilon", 1e-8)))
        tolerance = float(cfg.get("tolerance", 0.0))
        if metric_name == "mra":
            score = max(0.0, 1.0 - relative_error)
            correct = relative_error <= float(cfg.get("correct_relative_error", 0.5))
        elif metric_name == "numeric_relative_error":
            score = max(0.0, 1.0 - relative_error)
            correct = relative_error <= tolerance
        else:
            scale = float(cfg.get("score_scale", max(abs(gt_value), 1.0)))
            score = max(0.0, 1.0 - absolute_error / max(scale, 1e-8))
            correct = absolute_error <= tolerance
        return MetricScore(
            score,
            correct,
            {
                "prediction": pred_value,
                "ground_truth": gt_value,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "tolerance": tolerance,
                "metric": metric_name,
            },
        )

    if metric_name == "smape":
        pred_value, _ = _measurement(prediction, cfg.get("unit"))
        gt_value, _ = _measurement(ground_truth, cfg.get("ground_truth_unit"))
        if pred_value is None or gt_value is None:
            return MetricScore(
                0.0,
                False,
                {
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "smape_percent": 100.0,
                    "error": "non_numeric",
                    "metric": metric_name,
                },
            )
        denominator = abs(pred_value) + abs(gt_value)
        smape_percent = (
            abs(pred_value - gt_value) / denominator * 100.0
            if denominator > 0
            else 0.0
        )
        threshold = float(cfg.get("correct_smape_percent", 0.0))
        return MetricScore(
            max(0.0, 1.0 - smape_percent / 100.0),
            smape_percent <= threshold,
            {
                "prediction": pred_value,
                "ground_truth": gt_value,
                "smape_percent": smape_percent,
                "correct_smape_percent": threshold,
                "metric": metric_name,
            },
        )

    if metric_name == "point_in_mask":
        points = _normalized_points(prediction)
        mask_path = str((item.get("evaluation") or {}).get("mask_path") or "").strip()
        mask = cv2.imread(_resolve_path(benchmark_root, mask_path), cv2.IMREAD_GRAYSCALE) if mask_path else None
        if mask is None or not points:
            return MetricScore(0.0, False, {"error": "mask_or_points_unavailable", "metric": metric_name})
        aggregation = str(cfg.get("aggregation") or "any").strip().lower()
        if aggregation not in {"any", "mean"}:
            return MetricScore(
                0.0,
                False,
                {
                    "error": "unsupported_point_aggregation",
                    "aggregation": aggregation,
                    "metric": metric_name,
                },
            )
        height, width = mask.shape[:2]
        hits = []
        for x, y in points:
            px = min(max(int(round(x * max(0, width - 1))), 0), max(0, width - 1))
            py = min(max(int(round(y * max(0, height - 1))), 0), max(0, height - 1))
            hits.append(bool(mask[py, px] > 0))
        hit_rate = sum(hits) / len(hits)
        score = 1.0 if any(hits) else 0.0
        if aggregation == "mean":
            score = hit_rate
        correct_threshold = float(cfg.get("correct_threshold", 1.0))
        correct = score >= correct_threshold
        return MetricScore(
            score,
            correct,
            {
                "prediction": points,
                "ground_truth": mask_path,
                "point_hits": hits,
                "hit_rate": hit_rate,
                "aggregation": aggregation,
                "correct_threshold": correct_threshold,
                "metric": metric_name,
            },
        )

    if metric_name == "point_distance":
        predicted = _normalized_points(prediction)
        target = _normalized_points(
            (item.get("evaluation") or {}).get("point") or item.get("answer")
        )
        if not predicted or not target:
            return MetricScore(0.0, False, {"error": "points_unavailable", "metric": metric_name})
        distance = float(np.linalg.norm(np.asarray(predicted[0]) - np.asarray(target[0])))
        threshold = float(cfg.get("threshold", 0.05))
        return MetricScore(
            max(0.0, 1.0 - distance / max(threshold, 1e-9)),
            distance <= threshold,
            {"prediction": predicted[0], "ground_truth": target[0], "distance": distance, "threshold": threshold},
        )

    if metric_name == "bbox_iou":
        pred_box = _bbox_value(prediction)
        gt_box = _bbox_value((item.get("evaluation") or {}).get("bbox") or item.get("answer"))
        if pred_box is None or gt_box is None:
            return MetricScore(0.0, False, {"error": "bbox_unavailable", "metric": metric_name})
        pred_box = _normalize_bbox_scale(pred_box, item)
        gt_box = _normalize_bbox_scale(gt_box, item)
        pred_empty = _empty_bbox(pred_box)
        gt_empty = _empty_bbox(gt_box)
        if pred_empty or gt_empty:
            iou = 1.0 if pred_empty and gt_empty else 0.0
        else:
            intersection = max(
                0.0,
                min(pred_box[2], gt_box[2]) - max(pred_box[0], gt_box[0]),
            ) * max(0.0, min(pred_box[3], gt_box[3]) - max(pred_box[1], gt_box[1]))
            pred_area = max(0.0, pred_box[2] - pred_box[0]) * max(
                0.0, pred_box[3] - pred_box[1]
            )
            gt_area = max(0.0, gt_box[2] - gt_box[0]) * max(
                0.0, gt_box[3] - gt_box[1]
            )
            iou = intersection / max(pred_area + gt_area - intersection, 1e-9)
        threshold = float(cfg.get("threshold", 0.5))
        return MetricScore(iou, iou >= threshold, {"prediction": pred_box, "ground_truth": gt_box, "iou": iou})

    if metric_name == "angle_error":
        pred_value, _ = _measurement(prediction)
        gt_value, _ = _measurement(ground_truth)
        if pred_value is None or gt_value is None:
            return MetricScore(0.0, False, {"error": "angle_unavailable", "metric": metric_name})
        difference = abs((pred_value - gt_value + 180.0) % 360.0 - 180.0)
        threshold = float(cfg.get("threshold_degrees", 15.0))
        return MetricScore(
            max(0.0, 1.0 - difference / 180.0),
            difference <= threshold,
            {"prediction": pred_value, "ground_truth": gt_value, "angle_error": difference, "threshold": threshold},
        )

    if metric_name == "dfd":
        predicted = _normalized_polyline(prediction, item, benchmark_root)
        target = _normalized_polyline(
            (item.get("evaluation") or {}).get("trajectory") or ground_truth,
            item,
            benchmark_root,
        )
        if not predicted or not target:
            return MetricScore(0.0, False, {"error": "trajectory_unavailable", "metric": metric_name})
        distance = _discrete_frechet(predicted, target)
        threshold = float(cfg.get("threshold", cfg.get("success_dfd", 0.4)))
        return MetricScore(
            max(0.0, 1.0 - distance),
            distance < threshold,
            {
                "prediction": predicted,
                "ground_truth": target,
                "dfd": distance,
                "threshold": threshold,
                "metric": metric_name,
            },
        )

    if metric_name == "state_similarity":
        pred = _normalize_label(prediction, item)
        gt = _normalize_label(ground_truth, item)
        correct = bool(pred) and pred == gt
        return MetricScore(
            1.0 if correct else 0.0,
            correct,
            {"prediction": pred, "ground_truth": gt, "metric": metric_name},
        )

    if metric_name in {"mask_precision", "mask_iou"}:
        pred_mask = _prediction_mask(prediction)
        target_path = str((item.get("evaluation") or {}).get("mask_path") or "").strip()
        gt_mask = cv2.imread(_resolve_path(benchmark_root, target_path), cv2.IMREAD_GRAYSCALE) if target_path else None
        if pred_mask is None or gt_mask is None:
            return MetricScore(0.0, False, {"error": "mask_unavailable", "metric": metric_name})
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        pred_bool = pred_mask > 0
        gt_bool = gt_mask > 0
        intersection = int(np.count_nonzero(pred_bool & gt_bool))
        if metric_name == "mask_precision":
            value = intersection / max(1, int(np.count_nonzero(pred_bool)))
        else:
            value = intersection / max(1, int(np.count_nonzero(pred_bool | gt_bool)))
        threshold = float(cfg.get("threshold", 0.5))
        return MetricScore(
            float(value),
            value >= threshold,
            {"prediction": "mask", "ground_truth": target_path, metric_name: value, "threshold": threshold},
        )

    if metric_name == "unverified":
        return MetricScore(
            0.0,
            False,
            {
                "prediction": prediction,
                "ground_truth": ground_truth,
                "metric": metric_name,
                "metric_unverified": True,
            },
        )
    raise ValueError(f"Unsupported metric adapter: {metric_name}")


def _normalize_label(value: Any, item: Mapping[str, Any]) -> str:
    if isinstance(value, Mapping):
        value = value.get("label", value.get("value", value.get("prediction", value)))
    choices = list(item.get("choices") or item.get("options") or [])
    if isinstance(value, bool):
        value = "yes" if value else "no"
    if isinstance(value, int) and choices:
        if 0 <= value < len(choices):
            return _choice_rows(choices)[value][0]
    text = _normalize_label_text(value)
    for label, choice_text in _choice_rows(choices):
        if text in {label, choice_text}:
            return label
    return text


def _choice_rows(choices: list[Any]) -> list[tuple[str, str]]:
    rows = []
    for index, choice in enumerate(choices):
        default_label = chr(ord("a") + index) if index < 26 else str(index + 1)
        if isinstance(choice, Mapping):
            raw_label = choice.get("label")
            raw_text = choice.get("text", choice.get("value", raw_label))
        else:
            raw_label = default_label
            raw_text = choice
        label = _normalize_label_text(raw_label or default_label)
        rows.append((label, _normalize_label_text(raw_text)))
    return rows


def _normalize_label_text(value: Any) -> str:
    text = str(value if value is not None else "").strip()
    if re.match(r"^[A-Za-z][\).:]", text):
        text = text[0]
    return text.rstrip(".").strip().lower()


def _resolve_path(benchmark_root: str | Path, value: str) -> str:
    # Import lazily because benchmark task partitioning imports metric metadata.
    from ..benchmark.media import resolve_path

    return resolve_path(benchmark_root, value)


def _measurement(value: Any, default_unit: Any = "") -> Tuple[float | None, str]:
    unit = str(default_unit or "").strip().lower()
    scalar: Any = value
    if isinstance(value, Mapping):
        scalar = value.get("value", value.get("scalar", value.get("prediction")))
        unit = str(value.get("unit") or unit).strip().lower()
    if isinstance(scalar, bool):
        return None, unit
    if isinstance(scalar, (int, float)) and math.isfinite(float(scalar)):
        return float(scalar), unit
    text = str(scalar or "")
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    if not match:
        return None, unit
    if not unit:
        unit_match = re.search(
            r"(?i)\b(mm|cm|m|ft|in|meters?|metres?|centimeters?|centimetres?|millimeters?|feet|foot|inches?|inch)\b",
            text,
        )
        unit = unit_match.group(1).lower() if unit_match else ""
    return float(match.group(0)), unit


def _to_centimeters(value: float | None, unit: str) -> float | None:
    if value is None:
        return None
    normalized = str(unit or "cm").strip().lower()
    multiplier = UNIT_TO_CENTIMETERS.get(normalized)
    return None if multiplier is None else float(value) * multiplier


def _normalized_points(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, Mapping):
        value = value.get("points", value.get("point", value.get("value")))
    value = _structured_value(value)
    if isinstance(value, (list, tuple)) and len(value) >= 2 and all(
        isinstance(item, (int, float)) for item in value[:2]
    ):
        return [(float(value[0]), float(value[1]))]
    points = []
    if isinstance(value, (list, tuple)):
        for row in value:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                points.append((float(row[0]), float(row[1])))
    return points


def _prediction_mask(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value.astype(np.uint8)
    if isinstance(value, Mapping):
        value = value.get("mask", value.get("path", value.get("value")))
    if isinstance(value, (str, Path)):
        return cv2.imread(str(value), cv2.IMREAD_GRAYSCALE)
    return None


def _bbox_value(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, Mapping):
        value = value.get("bbox", value.get("box", value.get("value")))
    value = _structured_value(value)
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _normalize_bbox_scale(
    box: tuple[float, float, float, float],
    item: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    if max(abs(value) for value in box) <= 1.0:
        return box
    metadata = item.get("metadata") or {}
    evaluation = item.get("evaluation") or {}
    try:
        width = float(metadata.get("width") or evaluation.get("width") or 0)
        height = float(metadata.get("height") or evaluation.get("height") or 0)
    except (TypeError, ValueError):
        return box
    if width <= 0 or height <= 0:
        return box
    return box[0] / width, box[1] / height, box[2] / width, box[3] / height


def _empty_bbox(box: tuple[float, float, float, float]) -> bool:
    return box[2] <= box[0] or box[3] <= box[1]


def _normalized_polyline(
    value: Any,
    item: Mapping[str, Any],
    benchmark_root: str | Path,
) -> list[list[float]]:
    if isinstance(value, Mapping):
        value = value.get(
            "points",
            value.get("path", value.get("polyline", value.get("trajectory", value.get("value")))),
        )
    value = _structured_value(value)
    if not isinstance(value, (list, tuple)):
        return []
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            points.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            return []
    if not points:
        return []
    if max(abs(coordinate) for point in points for coordinate in point) <= 1.0:
        return points
    input_media = (item.get("input") or {}).get("media") or []
    image_path = ""
    for media in input_media:
        if isinstance(media, Mapping) and media.get("path"):
            image_path = _resolve_path(benchmark_root, str(media["path"]))
            break
    image = cv2.imread(image_path) if image_path else None
    if image is None:
        return []
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return []
    return [[x / width, y / height] for x, y in points]


def _discrete_frechet(predicted: list[list[float]], target: list[list[float]]) -> float:
    pred = np.asarray(predicted, dtype=float)
    gt = np.asarray(target, dtype=float)
    distances = np.sqrt(((pred[:, None, :] - gt[None, :, :]) ** 2).sum(axis=2))
    rows, columns = distances.shape
    accumulated = np.full((rows, columns), np.inf)
    accumulated[0, 0] = distances[0, 0]
    for row in range(1, rows):
        accumulated[row, 0] = max(accumulated[row - 1, 0], distances[row, 0])
    for column in range(1, columns):
        accumulated[0, column] = max(
            accumulated[0, column - 1], distances[0, column]
        )
    for row in range(1, rows):
        for column in range(1, columns):
            accumulated[row, column] = max(
                distances[row, column],
                min(
                    accumulated[row - 1, column],
                    accumulated[row, column - 1],
                    accumulated[row - 1, column - 1],
                ),
            )
    return float(accumulated[-1, -1])


def _structured_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{(":
        return value
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return value
