from __future__ import annotations

import re
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from ..parser_ops import DEFAULT_REGISTRY, ParserContext, cyan_point_marker_pipeline
from ..benchmark.media import resolve_path
from .base import BaseProtocol, ParseResult, ScoreResult


class AgenticPointMarkerProtocol(BaseProtocol):
    """Protocol generated for 2D point or box localization tasks."""

    name = "agentic_point_marker"

    def parse(self, generated_path: str, item: Dict[str, Any], benchmark_root: str) -> ParseResult:
        pipeline = self.config.get("parser_pipeline") or cyan_point_marker_pipeline()
        try:
            source_paths = tuple(self.input_paths(item, benchmark_root))
        except (KeyError, TypeError, ValueError):
            source_paths = ()
        result = DEFAULT_REGISTRY.execute(
            pipeline,
            ParserContext(
                generated_path=generated_path,
                item=item,
                benchmark_root=str(benchmark_root),
                source_paths=source_paths,
                protocol_config=self.config,
            ),
        )
        steps = result.diagnostics.get("steps") or {}
        if not result.success:
            mask_diagnostics = steps.get("mask") or {}
            component_diagnostics = steps.get("components") or {}
            failed_diagnostics = next(
                (row for row in reversed(list(steps.values())) if row.get("status") == "failed"),
                {},
            )
            extra = {
                "error": result.error,
                "error_type": result.error_type,
                "pixel_count": int(mask_diagnostics.get("pixel_count") or 0),
                "component_count": int(component_diagnostics.get("component_count") or 0),
                "parser": "parser_ops",
                "parser_ops": result.diagnostics,
            }
            if failed_diagnostics.get("candidate_scores") is not None:
                extra["candidate_scores"] = failed_diagnostics["candidate_scores"]
            return ParseResult(None, False, extra)

        point = result.prediction
        marker_diagnostics = steps.get("marker") or {}
        return ParseResult(
            point,
            True,
            {
                "prediction": point,
                "pixel_count": int(marker_diagnostics.get("area") or 0),
                "component_count": int(marker_diagnostics.get("component_count") or 0),
                "fill_ratio": float(marker_diagnostics.get("fill_ratio") or 0.0),
                "compactness": float(marker_diagnostics.get("compactness") or 0.0),
                "parser": "parser_ops",
                "parser_ops": result.diagnostics,
            },
        )

    def score(self, parsed: ParseResult, item: Dict[str, Any], benchmark_root: str) -> ScoreResult:
        point = parsed.prediction
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return ScoreResult(0.0, False, {"prediction": point, "ground_truth": item.get("answer")})

        mask_path = str((item.get("evaluation") or {}).get("mask_path") or "").strip()
        if mask_path:
            resolved_mask = resolve_path(benchmark_root, mask_path)
            mask = cv2.imread(resolved_mask, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return ScoreResult(
                    0.0,
                    False,
                    {"prediction": point, "ground_truth": resolved_mask, "error": "mask_unreadable"},
                )
            height, width = mask.shape[:2]
            x = int(round(float(point[0]) * max(0, width - 1)))
            y = int(round(float(point[1]) * max(0, height - 1)))
            x = min(max(x, 0), max(0, width - 1))
            y = min(max(y, 0), max(0, height - 1))
            ok = bool(mask[y, x] > 0)
            return ScoreResult(
                1.0 if ok else 0.0,
                ok,
                {"prediction": point, "ground_truth": resolved_mask, "mask_pixel": int(mask[y, x])},
            )

        bbox = _bbox_from_item(item)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            ok = x1 <= float(point[0]) <= x2 and y1 <= float(point[1]) <= y2
            return ScoreResult(1.0 if ok else 0.0, ok, {"prediction": point, "ground_truth": bbox})

        gt_point = _point_from_answer(item.get("answer"))
        if gt_point is None:
            return ScoreResult(0.0, False, {"prediction": point, "ground_truth": item.get("answer")})

        dist = float(np.linalg.norm(np.asarray(point[:2], dtype=float) - np.asarray(gt_point[:2], dtype=float)))
        threshold = float(self.config.get("distance_threshold", 0.05))
        return ScoreResult(
            max(0.0, 1.0 - dist / max(threshold, 1e-6)),
            dist <= threshold,
            {"prediction": point, "ground_truth": gt_point, "distance": dist, "threshold": threshold},
        )


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict) and "value" in value:
        return _coerce_number(value["value"])
    match = re.search(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", str(value or ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _bbox_from_item(item: Dict[str, Any]) -> Tuple[float, float, float, float] | None:
    bbox = (
        item.get("evaluation", {}).get("bbox")
        or item.get("target", {}).get("bbox")
        or item.get("metadata", {}).get("bbox")
    )
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    vals = [_coerce_number(x) for x in bbox[:4]]
    if any(x is None for x in vals):
        return None
    x1, y1, x2, y2 = [float(x) for x in vals if x is not None]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        width = _coerce_number(item.get("metadata", {}).get("image_width")) or _coerce_number(
            item.get("evaluation", {}).get("image_width")
        )
        height = _coerce_number(item.get("metadata", {}).get("image_height")) or _coerce_number(
            item.get("evaluation", {}).get("image_height")
        )
        if width and height:
            x1, x2 = x1 / width, x2 / width
            y1, y2 = y1 / height, y2 / height
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _point_from_answer(answer: Any) -> Tuple[float, float] | None:
    if isinstance(answer, dict) and "point" in answer:
        return _point_from_answer(answer["point"])
    if isinstance(answer, (list, tuple)) and len(answer) >= 2:
        x = _coerce_number(answer[0])
        y = _coerce_number(answer[1])
        if x is not None and y is not None:
            return float(x), float(y)
    numbers = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", str(answer or ""))
    if len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])
    return None
