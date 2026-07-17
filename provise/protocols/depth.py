from __future__ import annotations

import math
import os
import re
from typing import Any, Dict

import cv2
import numpy as np

from ..benchmark.media import resolve_path

from .base import BaseProtocol, ParseResult, ScoreResult


DEPTH_VLM_PARSE_PROMPT = """You are given two images of the same scene.

Image 1 is the original RGB image, where points A and B may be marked.
Image 2 is a grayscale depth map of the same scene, where brighter means closer
and darker means farther.

Use Image 1 to understand the marked points or options, then use Image 2 to
decide which point is closer to the camera.

Answer with ONLY one letter: A or B."""


DEPTH_COORDINATE_PROMPT = (
    "Two points are circled on the image, labeled by A and B beside each red circle."
    "Output the coordinates strictly in the format: [[x_A, y_A], [x_B, y_B]], ranging from 0 to 1."
)


class DenseDepthABProtocol(BaseProtocol):
    """Parse generated grayscale depth map by sampling point A/B coordinates."""

    name = "dense_depth_ab"
    _eval_vlm = None
    _eval_vlm_key = None

    def parse(self, generated_path: str, item: Dict[str, Any], benchmark_root: str) -> ParseResult:
        image = cv2.imread(generated_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return ParseResult("", False, {"error": "generated image unreadable"})

        eval_mode = str(self.config.get("eval_mode", "coords")).strip().lower()
        if eval_mode == "coords":
            return self._parse_with_coords(image, item, benchmark_root)

        original_path = self._original_image_path(item, benchmark_root)
        try:
            vlm = self._get_eval_vlm()
            if hasattr(vlm, "predict_multi"):
                response = vlm.predict_multi([original_path, generated_path], DEPTH_VLM_PARSE_PROMPT)
            else:
                response = vlm.predict(generated_path, DEPTH_VLM_PARSE_PROMPT)
            pred = parse_depth_answer(response)
            return ParseResult(
                pred,
                bool(pred),
                {"depth_eval_mode": "vlm", "vlm_response": response[:500]},
            )
        except Exception as exc:
            if self.config.get("fallback_to_coords", False):
                parsed = self._parse_with_coords(image, item, benchmark_root)
                parsed.extra["vlm_error"] = str(exc)
                parsed.extra["depth_eval_mode"] = "coords_fallback"
                return parsed
            return ParseResult("", False, {"error": str(exc), "depth_eval_mode": "vlm"})

    def _parse_with_coords(
        self,
        image: np.ndarray,
        item: Dict[str, Any],
        benchmark_root: str | None = None,
    ) -> ParseResult:
        try:
            coords, coord_extra = self._get_coordinates(item, benchmark_root)
        except Exception as exc:
            return ParseResult(
                "",
                False,
                {
                    "error": str(exc),
                    "depth_eval_mode": "coords",
                    "coordinate_source": "vlm",
                },
            )
        if not coords or len(coords) < 2:
            extra = {"error": "missing coordinates", "depth_eval_mode": "coords"}
            extra.update(coord_extra)
            return ParseResult("", False, extra)
        da = self._sample(image, float(coords[0][0]), float(coords[0][1]))
        db = self._sample(image, float(coords[1][0]), float(coords[1][1]))
        pred = "(A)" if da >= db else "(B)"
        extra = {
            "depth_a": da,
            "depth_b": db,
            "depth_eval_mode": "coords",
            "coordinates": coords,
        }
        extra.update(coord_extra)
        return ParseResult(pred, True, extra)

    def _get_coordinates(
        self,
        item: Dict[str, Any],
        benchmark_root: str | None,
    ) -> tuple[list[list[float]], Dict[str, Any]]:
        coords = coordinates_from_item(item)
        if coords:
            return coords, {"coordinate_source": "benchmark"}

        if not benchmark_root:
            return [], {"coordinate_source": "missing"}

        original_path = self._original_image_path(item, benchmark_root)
        prompt = str(self.config.get("coordinate_prompt") or DEPTH_COORDINATE_PROMPT)
        vlm = self._get_eval_vlm()
        response = vlm.predict(original_path, prompt)
        coords = parse_depth_coordinates(response)
        extra = {
            "coordinate_source": "vlm",
            "vlm_model": getattr(vlm, "model_name", ""),
            "coordinate_vlm_response": str(response or "")[:500],
        }
        if not coords:
            extra["error"] = "invalid coordinate response"
        return coords, extra

    def score(self, parsed: ParseResult, item: Dict[str, Any], benchmark_root: str) -> ScoreResult:
        pred = normalize_depth_label(parsed.prediction)
        gt = normalize_depth_label(item.get("answer", ""))
        ok = pred == gt
        return ScoreResult(1.0 if ok else 0.0, ok, {"prediction": pred, "ground_truth": gt})

    def _original_image_path(self, item: Dict[str, Any], benchmark_root: str) -> str:
        paths = self.input_paths(item, benchmark_root)
        if paths:
            return paths[0]
        return resolve_path(benchmark_root, item["image_path"])

    def _get_eval_vlm(self):
        from ..models.vlm import create_eval_vlm

        timeout = int(self.config.get("vlm_timeout", 60))
        max_tokens = int(self.config.get("vlm_max_tokens", 64))
        model_name = str(
            self.config.get("vlm_model")
            or os.getenv("PROVISE_PARSER_MODEL")
            or "gpt-5.4"
        ).strip()
        key = (model_name or "", timeout, max_tokens)
        if self.__class__._eval_vlm is None or self.__class__._eval_vlm_key != key:
            self.__class__._eval_vlm = create_eval_vlm(
                timeout=timeout,
                max_tokens=max_tokens,
                model_name=model_name,
            )
            self.__class__._eval_vlm.load_model()
            self.__class__._eval_vlm_key = key
        return self.__class__._eval_vlm

    def _sample(self, image: np.ndarray, nx: float, ny: float) -> float:
        height, width = image.shape
        cx = max(0, min(int(nx * width), width - 1))
        cy = max(0, min(int(ny * height), height - 1))
        half = int(self.config.get("kernel_size", 5)) // 2
        roi = image[max(0, cy - half): cy + half + 1, max(0, cx - half): cx + half + 1]
        return float(np.mean(roi)) if roi.size else 0.0


def coordinates_from_item(item: Dict[str, Any]) -> list[list[float]]:
    for container in (item.get("evaluation", {}), item.get("metadata", {})):
        raw_coords = container.get("coordinates") if isinstance(container, dict) else None
        coords = normalize_depth_coordinates(raw_coords)
        if coords:
            return coords
    return []


def parse_depth_coordinates(response: str) -> list[list[float]]:
    text = str(response or "").strip()
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    pattern = re.compile(
        r"\[\s*\[\s*"
        rf"({number})\s*,\s*({number})"
        r"\s*\]\s*,\s*\[\s*"
        rf"({number})\s*,\s*({number})"
        r"\s*\]\s*\]"
    )
    match = pattern.search(text)
    if not match:
        return []
    values = [float(match.group(i)) for i in range(1, 5)]
    return normalize_depth_coordinates(
        [[values[0], values[1]], [values[2], values[3]]]
    )


def normalize_depth_coordinates(value: Any) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return []
    coords: list[list[float]] = []
    for point in value[:2]:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return []
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            return []
        if not (math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return []
        coords.append([x, y])
    return coords


def parse_depth_answer(response: str) -> str:
    text = str(response or "").strip().upper()
    if text in {"A", "B", "(A)", "(B)"}:
        return normalize_depth_label(text)
    if "(A)" in text or "A IS CLOSER" in text or "POINT A" in text:
        return "(A)"
    if "(B)" in text or "B IS CLOSER" in text or "POINT B" in text:
        return "(B)"
    match = re.search(r"\b([AB])\b", text)
    return f"({match.group(1)})" if match else ""


def normalize_depth_label(value: Any) -> str:
    text = str(value).strip().upper()
    if text in {"A", "(A)"}:
        return "(A)"
    if text in {"B", "(B)"}:
        return "(B)"
    match = re.search(r"\(?\b([AB])\b\)?", text)
    return f"({match.group(1)})" if match else text
