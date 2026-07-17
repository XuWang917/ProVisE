from __future__ import annotations

import re
from typing import Any, Dict

import cv2
import numpy as np

from .base import BaseProtocol, ParseResult, ScoreResult


BLUE_LOWER = np.array([100, 120, 50])
BLUE_UPPER = np.array([130, 255, 255])

CELL_TO_DIR = {
    (0, 0): "back-left",
    (0, 1): "back",
    (0, 2): "back-right",
    (1, 0): "left",
    (1, 2): "right",
    (2, 0): "front-left",
    (2, 1): "front",
    (2, 2): "front-right",
}


class DirectionGridProtocol(BaseProtocol):
    """Parse a blue 3x3 outer-cell direction code."""

    name = "direction_grid"

    def parse(self, generated_path: str, item: Dict[str, Any], benchmark_root: str) -> ParseResult:
        image = cv2.imread(generated_path)
        if image is None:
            return ParseResult("", False, {"error": "generated image unreadable"})
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

        height, width = mask.shape
        rh, rw = height // 3, width // 3
        counts = {}
        for (row, col), direction in CELL_TO_DIR.items():
            y1, y2 = row * rh, (row + 1) * rh
            x1, x2 = col * rw, (col + 1) * rw
            counts[direction] = int(np.sum(mask[y1:y2, x1:x2] > 0))
        best = max(counts, key=counts.get)
        min_pixels = int(self.config.get("min_pixels", 50))
        return ParseResult(best if counts[best] >= min_pixels else "", counts[best] >= min_pixels, {"cell_counts": counts})

    def score(self, parsed: ParseResult, item: Dict[str, Any], benchmark_root: str) -> ScoreResult:
        pred = normalize_direction(parsed.prediction)
        gt = normalize_direction(item.get("answer", ""))
        ok = pred == gt
        return ScoreResult(1.0 if ok else 0.0, ok, {"prediction": pred, "ground_truth": gt})


def normalize_direction(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"^[a-d]\.\s*", "", text)
    text = text.replace("_", "-").replace(" ", "-")
    aliases = {
        "frontleft": "front-left",
        "frontright": "front-right",
        "backleft": "back-left",
        "backright": "back-right",
    }
    return aliases.get(text, text)

