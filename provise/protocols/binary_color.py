from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np

from .base import BaseProtocol, ParseResult


GREEN_LOWER = np.array([40, 60, 60])
GREEN_UPPER = np.array([85, 255, 255])
BLUE_LOWER = np.array([100, 60, 60])
BLUE_UPPER = np.array([130, 255, 255])


class BinaryColorPresenceProtocol(BaseProtocol):
    """Parse True when green/blue highlight masks are present, else False."""

    name = "binary_color_presence"

    def parse(self, generated_path: str, item: Dict[str, Any], benchmark_root: str) -> ParseResult:
        image = cv2.imread(generated_path)
        if image is None:
            return ParseResult(False, False, {"error": "generated image unreadable"})
        total = image.shape[0] * image.shape[1]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        green_ratio = float(np.sum(cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER) > 0)) / total
        blue_ratio = float(np.sum(cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER) > 0)) / total
        threshold = float(self.config.get("ratio_threshold", 0.005))
        pred = green_ratio > threshold or blue_ratio > threshold
        return ParseResult(pred, True, {"green_ratio": green_ratio, "blue_ratio": blue_ratio})

