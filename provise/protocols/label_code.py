from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from .base import BaseProtocol, ParseResult, ScoreResult


MAGENTA_LOWER = np.array([140, 80, 80])
MAGENTA_UPPER = np.array([175, 255, 255])


class LabelCodeProtocol(BaseProtocol):
    """Generic protocol for discrete VLM-style labels.

    The model keeps the original input semantics, but writes the answer as a
    fixed visual code in the generated image. This avoids rendering questions or
    options into the input image and works with single-image or multi-image input.

    Default layouts:
    - corners4: A=top-left, B=top-right, C=bottom-left, D=bottom-right.
    - hstrip: labels are slots on the bottom strip from left to right.
    """

    name = "label_code"

    def labels(self, item: Dict[str, Any] | None = None) -> List[str]:
        labels = self.config.get("labels", ["A", "B", "C", "D"])
        if isinstance(labels, str):
            key = labels.strip()
            if key in {"from_choices", "choices", "auto_choices"}:
                return self._choice_labels(item) or ["A", "B", "C", "D"]
            if key == "A-D":
                return ["A", "B", "C", "D"]
            span = re.match(r"^A-([A-Z])$", key, flags=re.IGNORECASE)
            if span:
                last = ord(span.group(1).upper()) - ord("A")
                return [chr(ord("A") + idx) for idx in range(max(0, last) + 1)]
            if key == "true_false":
                return ["true", "false"]
            return [key]
        return [str(x) for x in labels]

    def variables(self, item: Dict[str, Any], benchmark_root: str) -> Dict[str, Any]:
        values = super().variables(item, benchmark_root)
        labels = self.labels(item)
        values.update(
            {
                "label_count": len(labels),
                "label_list": ", ".join(labels),
                "label_slots": self._format_label_slots(item, labels),
            }
        )
        return values

    def parse(self, generated_path: str, item: Dict[str, Any], benchmark_root: str) -> ParseResult:
        image = cv2.imread(generated_path)
        if image is None:
            return ParseResult("", False, {"error": "generated image unreadable"})

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, MAGENTA_LOWER, MAGENTA_UPPER)
        labels = self.labels(item)
        zones = self._zones(mask.shape, labels)

        counts = {}
        for label, (x1, y1, x2, y2) in zones.items():
            counts[label] = int(np.sum(mask[y1:y2, x1:x2] > 0))

        best_label = max(counts, key=counts.get) if counts else ""
        best_count = counts.get(best_label, 0)
        min_pixels = int(self.config.get("min_pixels", 30))
        return ParseResult(
            best_label if best_count >= min_pixels else "",
            best_count >= min_pixels,
            {"zone_counts": counts, "best_count": best_count, "labels": labels},
        )

    def score(self, parsed: ParseResult, item: Dict[str, Any], benchmark_root: str) -> ScoreResult:
        prediction = self._normalize_answer(parsed.prediction, item)
        ground_truth = self._normalize_answer(item.get("answer", ""), item)
        ok = prediction == ground_truth
        return ScoreResult(
            score=1.0 if ok else 0.0,
            is_correct=ok,
            extra={"prediction": prediction, "ground_truth": ground_truth},
        )

    def normalize_label(self, value: Any) -> str:
        return self._canonical_label(value)

    def _choice_labels(self, item: Dict[str, Any] | None) -> List[str]:
        return [entry["label"] for entry in self._choice_entries(item)]

    def _choice_entries(self, item: Dict[str, Any] | None) -> List[Dict[str, str]]:
        if not item:
            return []
        choices = item.get("choices") or []
        entries = []
        seen = set()
        for idx, choice in enumerate(choices):
            raw_label = ""
            text = ""
            value = ""
            if isinstance(choice, dict):
                raw_label = str(choice.get("label") or "")
                text = str(choice.get("text") or "")
                value = str(choice.get("value") or choice.get("answer") or "")
            else:
                text = str(choice)

            label = self._choice_label(raw_label, idx)
            key = self._answer_key(label)
            if key in seen:
                label = self._default_choice_label(idx)
                key = self._answer_key(label)
            seen.add(key)
            entries.append({"label": label, "raw_label": raw_label, "text": text, "value": value})
        return entries

    def _choice_label(self, raw_label: str, idx: int) -> str:
        if not raw_label:
            return self._default_choice_label(idx)
        code = self._extract_leading_code(raw_label)
        if code:
            return code
        return raw_label.strip()

    def _default_choice_label(self, idx: int) -> str:
        if 0 <= idx < 26:
            return chr(ord("A") + idx)
        return str(idx + 1)

    def _format_label_slots(self, item: Dict[str, Any], labels: List[str]) -> str:
        entries = self._choice_entries(item)
        parts = []
        for idx, label in enumerate(labels):
            text = entries[idx].get("text", "") if idx < len(entries) else ""
            if text and self._answer_key(text) != self._answer_key(label):
                parts.append(f"slot {idx + 1}={label} ({text})")
            else:
                parts.append(f"slot {idx + 1}={label}")
        return "; ".join(parts)

    def _normalize_answer(self, value: Any, item: Dict[str, Any]) -> str:
        raw = self._value_text(value)
        direct_code = self._extract_leading_code(raw)
        entries = self._choice_entries(item)

        if direct_code:
            direct = self._canonical_label(direct_code)
            for entry in entries:
                if direct == self._canonical_label(entry["label"]):
                    return direct
            return direct

        raw_key = self._answer_key(raw)
        for entry in entries:
            canonical = self._canonical_label(entry["label"])
            aliases = [
                entry.get("label", ""),
                entry.get("raw_label", ""),
                entry.get("text", ""),
                entry.get("value", ""),
                f"{entry.get('label', '')}. {entry.get('text', '')}".strip(),
            ]
            if any(raw_key and raw_key == self._answer_key(alias) for alias in aliases):
                return canonical

        return self._canonical_label(raw)

    def _canonical_label(self, value: Any) -> str:
        text = self._value_text(value)
        code = self._extract_leading_code(text)
        if code:
            return code
        return self._answer_key(text)

    def _extract_leading_code(self, value: Any) -> str:
        text = self._value_text(value)
        bool_text = self._bool_code(value)
        if bool_text:
            return bool_text

        match = re.match(r"^\(?\s*([A-Za-z])\s*\)?(?:[.)\]:\-\s]|$)", text)
        if match:
            return match.group(1).upper()

        match = re.match(r"^\(?\s*(\d+)\s*\)?(?:[.)\]:\-\s]|$)", text)
        if match:
            return match.group(1)

        return ""

    def _answer_key(self, value: Any) -> str:
        text = (self._bool_code(value) or self._value_text(value)).strip().lower()
        text = re.sub(r"^[\(\[]\s*|\s*[\)\]]$", "", text)
        return re.sub(r"[^a-z0-9]+", " ", text).strip()

    def _bool_code(self, value: Any) -> str:
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        if isinstance(value, bool):
            return "true" if value else "false"
        text = str(value).strip().lower()
        if text in {"true", "yes"}:
            return "true"
        if text in {"false", "no"}:
            return "false"
        return ""

    def _value_text(self, value: Any) -> str:
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        return str(value).strip()

    def _zones(self, shape: Tuple[int, int], labels: List[str]) -> Dict[str, Tuple[int, int, int, int]]:
        height, width = shape
        layout = self.config.get("layout", "corners4")
        if layout == "corners4":
            margin_x = int(width * 0.04)
            margin_y = int(height * 0.04)
            zone_w = int(width * 0.24)
            zone_h = int(height * 0.24)
            corners = [
                (margin_x, margin_y, margin_x + zone_w, margin_y + zone_h),
                (width - margin_x - zone_w, margin_y, width - margin_x, margin_y + zone_h),
                (margin_x, height - margin_y - zone_h, margin_x + zone_w, height - margin_y),
                (width - margin_x - zone_w, height - margin_y - zone_h, width - margin_x, height - margin_y),
            ]
            return {label: corners[idx] for idx, label in enumerate(labels[:4])}

        if layout == "top2":
            zones = [
                (0, 0, width // 2, max(1, height // 4)),
                (width // 2, 0, width, max(1, height // 4)),
            ]
            return {label: zones[idx] for idx, label in enumerate(labels[:2])}

        if layout == "hstrip":
            strip_h = max(1, int(height * float(self.config.get("strip_ratio", 0.18))))
            y1, y2 = height - strip_h, height
            cell_w = max(1, width // max(1, len(labels)))
            return {
                label: (idx * cell_w, y1, width if idx == len(labels) - 1 else (idx + 1) * cell_w, y2)
                for idx, label in enumerate(labels)
            }

        raise ValueError(f"Unsupported label_code layout: {layout}")
