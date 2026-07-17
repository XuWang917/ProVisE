from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ..benchmark.media import (
    normalize_bool_label,
    primary_media_path,
    resolve_input_media_paths,
    resolve_path,
)
from ..evaluation.results import summarize_details


@dataclass
class ParseResult:
    prediction: Any
    parse_success: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    score: float
    is_correct: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseProtocol:
    """Base class for prompt-parser protocols.

    A protocol owns three things:
    1. how benchmark inputs are passed to the image generation model;
    2. how a generated image is parsed back into a structured prediction;
    3. how that prediction is bridged back to the original benchmark metric.
    """

    name = "base"

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    def input_paths(self, item: Dict[str, Any], benchmark_root: str | Path) -> List[str]:
        unified_paths = self._unified_input_paths(item, benchmark_root)
        if unified_paths:
            return unified_paths

        mode = self.config.get("input_mode", "single")
        primary_path = primary_media_path(item)
        if mode == "single":
            if primary_path:
                return [resolve_path(benchmark_root, primary_path)]
            return [resolve_path(benchmark_root, item["image_path"])]
        if mode == "file_names_same_dir":
            base_ref = primary_path or str(item["image_path"])
            base = Path(resolve_path(benchmark_root, base_ref)).parent
            names = item.get("metadata", {}).get("file_names", [])
            return [str((base / name).resolve()) for name in names]
        if mode == "metadata_images":
            return self._metadata_image_paths(item, benchmark_root)
        raise ValueError(f"Unsupported input_mode for protocol {self.name}: {mode}")

    def _unified_input_paths(self, item: Dict[str, Any], benchmark_root: str | Path) -> List[str]:
        mode = self.config.get("input_mode", "")
        roles = set(_as_list(self.config.get("media_roles")))
        labels = {str(x) for x in _as_list(self.config.get("media_labels"))}
        limit = 1 if mode == "single" else None
        return resolve_input_media_paths(item, benchmark_root, roles=roles, labels=labels, limit=limit)

    def _metadata_image_paths(self, item: Dict[str, Any], benchmark_root: str | Path) -> List[str]:
        images = item.get("metadata", {}).get("images", [])
        if not images:
            fallback = primary_media_path(item)
            if fallback:
                return [resolve_path(benchmark_root, fallback)]
            return [resolve_path(benchmark_root, item["image_path"])]

        base_ref = primary_media_path(item) or str(item["image_path"])
        parts = str(base_ref).split("/")
        if len(parts) >= 2:
            base = Path(benchmark_root) / parts[0] / parts[1]
        else:
            base = Path(benchmark_root)

        resolved = []
        for image in images:
            rel = str(image).lstrip("./")
            if rel.startswith("image/"):
                rel = rel[len("image/"):]
            resolved.append(str((base / rel).resolve()))
        return resolved

    def variables(self, item: Dict[str, Any], benchmark_root: str | Path) -> Dict[str, Any]:
        meta = item.get("metadata", {}) or {}
        choices = item.get("choices") or []
        values = {
            "id": item.get("id", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "choices": format_choices(choices),
            "choices_text": format_choices(choices),
            "category": meta.get("category_name", meta.get("category", "object")),
            "caption": meta.get("caption", item.get("question", "")),
            "subj": meta.get("subj", ""),
            "obj": meta.get("obj", ""),
            "human_instruction_text": item.get("question", ""),
            "n_images": len(self.input_paths(item, benchmark_root)),
            "start_point": "",
        }
        values.update({k: v for k, v in meta.items() if isinstance(k, str)})
        return values

    def render_prompt(self, prompt_template: str, item: Dict[str, Any], benchmark_root: str | Path) -> str:
        variables = _SafeFormatDict(self.variables(item, benchmark_root))
        return prompt_template.format_map(variables)

    def parse(self, generated_path: str, item: Dict[str, Any], benchmark_root: str | Path) -> ParseResult:
        raise NotImplementedError

    def score(self, parsed: ParseResult, item: Dict[str, Any], benchmark_root: str | Path) -> ScoreResult:
        prediction = self.normalize_label(parsed.prediction)
        ground_truth = self.normalize_label(item.get("answer", ""))
        ok = prediction == ground_truth
        return ScoreResult(
            score=1.0 if ok else 0.0,
            is_correct=ok,
            extra={"prediction": prediction, "ground_truth": ground_truth},
        )

    def aggregate(self, details: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
        summary = summarize_details(details)
        return {
            "task": task,
            "protocol": self.name,
            **summary,
            "detailed_results": details,
        }

    def normalize_label(self, value: Any) -> str:
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        text = normalize_bool_label(value).strip()
        if re.match(r"^[A-D]\.", text, flags=re.IGNORECASE):
            return text[0].upper()
        if text.upper() in {"A", "B", "C", "D"}:
            return text.upper()
        return text.lower()


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def _as_list(value: Any) -> List[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def format_choices(choices: Iterable[Any]) -> str:
    parts = []
    for idx, choice in enumerate(choices):
        if isinstance(choice, dict):
            label = choice.get("label")
            text = choice.get("text")
            if label not in (None, "") and text not in (None, ""):
                parts.append(f"{label}. {text}")
            elif label not in (None, ""):
                parts.append(str(label))
            elif text not in (None, ""):
                parts.append(str(text))
            else:
                parts.append(str(choice))
        else:
            parts.append(str(choice))
    return ", ".join(parts)
