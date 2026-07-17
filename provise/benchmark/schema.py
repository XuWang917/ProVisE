from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .media import select_primary_media_entry


UNIFIED_SCHEMA_VERSION = "genbench.v1"


def load_unified_items(path: str | Path) -> List[Dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    items: List[Dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: each JSONL row must be an object")
            items.append(row)
    return items


def canonicalize_unified_item(item: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize legacy fields before an item enters construction or evaluation."""
    normalized = dict(item)
    evaluation = normalized.get("evaluation")
    if isinstance(evaluation, Mapping):
        normalized_evaluation = dict(evaluation)
        normalized_evaluation.pop("metric_provenance", None)
        normalized["evaluation"] = normalized_evaluation
    if normalized.get("image_path"):
        return normalized
    candidate = select_primary_media_entry(normalized)
    if candidate is not None and candidate.get("path"):
        normalized["image_path"] = candidate["path"]
    return normalized


def validate_unified_items(
    items: Iterable[Mapping[str, Any]],
    benchmark_root: str | Path,
    *,
    strict_choice_labels: bool = False,
) -> Dict[str, Any]:
    rows = list(items)
    root = Path(benchmark_root).expanduser().resolve()
    errors: List[str] = []
    warnings: List[str] = []
    ids = set()
    missing_media: List[str] = []
    for index, item in enumerate(rows):
        if str(item.get("schema_version") or "") != UNIFIED_SCHEMA_VERSION:
            errors.append(
                f"row {index}: schema_version must be {UNIFIED_SCHEMA_VERSION}"
            )
        sample_id = str(item.get("id") or "")
        if not sample_id:
            errors.append(f"row {index}: missing id")
        elif sample_id in ids:
            errors.append(f"row {index}: duplicate id {sample_id}")
        ids.add(sample_id)
        for key in ("benchmark", "task", "question"):
            if item.get(key) in (None, ""):
                errors.append(f"row {index}: missing {key}")
        if item.get("answer") in (None, ""):
            errors.append(f"row {index}: missing answer")

        input_spec = item.get("input") or {}
        if not isinstance(input_spec, Mapping):
            errors.append(f"row {index}: input must be an object")
            input_spec = {}
        media = input_spec.get("media") or []
        if not isinstance(media, list) or not media:
            errors.append(f"row {index}: no input media")
            media = []
        for media_index, entry in enumerate(media):
            if not isinstance(entry, Mapping):
                errors.append(f"row {index}: media[{media_index}] must be an object")
                continue
            path = str(entry.get("path") or "").strip()
            if not path:
                errors.append(f"row {index}: media[{media_index}] is missing path")
                continue
            resolved = resolve_package_path(root, path)
            if not resolved.exists():
                missing_media.append(str(resolved))

        choices = item.get("choices") or []
        if not isinstance(choices, list):
            errors.append(f"row {index}: choices must be a list")
            choices = []
        labels = [
            str(choice.get("label"))
            for choice in choices
            if isinstance(choice, Mapping) and choice.get("label") is not None
        ]
        if len(labels) != len(set(labels)):
            errors.append(f"row {index}: duplicate choice labels")
        answer_not_in_choices = choices and _normalized_label(item.get("answer")) not in {
            _normalized_label(label) for label in labels
        }
        evaluation_value = item.get("evaluation") or {}
        metric = (
            str(evaluation_value.get("metric") or "").strip().lower()
            if isinstance(evaluation_value, Mapping)
            else ""
        )
        if answer_not_in_choices:
            message = f"row {index}: answer is not one of the choice labels"
            if strict_choice_labels or metric in {"accuracy", "exact_match", "exact_count"}:
                errors.append(message)
            else:
                warnings.append(message)

        evaluation = evaluation_value
        if not isinstance(evaluation, Mapping):
            errors.append(f"row {index}: evaluation must be an object")
        elif not str(evaluation.get("metric") or "").strip():
            errors.append(f"row {index}: evaluation.metric is required")
        for target_key in ("mask_path", "target_path"):
            target_path = str(evaluation.get(target_key) or "").strip()
            if target_path and not resolve_package_path(root, target_path).exists():
                missing_media.append(str(resolve_package_path(root, target_path)))

    if missing_media:
        errors.append(f"missing media files: {len(missing_media)}")
    return {
        "valid": bool(rows) and not errors,
        "sample_count": len(rows),
        "unique_id_count": len(ids),
        "missing_media_count": len(missing_media),
        "missing_media_examples": missing_media[:20],
        "errors": errors[:100],
        "warnings": warnings[:100],
    }


def summarize_validation_failure(validation: Mapping[str, Any]) -> str:
    errors = [str(error) for error in validation.get("errors") or []]
    sample_count = int(validation.get("sample_count") or 0)
    if not errors:
        return f"loaded {sample_count} samples but unified validation failed"
    patterns = Counter(re.sub(r"^row \d+:\s*", "", error) for error in errors)
    details = "; ".join(
        f"{message} ({count} reported)" if count > 1 else message
        for message, count in patterns.most_common(3)
    )
    return f"loaded {sample_count} samples but unified validation failed: {details}"


def resolve_package_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _normalized_label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value if value is not None else "").strip().rstrip(".").lower()
