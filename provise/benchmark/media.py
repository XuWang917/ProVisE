from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable


def resolve_path(root: str | Path, path: str | Path) -> str:
    """Resolve a benchmark-relative path while tolerating old prefixes."""
    raw = str(path)
    if os.path.isabs(raw):
        return raw

    raw = raw.lstrip("./")
    for prefix in ("benchmarks_full/", "benchmarks/", "subbenchmarks/"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    return str((Path(root) / raw).resolve())


def normalize_media_entry(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"type": "image", "path": value}
    if isinstance(value, dict):
        if value.get("path"):
            return dict(value)
        nested = value.get("media")
        if isinstance(nested, dict) and nested.get("path"):
            merged = dict(nested)
            for key in ("role", "label"):
                if key in value and key not in merged:
                    merged[key] = value[key]
            return merged
    return {}


def input_media_entries(item: dict[str, Any]) -> list[dict[str, Any]]:
    input_spec = item.get("input") or {}
    if not isinstance(input_spec, dict):
        return []

    media = input_spec.get("media")
    if media is None and input_spec.get("images") is not None:
        media = input_spec.get("images")
    if media is None:
        return []
    if not isinstance(media, list):
        media = [media]

    entries = [normalize_media_entry(value) for value in media]
    return [entry for entry in entries if entry.get("path")]


def select_input_media_entries(
    item: dict[str, Any],
    roles: Iterable[Any] | None = None,
    labels: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    entries = input_media_entries(item)
    if not entries:
        return []

    role_set = _value_set(roles)
    label_set = _value_set(labels)
    filtered = entries
    if role_set:
        filtered = [entry for entry in filtered if str(entry.get("role", "")) in role_set]
    if label_set:
        filtered = [entry for entry in filtered if str(entry.get("label", "")) in label_set]
    return filtered or entries


def select_primary_media_entry(item: dict[str, Any]) -> dict[str, Any] | None:
    entries = select_input_media_entries(item, roles=("primary",))
    return entries[0] if entries else None


def primary_media_path(item: dict[str, Any], legacy_field: str = "image_path") -> str | None:
    entry = select_primary_media_entry(item)
    if entry is not None and entry.get("path"):
        return str(entry["path"])

    if legacy_field:
        raw = item.get(legacy_field)
        if raw not in (None, ""):
            return str(raw)
    return None


def resolve_input_media_paths(
    item: dict[str, Any],
    benchmark_root: str | Path,
    roles: Iterable[Any] | None = None,
    labels: Iterable[Any] | None = None,
    limit: int | None = None,
) -> list[str]:
    entries = select_input_media_entries(item, roles=roles, labels=labels)
    if limit is not None:
        entries = entries[: max(0, limit)]
    return [resolve_path(benchmark_root, entry["path"]) for entry in entries if entry.get("path")]


def resolve_primary_media_path(
    item: dict[str, Any],
    benchmark_root: str | Path,
    legacy_field: str = "image_path",
) -> str | None:
    raw = primary_media_path(item, legacy_field=legacy_field)
    if raw in (None, ""):
        return None
    return resolve_path(benchmark_root, raw)


def existing(paths: Iterable[str]) -> list[str]:
    return [p for p in paths if os.path.exists(p)]


def normalize_bool_label(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip()
    if text.lower() in {"true", "yes", "1"}:
        return "true"
    if text.lower() in {"false", "no", "0"}:
        return "false"
    return text


def _value_set(values: Iterable[Any] | None) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, (str, bytes)):
        return {str(values)} if values else set()
    return {str(value) for value in values if value not in (None, "")}
