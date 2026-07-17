from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

from .schema import (
    UNIFIED_SCHEMA_VERSION,
    load_unified_items,
    summarize_validation_failure,
    validate_unified_items,
)


PACKAGE_SCHEMA_VERSION = "provise.benchmark.v1"
PACKAGE_MANIFEST_NAMES = ("benchmark.yaml", "benchmark.yml", "provise.yaml", "provise.yml")


@dataclass
class BenchmarkPackage:
    benchmark_name: str
    data_file: Path
    benchmark_root: Path
    items: List[Dict[str, Any]]
    validation: Dict[str, Any]
    manifest_path: Path | None = None


@dataclass
class BenchmarkPackageProbe:
    status: str
    package: BenchmarkPackage | None = None
    reason: str = ""
    candidates: List[str] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)


def probe_benchmark_package(
    source: str | Path,
    *,
    benchmark_name: str = "",
) -> BenchmarkPackageProbe:
    source_path = Path(os.path.abspath(Path(source).expanduser()))
    source_is_file = source_path.suffix.lower() == ".jsonl"
    if source_is_file:
        if not source_path.is_file():
            raise FileNotFoundError(f"benchmark source does not exist: {source_path}")
        directory_entries = None
    else:
        try:
            directory_entries = {path.name: path for path in source_path.iterdir()}
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise FileNotFoundError(
                f"benchmark source does not exist: {source_path}"
            ) from exc
    manifest_path, manifest, manifest_error = _load_manifest(
        source_path, directory_entries, source_is_file=source_is_file
    )
    if manifest_error:
        return BenchmarkPackageProbe(
            status="invalid",
            reason=manifest_error,
            candidates=[str(manifest_path)] if manifest_path else [],
        )

    candidates = _candidate_data_files(
        source_path,
        manifest,
        directory_entries,
        source_is_file=source_is_file,
    )
    candidate_names = [str(path) for path in candidates]
    explicit_data_file = bool(manifest and manifest.get("data_file"))
    invalid_rows: List[tuple[Path, str, Dict[str, Any]]] = []
    for data_file in candidates:
        if not data_file.is_file():
            if explicit_data_file:
                return BenchmarkPackageProbe(
                    status="invalid",
                    reason=f"benchmark package data_file does not exist: {data_file}",
                    candidates=candidate_names,
                )
            continue
        try:
            items = load_unified_items(data_file)
        except (OSError, ValueError) as exc:
            if explicit_data_file:
                return BenchmarkPackageProbe(
                    status="invalid",
                    reason=f"could not load benchmark package data: {exc}",
                    candidates=candidate_names,
                )
            continue
        if not items:
            if explicit_data_file:
                return BenchmarkPackageProbe(
                    status="invalid",
                    reason=f"benchmark package data is empty: {data_file}",
                    candidates=candidate_names,
                )
            continue
        schema_versions = {str(item.get("schema_version") or "") for item in items}
        looks_unified = UNIFIED_SCHEMA_VERSION in schema_versions
        if not looks_unified:
            if explicit_data_file:
                return BenchmarkPackageProbe(
                    status="invalid",
                    reason=(
                        f"benchmark package data does not use {UNIFIED_SCHEMA_VERSION}: "
                        f"{data_file}"
                    ),
                    candidates=candidate_names,
                )
            continue
        if schema_versions != {UNIFIED_SCHEMA_VERSION}:
            invalid_rows.append(
                (
                    data_file,
                    "normalized data mixes schema versions",
                    {"schema_versions": sorted(schema_versions)},
                )
            )
            continue

        roots = _candidate_benchmark_roots(source_path, data_file, manifest)
        best_validation: Dict[str, Any] = {}
        for root in roots:
            validation = validate_unified_items(items, root)
            if not best_validation or _validation_rank(validation) < _validation_rank(best_validation):
                best_validation = validation
            if validation.get("valid"):
                name = _package_name(benchmark_name, manifest, items, source_path, data_file)
                return BenchmarkPackageProbe(
                    status="ready",
                    package=BenchmarkPackage(
                        benchmark_name=name,
                        data_file=data_file,
                        benchmark_root=root,
                        items=items,
                        validation=validation,
                        manifest_path=manifest_path,
                    ),
                    reason="validated normalized benchmark package",
                    candidates=candidate_names,
                    validation=validation,
                )
        invalid_rows.append(
            (data_file, summarize_validation_failure(best_validation), best_validation)
        )

    if invalid_rows:
        data_file, reason, validation = invalid_rows[0]
        return BenchmarkPackageProbe(
            status="invalid",
            reason=f"invalid normalized benchmark package {data_file}: {reason}",
            candidates=candidate_names,
            validation=validation,
        )
    return BenchmarkPackageProbe(
        status="absent",
        reason="no genbench.v1 JSONL package was found; raw ingestion is required",
        candidates=candidate_names,
    )


def write_package_manifest(
    path: str | Path,
    *,
    benchmark_name: str,
    data_file: str,
    benchmark_root: str = ".",
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "benchmark": benchmark_name,
        "data_file": data_file,
        "benchmark_root": benchmark_root,
    }
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return output


def infer_benchmark_name(source: str | Path) -> str:
    path = Path(source).expanduser()
    raw = path.stem if path.suffix else path.name
    for suffix in (".unified", "_unified", "-unified"):
        if raw.lower().endswith(suffix):
            raw = raw[: -len(suffix)]
            break
    normalized = "".join(character.lower() if character.isalnum() else "_" for character in raw)
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized or "benchmark"


def _load_manifest(
    source: Path,
    directory_entries: Mapping[str, Path] | None = None,
    *,
    source_is_file: bool | None = None,
) -> tuple[Path | None, Dict[str, Any], str]:
    is_file = source.is_file() if source_is_file is None else source_is_file
    if is_file:
        return None, {}, ""
    for name in PACKAGE_MANIFEST_NAMES:
        path = (
            directory_entries.get(name)
            if directory_entries is not None
            else source / name
        )
        if path is None or not path.is_file():
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            return path, {}, f"could not read benchmark package manifest {path}: {exc}"
        if not isinstance(payload, dict):
            continue
        schema = str(payload.get("schema_version") or "")
        if not schema:
            continue
        if schema != PACKAGE_SCHEMA_VERSION:
            if schema.startswith("provise."):
                return path, {}, f"unsupported benchmark package schema {schema!r}: {path}"
            continue
        if not payload.get("data_file"):
            return path, {}, f"benchmark package manifest is missing data_file: {path}"
        return path, dict(payload), ""
    return None, {}, ""


def _candidate_data_files(
    source: Path,
    manifest: Mapping[str, Any],
    directory_entries: Mapping[str, Path] | None = None,
    *,
    source_is_file: bool | None = None,
) -> List[Path]:
    is_file = source.is_file() if source_is_file is None else source_is_file
    if is_file:
        return [source] if source.suffix.lower() == ".jsonl" else []
    paths: List[Path] = []
    if manifest.get("data_file"):
        value = Path(str(manifest["data_file"])).expanduser()
        paths.append(value if value.is_absolute() else source / value)
    else:
        entries = directory_entries or {path.name: path for path in source.iterdir()}
        for name in ("data.jsonl", f"{source.name}.jsonl"):
            if name in entries:
                paths.append(entries[name])
        paths.extend(
            sorted(
                path
                for name, path in entries.items()
                if name.endswith(".unified.jsonl") or name.endswith(".jsonl")
            )
        )
        normalized = entries.get("normalized")
        if normalized is not None and normalized.is_dir():
            normalized_entries = {path.name: path for path in normalized.iterdir()}
            if "data.jsonl" in normalized_entries:
                paths.append(normalized_entries["data.jsonl"])
            paths.extend(
                sorted(
                    path
                    for name, path in normalized_entries.items()
                    if name.endswith(".jsonl")
                )
            )
    unique: List[Path] = []
    seen = set()
    for path in paths:
        resolved = Path(os.path.abspath(path))
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _candidate_benchmark_roots(
    source: Path,
    data_file: Path,
    manifest: Mapping[str, Any],
) -> List[Path]:
    package_dir = source if source.is_dir() else source.parent
    roots: List[Path] = []
    if manifest.get("benchmark_root") is not None:
        value = Path(str(manifest.get("benchmark_root") or ".")).expanduser()
        roots.append(value if value.is_absolute() else package_dir / value)
    roots.extend(
        [
            package_dir,
            package_dir / "assets",
            data_file.parent,
            data_file.parent / "assets",
        ]
    )
    unique: List[Path] = []
    seen = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _package_name(
    explicit_name: str,
    manifest: Mapping[str, Any],
    items: List[Mapping[str, Any]],
    source: Path,
    data_file: Path,
) -> str:
    if explicit_name:
        return explicit_name
    manifest_name = str(manifest.get("benchmark") or manifest.get("name") or "").strip()
    if manifest_name:
        return manifest_name
    item_names = {str(item.get("benchmark") or "").strip() for item in items}
    item_names.discard("")
    if len(item_names) == 1:
        return next(iter(item_names))
    return infer_benchmark_name(source if source.is_dir() else data_file)


def _validation_rank(validation: Mapping[str, Any]) -> tuple[int, int]:
    return (
        int(validation.get("missing_media_count") or 0),
        len(validation.get("errors") or []),
    )
