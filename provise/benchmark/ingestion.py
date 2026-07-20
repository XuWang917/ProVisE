from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping

import numpy as np
import yaml
from PIL import Image

from provise.benchmark.package import write_package_manifest
from provise.benchmark.tasks import partition_heterogeneous_tasks
from provise.benchmark.schema import summarize_validation_failure, validate_unified_items
from provise.reporting import NullProgressReporter, ProgressReporter


SUPPORTED_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".parquet",
    ".tsv",
    ".tfrecord",
    ".tfrecords",
}
ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz"}
ALLOWED_TRANSFORMS = {
    "text",
    "first_text",
    "slug",
    "first_slug",
    "raw",
    "conversation_user_text",
    "conversation_assistant_text",
}
ALLOWED_ANSWER_MODES = {"field", "evaluation_target"}
ALLOWED_CHOICE_MODES = {"none", "boolean", "field", "fields", "inline_mcq"}
ALLOWED_MEDIA_MODES = {
    "embedded_images",
    "hf_image",
    "path",
    "path_list",
    "path_template",
    "conversation_images",
}
ALLOWED_TARGET_MEDIA_MODES = {
    "embedded_images",
    "hf_image",
    "path",
    "path_template",
}
MEDIA_VALUE_TRANSFORM_ORDER = ("text", "first_underscore", "basename", "stem")
ALLOWED_MEDIA_VALUE_TRANSFORMS = set(MEDIA_VALUE_TRANSFORM_ORDER)
ALLOWED_METRICS = {
    "accuracy",
    "exact_match",
    "mask_precision",
    "numeric_error",
    "point_in_mask",
    "point_distance",
    "bbox_iou",
    "angle_error",
    "exact_count",
    "mask_iou",
    "numeric_absolute_error",
    "numeric_relative_error",
    "qspatial_ratio",
    "mra",
    "dfd",
    "state_similarity",
    "unverified",
}
ALLOWED_ANSWER_TYPES = {
    "auto",
    "bbox",
    "boolean",
    "choice",
    "mask",
    "number",
    "points",
    "text",
}
MAX_EXTRACTION_WARNING_EXAMPLES = 50
INVENTORY_SKIP_DIRS = {
    ".cache",
    ".git",
    ".github",
    "__pycache__",
    "assets",
    "build",
    "checkpoints",
    "dist",
    "images",
    "logs",
    "media",
    "models",
    "node_modules",
    "outputs",
    "site-packages",
}


INGESTION_SYSTEM_PROMPT = """You are ProVisE's benchmark-ingestion agent.

Your job is to map public benchmark records into the fixed GenBench UnifiedSample
schema. You do not write Python and you do not design a visual protocol. Return
only a declarative JSON mapping that the deterministic executor can validate.

Rules:
- Preserve the official question, answer semantics, media ordering, task names,
  and metric. Never invent missing fields or metrics.
- Prefer official test/evaluation files over train or SFT files.
- Use one source mapping per file when files have different metrics or tasks.
- For hierarchical task labels, prefer a stable official coarse task field; a
  first_slug transform reads the first hierarchy element.
- For embedded image bytes use embedded_images. For Hugging Face image structs
  with bytes/path use hf_image.
- Use path_template when the annotation stores image identity fields rather
  than a complete path. Set field to the identity field, value_transform when
  needed, and template using {value} plus other top-level record fields, e.g.
  {task_type}/{value}. An extension-less template resolves common image suffixes.
- Put only media that the official inference path actually passes to the tested
  model in the media list. Do not include depth maps, masks, annotations, or
  auxiliary images merely because those fields exist; evaluation targets belong
  in evaluation, and optional side data should be omitted unless documentation
  explicitly says it is a model input.
- Use inline_mcq only when choices are explicitly present in question text.
- Use choices.mode=field only when one record field contains the complete choice
  list or mapping. When choices are stored in separate columns such as A/B/C/D
  or option_1/option_2, use choices.mode=fields and list those columns in display
  order. Use choices.labels to declare the official labels in index order. This
  is required when option content is carried only by the input media and the
  annotation stores a numeric answer index; also declare answer.choice_index_base.
  Do not infer a label convention without evidence from the benchmark contract.
- Use point_in_mask only when the official evaluation checks predicted points
  against a target mask.
- For chat-style multimodal records, use conversation_user_text and
  conversation_assistant_text on the top-level conversation/messages field,
  and conversation_images on that same field. The deterministic executor reads
  typed content parts; do not invent flattened helper fields.
- When a benchmark has no separate answer field because its mask is the entire
  metric target, use answer.mode=evaluation_target. Do not serialize image bytes
  into answer or expose the mask as model input.
- Map a ground-truth mask with evaluation.mask. It uses the same scalar image
  mapping as media: hf_image for embedded/Hugging Face image values, path for a
  stored path, or path_template when the record stores only a mask identity.
  For example, use {"field":"mask","mode":"path_template",
  "template":"masks/{value}","value_transform":"text"} for a mask filename
  stored beside a masks/ directory. Evaluation masks are never model inputs.
- When an official grounding benchmark uses null to mean "no target", preserve
  that semantics with answer.null_value (commonly [0, 0, 0, 0]). Never invent a
  null replacement when the benchmark does not define one.
- Select the metric only when the supplied metric_evidence inventory supports
  that scoring rule. If the official metric is not recoverable, use
  metric=unverified. This still permits protocol smoke validation but blocks
  formal scoring. Metric evidence is validated by the framework and is not part
  of the normalized benchmark output.
- Distinguish response normalization from correctness scoring. Some choice
  benchmarks use an LLM only to recover A/B/C/D from a free-form model response,
  then compare that label with the official answer and report hit rate or
  accuracy. When the benchmark records expose the official choice label, map
  this contract to metric=accuracy; the normalization LLM is not itself the
  metric.
- Only use source paths and field names shown in the inventory.
- Keep official task labels unchanged. The deterministic executor partitions a
  label later only when it contains incompatible answer or metric contracts.

Return JSON only:
{
  "benchmark": "name",
  "decision": "ingest | unsupported",
  "reason": "concise rationale",
  "sources": [
    {
      "source": "exact relative source path from inventory",
      "split": "test",
      "id": {"mode": "row_index | field", "field": "optional", "prefix": "sample"},
      "task": {"field": "optional", "constant": "optional", "transform": "text | first_text | slug | first_slug", "default": "default"},
      "question": {"field": "field name", "transform": "text | conversation_user_text"},
      "answer": {"mode": "field | evaluation_target", "field": "required for field mode", "transform": "text | raw | conversation_assistant_text", "choice_index_base": "optional 0 or 1", "null_value": "optional official no-target value"},
      "answer_type": "auto | bbox | choice | boolean | number | points | mask | text",
      "choices": {
        "mode": "none | boolean | inline_mcq | field | fields",
        "field": "one field containing the complete choice collection",
        "fields": ["separate option columns in display order"],
        "labels": ["optional official labels in index order"]
      },
      "media": [
        {"field": "field name", "mode": "embedded_images | hf_image | path | path_list | path_template | conversation_images", "template": "required for path_template", "value_transform": "text | first_underscore | basename | stem", "role": "primary | view | frame", "order_field": "optional"}
      ],
      "evaluation": {
        "metric": "registered metric id or unverified",
        "mask": {"field": "optional target-mask field", "mode": "embedded_images | hf_image | path | path_template", "template": "required for path_template", "value_transform": "text | first_underscore | basename | stem"},
        "metric_config": {}
      },
      "metadata_fields": ["optional source fields to preserve"]
    }
  ]
}
"""


@dataclass
class IngestionResult:
    benchmark_name: str
    decision: str
    items: List[Dict[str, Any]]
    mapping: Dict[str, Any]
    manifest: Dict[str, Any]
    prompt: str
    raw_response: str
    attempts: List[Dict[str, Any]] = field(default_factory=list)


class AgenticBenchmarkIngestor:
    def __init__(
        self,
        *,
        source_root: str | Path,
        benchmark_name: str,
        output_root: str | Path,
        max_examples_per_source: int = 3,
        max_sources: int = 30,
        readme_char_limit: int = 12000,
        max_revisions: int = 1,
        reporter: ProgressReporter | None = None,
    ):
        self.source_root = Path(source_root).expanduser().resolve()
        self.benchmark_name = benchmark_name
        self.output_root = Path(output_root).expanduser().resolve()
        self.max_examples_per_source = max(1, int(max_examples_per_source))
        self.max_sources = max(1, int(max_sources))
        self.readme_char_limit = max(1000, int(readme_char_limit))
        self.max_revisions = max(0, min(1, int(max_revisions)))
        self.reporter = reporter or NullProgressReporter()
        self._media_directory_cache: Dict[str, bool] = {}
        self._media_child_directory_cache: Dict[str, set[str]] = {}

    def build(self, *, vlm: Any | None = None, raw_response: str = "") -> IngestionResult:
        selected_sources: List[str] = []
        if raw_response:
            try:
                saved_mapping = parse_json_response(raw_response)
                selected_sources = [
                    str(row.get("source") or "").strip()
                    for row in saved_mapping.get("sources") or []
                    if isinstance(row, dict) and str(row.get("source") or "").strip()
                ]
            except ValueError:
                selected_sources = []
        inventory = self.inspect(
            selected_sources=selected_sources or None,
            include_agent_context=True,
        )
        self.reporter.emit(
            f"Benchmark inventory ready: {sum(bool(row.get('supported')) for row in inventory.get('sources') or [])} "
            f"supported source(s), {len(inventory.get('metric_evidence') or [])} metric evidence file(s)",
            event="ingestion_inventory_ready",
            status="completed",
        )
        prompt = INGESTION_SYSTEM_PROMPT + "\nBenchmark inventory:\n" + json.dumps(
            inventory, ensure_ascii=False, indent=2
        )
        response = raw_response
        call_error = ""
        if not response:
            if vlm is None:
                raise ValueError("Either vlm or raw_response is required")
            try:
                response = self._predict(vlm, prompt)
            except Exception as exc:
                call_error = f"{type(exc).__name__}: {exc}"
                response = json.dumps(
                    {
                        "benchmark": self.benchmark_name,
                        "decision": "unsupported",
                        "reason": f"ingestion agent call failed: {call_error}",
                        "sources": [],
                    }
                )

        result = self._build_once(inventory=inventory, prompt=prompt, response=str(response or ""))
        attempts = [
            {
                "attempt": 0,
                "phase": "initial",
                "decision": result.decision,
                "reason": result.manifest.get("reason", ""),
                "diagnostics": ingestion_repair_diagnostics(result),
                "response": str(response or ""),
            }
        ]
        can_repair = (
            not raw_response
            and vlm is not None
            and not call_error
            and self.max_revisions > 0
            and result.decision != "ingest"
            and result.manifest.get("blocker_type") != "missing_media"
            and any(bool(row.get("supported")) for row in inventory.get("sources") or [])
        )
        if can_repair:
            diagnostics = ingestion_repair_diagnostics(result)
            repair_prompt = build_ingestion_repair_prompt(prompt, str(response or ""), diagnostics)
            self.reporter.emit(
                "Initial ingestion mapping failed deterministic validation; requesting one repair",
                event="ingestion_repair_started",
                reason=diagnostics.get("reason", ""),
            )
            try:
                repaired_response = self._predict(vlm, repair_prompt)
                repaired = self._build_once(
                    inventory=inventory,
                    prompt=prompt,
                    response=str(repaired_response or ""),
                )
                attempts.append(
                    {
                        "attempt": 1,
                        "phase": "repair",
                        "decision": repaired.decision,
                        "reason": repaired.manifest.get("reason", ""),
                        "diagnostics": ingestion_repair_diagnostics(repaired),
                        "prompt": repair_prompt,
                        "response": str(repaired_response or ""),
                    }
                )
                result = repaired
                self.reporter.emit(
                    f"Ingestion repair completed with decision={repaired.decision}",
                    event="ingestion_repair_completed",
                    status="completed" if repaired.decision == "ingest" else "failed",
                    decision=repaired.decision,
                )
            except Exception as exc:
                attempts.append(
                    {
                        "attempt": 1,
                        "phase": "repair",
                        "decision": "external_failure",
                        "reason": f"ingestion repair agent failed: {type(exc).__name__}: {exc}",
                        "diagnostics": diagnostics,
                        "prompt": repair_prompt,
                        "response": "",
                    }
                )
                result.manifest.setdefault("warnings", []).append(attempts[-1]["reason"])
                self.reporter.emit(
                    attempts[-1]["reason"],
                    event="ingestion_repair_failed",
                    status="failed",
                )

        result.attempts = attempts
        result.manifest["revision_count"] = max(0, len(attempts) - 1)
        result.manifest["attempts"] = [
            {
                key: value
                for key, value in attempt.items()
                if key not in {"prompt", "response"}
            }
            for attempt in attempts
        ]
        return result

    @staticmethod
    def _predict(vlm: Any, prompt: str) -> str:
        response = (
            vlm.predict_multi([], prompt)
            if hasattr(vlm, "predict_multi")
            else vlm.predict("", prompt)
        )
        return str(response or "")

    def _build_once(
        self,
        *,
        inventory: Dict[str, Any],
        prompt: str,
        response: str,
    ) -> IngestionResult:
        warnings = list(inventory.get("warnings") or [])

        try:
            payload = parse_json_response(str(response or ""))
        except ValueError as exc:
            payload = {
                "benchmark": self.benchmark_name,
                "decision": "unsupported",
                "reason": f"invalid ingestion-agent response: {exc}",
                "sources": [],
            }
            warnings.append(payload["reason"])

        mapping, mapping_errors = self._validate_mapping(payload, inventory)
        warnings.extend(mapping_errors)
        if mapping_errors or mapping.get("decision") != "ingest":
            decision = "unsupported"
            reason = mapping.get("reason") or "; ".join(mapping_errors) or "agent declined ingestion"
            manifest = self._manifest(
                inventory=inventory,
                mapping=mapping,
                decision=decision,
                reason=reason,
                warnings=warnings,
                items=[],
                validation={},
            )
            return IngestionResult(
                self.benchmark_name,
                decision,
                [],
                mapping,
                manifest,
                prompt,
                str(response or ""),
            )

        self.output_root.mkdir(parents=True, exist_ok=True)
        mapping_repairs = self._repair_path_template_mappings(mapping)
        for repair in mapping_repairs:
            warnings.append(
                "deterministically repaired media path transform for "
                f"{repair['source']} media[{repair['media_index']}]: "
                f"{repair['from_transform']} -> {repair['to_transform']} "
                f"({repair['probe_count']}/{repair['probe_count']} probe paths resolved)"
            )
            self.reporter.emit(
                f"Repaired media path transform: {repair['from_transform']} -> "
                f"{repair['to_transform']}",
                event="ingestion_mapping_repaired",
                status="completed",
                **repair,
            )
        items = []
        source_counts = Counter()
        source_extraction = {}
        extraction_warnings = []
        multiple_sources = len(mapping["sources"]) > 1
        for source_index, source_mapping in enumerate(mapping["sources"], 1):
            self.reporter.emit(
                f"Converting source {source_index}/{len(mapping['sources'])}: {source_mapping['source']}",
                event="ingestion_source_started",
                source=source_mapping["source"],
                source_index=source_index,
                source_count=len(mapping["sources"]),
            )
            id_namespace = source_namespace(source_mapping["source"]) if multiple_sources else ""
            converted, source_warnings, extraction = self._convert_source(
                source_mapping,
                id_namespace=id_namespace,
            )
            items.extend(converted)
            source_counts[source_mapping["source"]] += len(converted)
            source_extraction[source_mapping["source"]] = extraction
            extraction_warnings.extend(source_warnings)
            self.reporter.emit(
                f"Converted {len(converted)} sample(s) from {source_mapping['source']}",
                event="ingestion_source_completed",
                status="completed",
                source=source_mapping["source"],
                sample_count=len(converted),
            )
        warnings.extend(extraction_warnings)
        id_repair = namespace_duplicate_item_ids(items)
        if id_repair:
            warnings.append(
                "Deterministically namespaced duplicate sample IDs by task; "
                f"repaired {id_repair['affected_sample_count']} sample(s) across "
                f"{id_repair['duplicate_id_count']} colliding ID(s)."
            )
            self.reporter.emit(
                f"Repaired {id_repair['duplicate_id_count']} duplicate sample ID(s)",
                event="ingestion_ids_repaired",
                status="completed",
                **id_repair,
            )
        items, task_partitions = partition_heterogeneous_tasks(items)
        if task_partitions:
            warnings.append(
                "Deterministically partitioned heterogeneous task labels by answer and metric contract."
            )
        validation = validate_unified_items(
            items,
            self.output_root,
            strict_choice_labels=True,
        )
        incomplete_sources = {
            source: stats
            for source, stats in source_extraction.items()
            if int(stats.get("skipped_count", 0)) > 0
        }
        if incomplete_sources:
            details = "; ".join(
                f"{source}: {stats['converted_count']}/{stats['record_count']} converted"
                for source, stats in sorted(incomplete_sources.items())
            )
            validation["valid"] = False
            validation.setdefault("errors", []).append(
                "source extraction was incomplete; no benchmark rows may be silently dropped: "
                + details
            )
        missing_media_blocker = all_sources_missing_required_media(
            source_extraction,
            extraction_warnings,
        )
        if not validation["valid"]:
            decision = "unsupported"
            if missing_media_blocker:
                total_records = sum(
                    int(stats.get("record_count") or 0)
                    for stats in source_extraction.values()
                )
                reason = (
                    "The declarative annotation mapping is valid, but none of the "
                    f"{total_records} benchmark rows has resolvable input media. "
                    "Download and extract the official media archive at the paths "
                    "referenced by the annotations before building protocols."
                )
            else:
                reason = summarize_validation_failure(validation)
            items = []
        else:
            decision = "ingest"
            reason = mapping.get("reason") or "agent mapping validated and executed"
        manifest = self._manifest(
            inventory=inventory,
            mapping=mapping,
            decision=decision,
            reason=reason,
            warnings=warnings,
            items=items,
            validation=validation,
        )
        manifest["source_counts"] = dict(sorted(source_counts.items()))
        manifest["source_extraction"] = dict(sorted(source_extraction.items()))
        manifest["task_partitions"] = task_partitions
        manifest["deterministic_mapping_repairs"] = mapping_repairs
        manifest["deterministic_id_repair"] = id_repair
        manifest["blocker_type"] = "missing_media" if missing_media_blocker else ""
        return IngestionResult(
            self.benchmark_name,
            decision,
            items,
            mapping,
            manifest,
            prompt,
            str(response or ""),
        )

    def inspect(
        self,
        *,
        selected_sources: List[str] | None = None,
        include_agent_context: bool = True,
    ) -> Dict[str, Any]:
        if not self.source_root.exists():
            raise FileNotFoundError(f"benchmark source root does not exist: {self.source_root}")
        if selected_sources:
            candidates = []
            for value in selected_sources:
                path = (self.source_root / value).resolve()
                try:
                    path.relative_to(self.source_root)
                except ValueError:
                    continue
                if path.is_file() and path not in candidates:
                    candidates.append(path)
        else:
            candidates = discover_sources(self.source_root, self.max_sources)
        sources = []
        warnings = []
        for path in candidates:
            rel = str(path.relative_to(self.source_root))
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                sources.append(
                    {
                        "path": rel,
                        "format": path.suffix.lower().lstrip("."),
                        "supported": False,
                        "size_bytes": path.stat().st_size,
                    }
                )
                continue
            try:
                examples = diverse_record_examples(path, self.max_examples_per_source)
                fields = sorted({key for row in examples for key in row})
                sources.append(
                    {
                        "path": rel,
                        "format": source_format(path),
                        "supported": True,
                        "size_bytes": path.stat().st_size,
                        "fields": fields,
                        "record_examples": [summarize_record(row) for row in examples],
                        **(
                            {"record_collection": json_record_collection_path(path)}
                            if path.suffix.lower() == ".json"
                            else {}
                        ),
                    }
                )
            except Exception as exc:
                warnings.append(f"could not inspect {rel}: {type(exc).__name__}: {exc}")
                sources.append(
                    {
                        "path": rel,
                        "format": source_format(path),
                        "supported": False,
                        "size_bytes": path.stat().st_size,
                        "inspection_error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return {
            "benchmark": self.benchmark_name,
            "source_root": str(self.source_root),
            "sources": sources,
            "official_documentation_excerpt": (
                read_documentation_excerpt(self.source_root, self.readme_char_limit)
                if include_agent_context
                else ""
            ),
            "metric_evidence": (
                discover_metric_evidence(self.source_root) if include_agent_context else []
            ),
            "allowed_mapping_contract": {
                "transforms": sorted(ALLOWED_TRANSFORMS),
                "answer_modes": sorted(ALLOWED_ANSWER_MODES),
                "choice_modes": sorted(ALLOWED_CHOICE_MODES),
                "media_modes": sorted(ALLOWED_MEDIA_MODES),
                "media_value_transforms": sorted(ALLOWED_MEDIA_VALUE_TRANSFORMS),
                "metrics": sorted(ALLOWED_METRICS),
            },
            "warnings": warnings,
        }

    def _validate_mapping(
        self, payload: Dict[str, Any], inventory: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[str]]:
        mapping = dict(payload) if isinstance(payload, dict) else {}
        decision = str(mapping.get("decision") or "").strip().lower()
        if decision == "unsupported":
            mapping["decision"] = "unsupported"
            mapping.setdefault("sources", [])
            return mapping, []
        errors = []
        if decision != "ingest":
            errors.append("decision must be ingest or unsupported")
        inventory_sources = {
            row["path"]: row for row in inventory.get("sources", []) if row.get("supported")
        }
        raw_sources = mapping.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            errors.append("at least one source mapping is required")
            raw_sources = []
        normalized_sources = []
        seen_sources = set()
        metric_evidence = {
            str(row.get("id") or ""): row for row in inventory.get("metric_evidence") or []
        }
        for index, source in enumerate(raw_sources):
            if not isinstance(source, dict):
                errors.append(f"sources[{index}] must be an object")
                continue
            row = json.loads(json.dumps(source))
            rel = str(row.get("source") or "").strip()
            info = inventory_sources.get(rel)
            if info is None:
                errors.append(f"sources[{index}] references unavailable source: {rel!r}")
                continue
            if rel in seen_sources:
                errors.append(f"duplicate source mapping: {rel}")
                continue
            seen_sources.add(rel)
            fields = set(info.get("fields") or [])
            normalize_choice_mapping(row, fields)
            errors.extend(validate_source_mapping(row, fields, index))
            promote_verified_choice_accuracy(row, metric_evidence)
            normalize_mapping_metric(row, metric_evidence)
            normalized_sources.append(row)
        mapping["decision"] = "ingest" if not errors else "unsupported"
        mapping["sources"] = normalized_sources
        mapping["benchmark"] = self.benchmark_name
        return mapping, errors

    def _repair_path_template_mappings(
        self,
        mapping: Dict[str, Any],
        *,
        probe_limit: int = 64,
    ) -> List[Dict[str, Any]]:
        repairs = []
        for source_mapping in mapping.get("sources") or []:
            template_media = [
                (index, media_mapping)
                for index, media_mapping in enumerate(source_mapping.get("media") or [])
                if str(media_mapping.get("mode") or "") == "path_template"
            ]
            if not template_media:
                continue
            source_path = (self.source_root / source_mapping["source"]).resolve()
            records = []
            for record in iter_records(source_path):
                records.append(record)
                if len(records) >= probe_limit:
                    break
            for media_index, media_mapping in template_media:
                repair = infer_path_template_transform(
                    records,
                    media_mapping,
                    source_root=self.source_root,
                    source_path=source_path,
                )
                if not repair:
                    continue
                media_mapping["value_transform"] = repair["to_transform"]
                repairs.append(
                    {
                        "source": source_mapping["source"],
                        "media_index": media_index,
                        **repair,
                    }
                )
        return repairs

    def _convert_source(
        self,
        mapping: Dict[str, Any],
        *,
        id_namespace: str = "",
    ) -> tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
        source_path = (self.source_root / mapping["source"]).resolve()
        items = []
        warnings = []
        skipped_count = 0
        for row_index, record in enumerate(iter_records(source_path)):
            try:
                items.append(
                    self._convert_record(
                        mapping,
                        source_path,
                        row_index,
                        record,
                        id_namespace=id_namespace,
                    )
                )
            except Exception as exc:
                skipped_count += 1
                if len(warnings) < MAX_EXTRACTION_WARNING_EXAMPLES:
                    warnings.append(
                        f"skipped {mapping['source']} row {row_index}: "
                        f"{type(exc).__name__}: {exc}"
                    )
        if skipped_count > len(warnings):
            warnings.append(
                f"suppressed {skipped_count - len(warnings)} additional row extraction warning(s) "
                f"for {mapping['source']}"
            )
        record_count = len(items) + skipped_count
        extraction = {
            "record_count": record_count,
            "converted_count": len(items),
            "skipped_count": skipped_count,
            "coverage_rate": (len(items) / record_count if record_count else 0.0),
        }
        return items, warnings, extraction

    def _convert_record(
        self,
        mapping: Dict[str, Any],
        source_path: Path,
        row_index: int,
        record: Dict[str, Any],
        *,
        id_namespace: str = "",
    ) -> Dict[str, Any]:
        task = resolve_value(record, mapping["task"])
        question = resolve_value(record, mapping["question"])
        answer_config = mapping["answer"]
        evaluation_config = mapping["evaluation"]
        metric = str(evaluation_config["metric"])
        mask_mapping = evaluation_mask_mapping(evaluation_config)
        mask_field = str(mask_mapping.get("field") or "") if mask_mapping else ""
        answer_is_evaluation_target = str(answer_config.get("mode") or "field") == "evaluation_target"
        if (
            metric in {"point_in_mask", "mask_precision", "mask_iou"}
            and str(answer_config.get("field") or "")
            and str(answer_config.get("field")) == mask_field
        ):
            answer_is_evaluation_target = True
        answer = None if answer_is_evaluation_target else resolve_value(record, answer_config)
        if (
            not answer_is_evaluation_target
            and answer is None
            and "null_value" in answer_config
        ):
            answer = json_safe_metadata(answer_config["null_value"])
        answer_type = str(mapping.get("answer_type") or "auto")
        choices = build_choices(record, mapping.get("choices") or {}, question)
        if answer_type == "auto":
            answer_type = infer_answer_type(answer, choices)
        if not answer_is_evaluation_target:
            answer = normalize_answer(
                answer,
                answer_type,
                choices,
                choice_index_base=answer_config.get("choice_index_base"),
            )
        media = []
        for media_mapping in mapping.get("media") or []:
            media.extend(
                self._extract_media(record, source_path, row_index, media_mapping, target=False)
            )
        if not media:
            raise ValueError("no input media could be extracted")
        evaluation = {
            "metric": metric,
            "metric_config": json_safe_metadata(evaluation_config.get("metric_config") or {}),
        }
        if mask_mapping:
            mask_mapping = {**mask_mapping, "role": "mask"}
            targets = self._extract_media(
                record,
                source_path,
                row_index,
                mask_mapping,
                target=True,
            )
            if not targets:
                raise ValueError(f"target mask is missing from field {mask_field}")
            evaluation["mask_path"] = targets[0]["path"]
        if answer_is_evaluation_target:
            if not evaluation.get("mask_path"):
                raise ValueError("evaluation_target answer requires an extracted target mask")
            answer = {
                "type": "evaluation_target",
                "metric": metric,
                "path": evaluation["mask_path"],
            }

        item_id = resolve_id(
            record,
            mapping["id"],
            row_index,
            namespace=id_namespace,
        )
        metadata = {
            field: json_safe_metadata(record.get(field))
            for field in mapping.get("metadata_fields") or []
            if field in record
        }
        metadata.update(
            {
                "source_file": mapping["source"],
                "source_row_index": row_index,
            }
        )
        order_fields = {
            str(media_mapping.get("order_field"))
            for media_mapping in mapping.get("media") or []
            if media_mapping.get("order_field")
        }
        for order_field in sorted(order_fields):
            metadata[order_field] = json_safe_metadata(record.get(order_field))
        return {
            "schema_version": "genbench.v1",
            "id": item_id,
            "benchmark": self.benchmark_name,
            "task": task,
            "split": str(mapping.get("split") or "test"),
            "input": {
                "type": "multi_image" if len(media) > 1 else "image",
                "media": media,
            },
            "question": question,
            "answer": answer,
            "answer_type": answer_type,
            "choices": choices,
            "evaluation": evaluation,
            "metadata": metadata,
            "source": self.benchmark_name,
        }

    def _extract_media(
        self,
        record: Dict[str, Any],
        source_path: Path,
        row_index: int,
        mapping: Dict[str, Any],
        *,
        target: bool,
    ) -> List[Dict[str, Any]]:
        field = str(mapping["field"])
        value = record.get(field)
        if value is None:
            return []
        mode = str(mapping["mode"])
        materialize_mode = mode
        if mode == "conversation_images":
            values = conversation_image_paths(value)
            materialize_mode = "path"
        elif mode == "path_template":
            value = render_media_path_template(record, mapping)
            materialize_mode = "path"
            values = [value]
        else:
            values = (
                sequence_values(value)
                if mode in {"embedded_images", "path_list"}
                else [value]
            )
        order_field = str(mapping.get("order_field") or "")
        if order_field:
            order_values = [int(v) for v in sequence_values(record.get(order_field))]
            if len(order_values) == len(values):
                values = [value for _, value in sorted(zip(order_values, values), key=lambda pair: pair[0])]
        entries = []
        for media_index, media_value in enumerate(values):
            saved = self._materialize_media(
                media_value,
                mode=materialize_mode,
                source_path=source_path,
                row_index=row_index,
                field=field,
                target=target,
            )
            if not saved:
                if len(values) > 1:
                    return []
                continue
            entries.append(
                {
                    "type": "image",
                    "path": saved,
                    "role": str(mapping.get("role") or ("mask" if target else "primary")),
                    "label": f"Image {media_index + 1}" if len(values) > 1 else "",
                }
            )
        return entries

    def _materialize_media(
        self,
        value: Any,
        *,
        mode: str,
        source_path: Path,
        row_index: int,
        field: str,
        target: bool,
    ) -> str:
        value = scalar_value(value)
        raw_bytes = None
        source_name = ""
        if mode in {"embedded_images", "hf_image"}:
            if isinstance(value, dict):
                raw_bytes = value.get("bytes")
                source_name = str(value.get("path") or "")
                if raw_bytes is None and source_name:
                    candidate = Path(source_name)
                    if not candidate.is_absolute():
                        candidate = source_path.parent / candidate
                    if candidate.exists():
                        raw_bytes = candidate.read_bytes()
            elif isinstance(value, (bytes, bytearray, memoryview)):
                raw_bytes = bytes(value)
            elif mode == "embedded_images" and isinstance(value, str):
                encoded = value.strip()
                if encoded.startswith("data:"):
                    header, separator, encoded = encoded.partition(",")
                    if not separator or ";base64" not in header.lower():
                        raise ValueError(f"unsupported embedded image data URI for {field}")
                encoded = re.sub(r"\s+", "", encoded)
                try:
                    raw_bytes = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError(f"invalid base64 image data for {field}") from exc
        elif mode in {"path", "path_list"}:
            candidate = resolve_media_source_path(
                self.source_root,
                source_path,
                text_value(value),
                directory_cache=self._media_directory_cache,
                child_directory_cache=self._media_child_directory_cache,
            )
            parent_known_missing = self._media_directory_cache.get(str(candidate.parent)) is False
            if not parent_known_missing and candidate.exists():
                raw_bytes = candidate.read_bytes()
                source_name = candidate.name
        if not raw_bytes:
            return ""
        return self._save_image_bytes(
            bytes(raw_bytes),
            source_name=source_name,
            namespace="targets" if target else "media",
            context=f"{source_path.name}:{row_index}:{field}",
        )

    def _save_image_bytes(
        self, data: bytes, *, source_name: str, namespace: str, context: str
    ) -> str:
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
                detected_format = str(image.format or "PNG").lower()
        except Exception as exc:
            raise ValueError(f"invalid image bytes for {context}: {exc}") from exc
        suffix = Path(source_name).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            suffix = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "webp": ".webp"}.get(
                detected_format, ".png"
            )
        digest = hashlib.sha256(data).hexdigest()
        rel = Path("assets") / namespace / f"{digest}{suffix}"
        destination = self.output_root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.write_bytes(data)
        return str(rel)

    def _manifest(
        self,
        *,
        inventory: Dict[str, Any],
        mapping: Dict[str, Any],
        decision: str,
        reason: str,
        warnings: List[str],
        items: List[Dict[str, Any]],
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "ingestor": "agentic_benchmark_ingestor_v2",
            "source_root": str(self.source_root),
            "output_root": str(self.output_root),
            "decision": decision,
            "reason": reason,
            "sample_count": len(items),
            "task_counts": dict(sorted(Counter(item["task"] for item in items).items())),
            "mapping": mapping,
            "validation": validation,
            "inventory_summary": {
                "source_count": len(inventory.get("sources") or []),
                "supported_source_count": sum(
                    bool(row.get("supported")) for row in inventory.get("sources") or []
                ),
                "record_collections": {
                    str(row.get("path")): str(row.get("record_collection"))
                    for row in inventory.get("sources") or []
                    if row.get("record_collection")
                },
            },
            "warnings": warnings,
        }


def discover_sources(root: Path, limit: int) -> List[Path]:
    candidates = []
    preferred_names = {"annotations", "benchmark", "data", "dataset", "datasets", "records"}
    preferred_roots = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.lower() in preferred_names
    ]
    search_roots = preferred_roots or [root]
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES | ARCHIVE_SUFFIXES:
            candidates.append(path)
    for search_root in search_roots:
        for path in iter_inventory_files(search_root):
            suffix = path.suffix.lower()
            if suffix in SUPPORTED_SUFFIXES or suffix in ARCHIVE_SUFFIXES:
                candidates.append(path)

    def priority(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        if any(token in name for token in ("test", "eval", "benchmark")):
            tier = 0
        elif any(token in name for token in ("train", "sft")):
            tier = 2
        else:
            tier = 1
        supported = 0 if path.suffix.lower() in SUPPORTED_SUFFIXES else 1
        return supported, tier, str(path)

    return sorted(set(candidates), key=priority)[:limit]


def read_documentation_excerpt(root: Path, limit: int) -> str:
    paths = []
    for path in iter_inventory_files(root, max_depth=2, skip_dirs=INVENTORY_SKIP_DIRS - {"assets"}):
        if path.name.lower() in {"readme.md", "readme.txt", "dataset_info.md"}:
            paths.append(path)
    paths.sort(key=lambda path: (len(path.relative_to(root).parts), str(path)))
    chunks = []
    remaining = limit
    for path in paths:
        if remaining <= 0:
            break
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        chunk = text[:remaining]
        chunks.append(f"===== {path.relative_to(root)} =====\n{chunk}")
        remaining -= len(chunk)
    return "\n\n".join(chunks)


def discover_metric_evidence(
    root: Path,
    limit: int = 12000,
) -> List[Dict[str, str]]:
    candidates = []
    terms = re.compile(
        r"(?i)(metric|accuracy|exact.match|score|success|threshold|tolerance|delta|iou|"
        r"precision|relative.error|mra|point.in.mask|evaluate|process.results|metric.list|"
        r"doc.to.target|aggregation|overall.acc|category.acc|overall.hit.rate|hit.rate|"
        r"pred\s*[!=]=\s*gt|prediction\s*[!=]=\s*answer|calculate.accuracy|can.infer)"
    )
    preferred_names = {"code", "eval", "evaluation", "evaluator", "metrics", "scripts"}
    preferred_roots = [
        path for path in root.iterdir() if path.is_dir() and path.name.lower() in preferred_names
    ]
    search_roots = preferred_roots or [root]
    paths = [path for path in root.iterdir() if path.is_file()]
    for search_root in search_roots:
        paths.extend(iter_metric_files(search_root))
    seen_paths = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in {".py", ".ipynb", ".md", ".txt", ".yaml", ".yml"}:
            continue
        try:
            if path.stat().st_size > 2_000_000:
                continue
            if path.suffix.lower() == ".ipynb":
                notebook = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                text = "\n".join(
                    "".join(cell.get("source") or [])
                    for cell in notebook.get("cells") or []
                    if cell.get("cell_type") in {"code", "markdown"}
                )
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        lines = text.splitlines()
        selected = set()
        for index, line in enumerate(lines):
            if terms.search(line):
                selected.update(range(max(0, index - 2), min(len(lines), index + 4)))
        if not selected:
            continue
        candidates.append((metric_evidence_priority(path, text), path, lines, selected))

    candidates.sort(
        key=lambda row: (-row[0], len(row[1].relative_to(root).parts), str(row[1]))
    )
    rows = []
    remaining = int(limit)
    for _, path, lines, selected in candidates:
        if remaining <= 0:
            break
        excerpt = "\n".join(f"{index + 1}: {lines[index]}" for index in sorted(selected))
        excerpt = excerpt[: min(remaining, 4000)]
        rel = str(path.relative_to(root))
        rows.append(
            {
                "id": f"metric:{rel}",
                "source": rel,
                "kind": (
                    "official_code"
                    if path.suffix.lower() in {".py", ".ipynb", ".yaml", ".yml"}
                    else "official_documentation"
                ),
                "excerpt": excerpt,
            }
        )
        remaining -= len(excerpt)
    return rows


def iter_inventory_files(
    root: Path,
    *,
    max_depth: int | None = None,
    skip_dirs: set[str] | None = None,
) -> Iterator[Path]:
    skipped = {value.lower() for value in (skip_dirs or INVENTORY_SKIP_DIRS)}
    for directory, dirnames, filenames in os.walk(root):
        current = Path(directory)
        depth = len(current.relative_to(root).parts)
        dirnames[:] = [
            name
            for name in dirnames
            if not name.startswith(".") and name.lower() not in skipped
        ]
        if max_depth is not None and depth >= max_depth:
            dirnames[:] = []
        for filename in filenames:
            if filename.startswith("."):
                continue
            yield current / filename


def iter_metric_files(root: Path) -> Iterator[Path]:
    allowed = {".py", ".ipynb", ".md", ".txt", ".yaml", ".yml"}
    name_tokens = ("aggregate", "eval", "metric", "score", "task", "util")
    for path in iter_inventory_files(root):
        suffix = path.suffix.lower()
        if suffix not in allowed:
            continue
        name = path.name.lower()
        parts = {part.lower() for part in path.parts}
        if suffix in {".yaml", ".yml"} and ("task" in parts or "tasks" in parts):
            yield path
            continue
        if name in {"readme.md", "readme.txt", "dataset_info.md"}:
            yield path
            continue
        if any(token in name for token in name_tokens):
            yield path


def metric_evidence_priority(path: Path, text: str) -> int:
    lower = text.lower()
    parts = {part.lower() for part in path.parts}
    score = 0
    score += 35 if "metric_list" in lower else 0
    score += 30 if "process_results" in lower else 0
    score += 80 if "overall_acc" in lower or "category_acc" in lower else 0
    score += 80 if "overall_hit_rate" in lower or "pred != gt" in lower else 0
    score += 20 if "doc_to_target" in lower else 0
    score += 20 if "can_infer_option" in lower or "can_infer_text" in lower else 0
    score += 15 if "aggregation" in lower else 0
    score += 15 if "accuracy" in lower or "exact match" in lower else 0
    score += 10 if "tasks" in parts or "task" in parts else 0
    score += 5 if any(token in path.name.lower() for token in ("eval", "metric", "score", "util")) else 0
    return score


def diverse_record_examples(path: Path, limit: int, scan_limit: int = 200) -> List[Dict[str, Any]]:
    selected = []
    seen = set()
    effective_scan_limit = (
        min(scan_limit, max(limit * 4, 12)) if path.suffix.lower() == ".parquet" else scan_limit
    )
    for index, row in enumerate(iter_records(path, batch_size=min(128, effective_scan_limit))):
        fingerprint = record_fingerprint(row)
        if fingerprint not in seen:
            selected.append(row)
            seen.add(fingerprint)
            if len(selected) >= limit:
                break
        if index + 1 >= effective_scan_limit:
            break
    return selected


def iter_records(path: Path, *, batch_size: int = 128) -> Iterator[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value
        return
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        _, rows = select_json_record_collection(payload)
        for value in rows:
            if isinstance(value, dict):
                yield value
        return
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        _raise_csv_field_limit()
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle, delimiter=delimiter)
        return
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=max(1, int(batch_size))):
            yield from batch.to_pylist()
        return
    if suffix in {".tfrecord", ".tfrecords"}:
        from tfrecord.reader import tfrecord_loader

        yield from tfrecord_loader(str(path), None, description=None)
        return
    raise ValueError(f"unsupported source format: {path}")


def select_json_record_collection(payload: Any) -> tuple[str, List[Dict[str, Any]]]:
    candidates: List[tuple[int, str, List[Dict[str, Any]]]] = []

    def visit(value: Any, path: tuple[str, ...], depth: int) -> None:
        if depth > 4:
            return
        if isinstance(value, list):
            rows = [row for row in value if isinstance(row, dict)]
            if rows and len(rows) == len(value):
                path_text = ".".join(path) or "$"
                candidates.append((json_collection_priority(path, len(rows)), path_text, rows))
                return
            for index, row in enumerate(value[:10]):
                if isinstance(row, (dict, list)):
                    visit(row, (*path, str(index)), depth + 1)
            return
        if isinstance(value, dict):
            for key, row in value.items():
                if isinstance(row, (dict, list)):
                    visit(row, (*path, str(key)), depth + 1)

    visit(payload, (), 0)
    if not candidates:
        return "", []
    candidates.sort(key=lambda row: (-row[0], row[1]))
    _, path, rows = candidates[0]
    return path, rows


def json_record_collection_path(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    collection, _ = select_json_record_collection(payload)
    return collection


def json_collection_priority(path: tuple[str, ...], row_count: int) -> int:
    tokens = [token.lower() for token in path]
    score = min(int(row_count), 10000)
    if any(token in {"test", "eval", "evaluation", "validation", "val"} for token in tokens):
        score += 100_000
    if any(
        token in {"data", "questions", "samples", "records", "items", "annotations", "instances", "examples"}
        for token in tokens
    ):
        score += 50_000
    if any(token in {"train", "training", "sft"} for token in tokens):
        score -= 100_000
    if not path:
        score += 25_000
    return score


def source_format(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def record_fingerprint(record: Dict[str, Any]) -> tuple[Any, ...]:
    return tuple(sorted((key, value_shape(value)) for key, value in record.items()))


def value_shape(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray:{value.dtype}:{tuple(value.shape)}"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "bytes"
    if isinstance(value, dict):
        return "dict:" + ",".join(sorted(value))
    if isinstance(value, list):
        subtype = type(value[0]).__name__ if value else "empty"
        return f"list:{len(value)}:{subtype}"
    return type(value).__name__


def summarize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: summarize_value(value) for key, value in record.items()}


def summarize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        values = value.tolist()
        return {
            "type": "array",
            "shape": list(value.shape),
            "items": [summarize_value(item) for item in values[:6]],
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return {"type": "bytes", "length": len(data), "looks_like_image": looks_like_image(data)}
        return {"type": "text_bytes", "value": text[:800]}
    if isinstance(value, dict):
        return {key: summarize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [summarize_value(item) for item in value[:6]]
    if isinstance(value, str):
        if looks_like_base64_image_text(value):
            return {"type": "base64_image", "encoded_length": len(value)}
        return value[:800]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:800]


def looks_like_image(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        return True
    except Exception:
        return False


def looks_like_base64_image_text(value: str) -> bool:
    encoded = str(value or "").strip()
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header.lower():
            return False
    encoded = re.sub(r"\s+", "", encoded)
    if len(encoded) < 64 or not re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", encoded):
        return False
    prefix_length = min(len(encoded), 256)
    prefix_length -= prefix_length % 4
    try:
        prefix = base64.b64decode(encoded[:prefix_length], validate=True)
    except (binascii.Error, ValueError):
        return False
    return (
        prefix.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"BM"))
        or (prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP")
    )


def normalize_choice_mapping(mapping: Dict[str, Any], fields: set[str]) -> None:
    choices = mapping.get("choices")
    if not isinstance(choices, dict) or str(choices.get("mode") or "none") != "field":
        return
    selected_field = str(choices.get("field") or "")
    letter_fields = []
    for codepoint in range(ord("A"), ord("Z") + 1):
        label = chr(codepoint)
        if label not in fields:
            break
        letter_fields.append(label)
    if len(letter_fields) < 2 or selected_field not in letter_fields:
        return
    mapping["choices"] = {
        "mode": "fields",
        "fields": letter_fields,
        "labels": letter_fields,
    }


def evaluation_mask_mapping(evaluation: Mapping[str, Any]) -> Dict[str, Any] | None:
    """Return the explicit target-mask mapping or the legacy field shorthand."""
    mask = evaluation.get("mask")
    if isinstance(mask, dict) and mask:
        return dict(mask)
    mask_field = str(evaluation.get("mask_field") or "").strip()
    if mask_field:
        return {"field": mask_field, "mode": "hf_image"}
    return None


def validate_source_mapping(mapping: Dict[str, Any], fields: set[str], index: int) -> List[str]:
    errors = []
    prefix = f"sources[{index}]"
    for section in ("id", "task", "question", "answer", "choices", "evaluation"):
        if not isinstance(mapping.get(section), dict):
            errors.append(f"{prefix}.{section} must be an object")
    id_cfg = mapping.get("id") or {}
    if id_cfg.get("mode") not in {"row_index", "field"}:
        errors.append(f"{prefix}.id.mode must be row_index or field")
    if id_cfg.get("mode") == "field":
        validate_field(id_cfg.get("field"), fields, f"{prefix}.id.field", errors)
    task_cfg = mapping.get("task") or {}
    if not task_cfg.get("constant"):
        validate_field(task_cfg.get("field"), fields, f"{prefix}.task.field", errors)
    for section in ("task", "question", "answer"):
        cfg = mapping.get(section) or {}
        transform = str(cfg.get("transform") or "text")
        if transform not in ALLOWED_TRANSFORMS:
            errors.append(f"{prefix}.{section}.transform is unsupported: {transform}")
    validate_field((mapping.get("question") or {}).get("field"), fields, f"{prefix}.question.field", errors)
    answer = mapping.get("answer") or {}
    answer_mode = str(answer.get("mode") or "field")
    if answer_mode not in ALLOWED_ANSWER_MODES:
        errors.append(f"{prefix}.answer.mode is unsupported: {answer_mode}")
    elif answer_mode == "field":
        validate_field(answer.get("field"), fields, f"{prefix}.answer.field", errors)
    if answer.get("choice_index_base") not in {None, 0, 1, "0", "1"}:
        errors.append(f"{prefix}.answer.choice_index_base must be 0 or 1")
    answer_type = str(mapping.get("answer_type") or "auto")
    if answer_type not in ALLOWED_ANSWER_TYPES:
        errors.append(f"{prefix}.answer_type is unsupported: {answer_type}")
    choices = mapping.get("choices") or {}
    choice_mode = str(choices.get("mode") or "none")
    if choice_mode not in ALLOWED_CHOICE_MODES:
        errors.append(f"{prefix}.choices.mode is unsupported: {choice_mode}")
    labels = choices.get("labels")
    if labels is not None:
        if (
            not isinstance(labels, list)
            or len(labels) < 2
            or any(not str(label).strip() for label in labels)
            or len({str(label) for label in labels}) != len(labels)
        ):
            errors.append(f"{prefix}.choices.labels must contain at least two unique labels")
    if choice_mode == "field":
        validate_field(choices.get("field"), fields, f"{prefix}.choices.field", errors)
    elif choice_mode == "fields":
        choice_fields = choices.get("fields")
        if not isinstance(choice_fields, list) or len(choice_fields) < 2:
            errors.append(f"{prefix}.choices.fields must contain at least two fields")
        else:
            for field in choice_fields:
                validate_field(field, fields, f"{prefix}.choices.fields", errors)
        if labels is not None and (
            not isinstance(labels, list) or len(labels) != len(choice_fields or [])
        ):
            errors.append(f"{prefix}.choices.labels must align one-to-one with choices.fields")
    media = mapping.get("media")
    if not isinstance(media, list) or not media:
        errors.append(f"{prefix}.media must contain at least one mapping")
    else:
        for media_index, media_cfg in enumerate(media):
            if not isinstance(media_cfg, dict):
                errors.append(f"{prefix}.media[{media_index}] must be an object")
                continue
            validate_field(media_cfg.get("field"), fields, f"{prefix}.media[{media_index}].field", errors)
            mode = str(media_cfg.get("mode") or "")
            if mode not in ALLOWED_MEDIA_MODES:
                errors.append(f"{prefix}.media[{media_index}].mode is unsupported: {mode}")
            if mode == "path_template":
                template = str(media_cfg.get("template") or "").strip()
                if not template:
                    errors.append(
                        f"{prefix}.media[{media_index}].template is required for path_template"
                    )
                value_transform = str(media_cfg.get("value_transform") or "text")
                if value_transform not in ALLOWED_MEDIA_VALUE_TRANSFORMS:
                    errors.append(
                        f"{prefix}.media[{media_index}].value_transform is unsupported: "
                        f"{value_transform}"
                    )
            if media_cfg.get("order_field"):
                validate_field(
                    media_cfg.get("order_field"),
                    fields,
                    f"{prefix}.media[{media_index}].order_field",
                    errors,
                )
    evaluation = mapping.get("evaluation") or {}
    metric = str(evaluation.get("metric") or "")
    if metric not in ALLOWED_METRICS:
        errors.append(f"{prefix}.evaluation.metric is unsupported: {metric}")
    if evaluation.get("metric_config") is not None and not isinstance(
        evaluation.get("metric_config"), dict
    ):
        errors.append(f"{prefix}.evaluation.metric_config must be an object")
    raw_mask_mapping = evaluation.get("mask")
    if raw_mask_mapping is not None and not isinstance(raw_mask_mapping, dict):
        errors.append(f"{prefix}.evaluation.mask must be an object")
    mask_mapping = evaluation_mask_mapping(evaluation)
    if mask_mapping:
        validate_field(
            mask_mapping.get("field"),
            fields,
            f"{prefix}.evaluation.mask.field",
            errors,
        )
        mask_mode = str(mask_mapping.get("mode") or "")
        if mask_mode not in ALLOWED_TARGET_MEDIA_MODES:
            errors.append(f"{prefix}.evaluation.mask.mode is unsupported: {mask_mode}")
        if mask_mode == "path_template":
            if not str(mask_mapping.get("template") or "").strip():
                errors.append(
                    f"{prefix}.evaluation.mask.template is required for path_template"
                )
            value_transform = str(mask_mapping.get("value_transform") or "text")
            if value_transform not in ALLOWED_MEDIA_VALUE_TRANSFORMS:
                errors.append(
                    f"{prefix}.evaluation.mask.value_transform is unsupported: "
                    f"{value_transform}"
                )
    if answer_mode == "evaluation_target":
        if metric not in {"point_in_mask", "mask_precision", "mask_iou"}:
            errors.append(
                f"{prefix}.answer.mode=evaluation_target is incompatible with metric {metric!r}"
            )
        if not mask_mapping:
            errors.append(f"{prefix}.answer.mode=evaluation_target requires evaluation.mask")
        if answer_type not in {"points", "mask"}:
            errors.append(
                f"{prefix}.answer.mode=evaluation_target requires answer_type points or mask"
            )
    metadata_fields = mapping.get("metadata_fields") or []
    if not isinstance(metadata_fields, list):
        errors.append(f"{prefix}.metadata_fields must be a list")
    else:
        for field in metadata_fields:
            validate_field(field, fields, f"{prefix}.metadata_fields", errors)
    return errors


def render_media_path_template(
    record: Mapping[str, Any], mapping: Mapping[str, Any]
) -> str:
    field = str(mapping.get("field") or "")
    value = text_value(record.get(field))
    transform = str(mapping.get("value_transform") or "text")
    if transform == "first_underscore":
        value = value.split("_", 1)[0]
    elif transform == "basename":
        value = Path(value).name
    elif transform == "stem":
        value = Path(value).stem
    elif transform != "text":
        raise ValueError(f"unsupported path_template value_transform: {transform}")
    values = {key: text_value(item) for key, item in record.items()}
    values["value"] = value
    try:
        return str(mapping.get("template") or "").format_map(values)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"could not render media path_template: {exc}") from exc


def infer_path_template_transform(
    records: List[Dict[str, Any]],
    mapping: Mapping[str, Any],
    *,
    source_root: Path,
    source_path: Path,
) -> Dict[str, Any] | None:
    field = str(mapping.get("field") or "")
    eligible = [record for record in records if record.get(field) is not None]
    if not eligible:
        return None

    current = str(mapping.get("value_transform") or "text")
    probes: Dict[str, tuple[int, tuple[str, ...]]] = {}
    for transform in MEDIA_VALUE_TRANSFORM_ORDER:
        candidate_mapping = dict(mapping)
        candidate_mapping["value_transform"] = transform
        resolved = []
        for record in eligible:
            try:
                rendered = render_media_path_template(record, candidate_mapping)
                candidate = resolve_media_source_path(source_root, source_path, rendered)
            except (TypeError, ValueError):
                continue
            if candidate.is_file():
                resolved.append(str(candidate.resolve()))
        probes[transform] = (len(resolved), tuple(resolved))

    if probes.get(current, (0, ()))[0] != 0:
        return None
    full_coverage = {
        signature: []
        for transform, (count, signature) in probes.items()
        if transform != current and count == len(eligible)
    }
    for transform, (count, signature) in probes.items():
        if transform != current and count == len(eligible):
            full_coverage[signature].append(transform)
    if len(full_coverage) != 1:
        return None

    transforms = next(iter(full_coverage.values()))
    selected = next(
        transform for transform in MEDIA_VALUE_TRANSFORM_ORDER if transform in transforms
    )
    return {
        "from_transform": current,
        "to_transform": selected,
        "probe_count": len(eligible),
    }


def resolve_media_source_path(
    source_root: Path,
    source_path: Path,
    value: str,
    *,
    directory_cache: Dict[str, bool] | None = None,
    child_directory_cache: Dict[str, set[str]] | None = None,
) -> Path:
    def candidate_from(base: Path) -> tuple[Path, bool]:
        candidate = base / raw_path
        if child_directory_cache is not None and len(raw_path.parts) > 1:
            base_key = str(base)
            child_directories = child_directory_cache.get(base_key)
            if child_directories is None:
                base_exists = directory_cache.get(base_key) if directory_cache is not None else None
                if base_exists is None:
                    base_exists = base.is_dir()
                    if directory_cache is not None:
                        directory_cache[base_key] = base_exists
                child_directories = (
                    {entry.name for entry in base.iterdir() if entry.is_dir()}
                    if base_exists
                    else set()
                )
                child_directory_cache[base_key] = child_directories
            if raw_path.parts[0] not in child_directories:
                if directory_cache is not None:
                    directory_cache[str(candidate.parent)] = False
                return candidate, False
        if directory_cache is not None:
            parent_key = str(candidate.parent)
            parent_exists = directory_cache.get(parent_key)
            if parent_exists is None:
                parent_exists = candidate.parent.is_dir()
                directory_cache[parent_key] = parent_exists
            if not parent_exists:
                return candidate, False
        return resolve_image_candidate(candidate), True

    raw_path = Path(value)
    if raw_path.is_absolute():
        return resolve_image_candidate(raw_path)
    source_relative, source_parent_exists = candidate_from(source_path.parent)
    if source_parent_exists and source_relative.exists():
        return source_relative
    root_relative, root_parent_exists = candidate_from(source_root)
    if root_parent_exists and root_relative.exists():
        return root_relative
    for directory in ("data", "images", "media", "assets"):
        candidate, parent_exists = candidate_from(source_root / directory)
        if parent_exists and candidate.exists():
            return candidate
    return root_relative


def resolve_image_candidate(path: Path) -> Path:
    if path.exists() or path.suffix:
        return path
    for suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return path


def normalize_mapping_metric(
    mapping: Dict[str, Any], metric_evidence: Dict[str, Dict[str, Any]]
) -> None:
    """Validate a selected metric against discovered evidence without persisting it."""
    evaluation = mapping.setdefault("evaluation", {})
    evaluation.pop("metric_provenance", None)
    metric = str(evaluation.get("metric") or "unverified").strip().lower()
    if metric not in ALLOWED_METRICS:
        evaluation["metric"] = "unverified"
        return
    if metric == "unverified":
        return
    supported = any(
        metric_evidence_supports(metric, str(row.get("excerpt") or ""))
        for row in metric_evidence.values()
    )
    if metric_evidence and not supported:
        evaluation["metric"] = "unverified"


def promote_verified_choice_accuracy(
    mapping: Dict[str, Any], metric_evidence: Dict[str, Dict[str, Any]]
) -> None:
    """Recover exact choice accuracy when an official evaluator only normalizes replies."""
    evaluation = mapping.setdefault("evaluation", {})
    if str(evaluation.get("metric") or "").strip().lower() != "unverified":
        return
    if str(mapping.get("answer_type") or "auto").strip().lower() not in {"auto", "choice"}:
        return
    answer = mapping.get("answer") if isinstance(mapping.get("answer"), dict) else {}
    choices = mapping.get("choices") if isinstance(mapping.get("choices"), dict) else {}
    if str(answer.get("mode") or "field").strip().lower() != "field":
        return
    if str(choices.get("mode") or "none").strip().lower() == "none":
        return

    supported = False
    for candidate in metric_evidence.values():
        excerpt = str(candidate.get("excerpt") or "")
        text = excerpt.lower()
        if not metric_evidence_supports("accuracy", excerpt):
            continue
        if not any(token in text for token in ("answer", "prediction", "pred", "option", "choice")):
            continue
        supported = True
        break
    if not supported:
        return

    evaluation["metric"] = "accuracy"
    evaluation["metric_config"] = dict(evaluation.get("metric_config") or {})


def metric_evidence_supports(metric: str, excerpt: str) -> bool:
    metric = str(metric or "").strip().lower()
    text = str(excerpt or "").lower()
    required_any = {
        "accuracy": (
            "accuracy",
            "exact match",
            "overall_acc",
            "category_acc",
            "overall_hit_rate",
            "pred != gt",
            "prediction != answer",
            "calculate_accuracy",
        ),
        "exact_match": ("exact match", "=="),
        "exact_count": ("count", "exact"),
        "point_in_mask": (
            "point_in_mask",
            "point-in-mask",
            "point in mask",
            "mask pixel",
        ),
        "point_distance": ("point distance", "euclidean distance", "distance threshold"),
        "bbox_iou": ("bbox_iou", "box iou", "bounding box iou", "intersection over union"),
        "angle_error": ("angle error", "angular error", "degree threshold"),
        "mask_precision": ("precision",),
        "mask_iou": ("iou", "intersection over union"),
        "numeric_absolute_error": ("absolute error", "mae"),
        "numeric_error": ("absolute error", "relative error", "tolerance", "delta"),
        "numeric_relative_error": ("relative error",),
        "qspatial_ratio": ("delta", "pred_value", "ground_truth_value"),
        "mra": ("mean relative accuracy", "mra"),
        "dfd": ("discrete frechet", "dfd"),
        "state_similarity": ("similarity", "clip"),
    }
    if metric == "unverified" or not text:
        return False
    terms = required_any.get(metric)
    return bool(terms and any(term in text for term in terms))


def validate_field(value: Any, fields: set[str], label: str, errors: List[str]) -> None:
    field = str(value or "").strip()
    if not field or field not in fields:
        errors.append(f"{label} references unavailable field: {field!r}")


def resolve_value(record: Dict[str, Any], config: Dict[str, Any]) -> Any:
    if config.get("constant") not in (None, ""):
        value = config["constant"]
    else:
        value = record.get(str(config.get("field") or ""))
    transform = str(config.get("transform") or "text")
    if transform == "raw":
        return json_safe_metadata(value)
    if transform == "conversation_user_text":
        return conversation_text(value, role="user") or str(config.get("default") or "")
    if transform == "conversation_assistant_text":
        return conversation_text(value, role="assistant") or str(config.get("default") or "")
    values = sequence_values(value)
    if transform in {"first_text", "first_slug"}:
        value = values[0] if values else config.get("default", "")
    if transform in {"text", "first_text"}:
        result = text_value(value)
    elif transform in {"slug", "first_slug"}:
        result = slugify(text_value(value))
    else:
        raise ValueError(f"unsupported transform: {transform}")
    return result or str(config.get("default") or "")


def resolve_id(
    record: Dict[str, Any],
    config: Dict[str, Any],
    row_index: int,
    *,
    namespace: str = "",
) -> str:
    prefix = slugify(str(config.get("prefix") or "sample")) or "sample"
    id_prefix = "_".join(part for part in (prefix, slugify(namespace) if namespace else "") if part)
    if config.get("mode") == "field":
        value = text_value(record.get(str(config.get("field") or "")))
        if value:
            return f"{id_prefix}_{slugify(value)}"
    return f"{id_prefix}_{row_index:06d}"


def source_namespace(source: str) -> str:
    path = Path(str(source))
    without_suffix = path.with_suffix("")
    return slugify("_".join(without_suffix.parts))


def namespace_duplicate_item_ids(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Make colliding source IDs unique without changing already-unique IDs."""
    counts = Counter(str(item.get("id") or "") for item in items)
    duplicate_ids = {sample_id for sample_id, count in counts.items() if sample_id and count > 1}
    if not duplicate_ids:
        return {}

    used = {
        sample_id
        for sample_id, count in counts.items()
        if sample_id and count == 1
    }
    repaired_examples = []
    affected = 0
    for item in items:
        original_id = str(item.get("id") or "")
        if original_id not in duplicate_ids:
            continue
        task_namespace = slugify(str(item.get("task") or "task")) or "task"
        base = f"{original_id}_{task_namespace}"
        candidate = base
        ordinal = 2
        while candidate in used:
            candidate = f"{base}_{ordinal}"
            ordinal += 1
        item["id"] = candidate
        used.add(candidate)
        affected += 1
        if len(repaired_examples) < 10:
            repaired_examples.append(
                {"original_id": original_id, "task": str(item.get("task") or ""), "id": candidate}
            )

    return {
        "strategy": "task_namespace_then_stable_ordinal",
        "duplicate_id_count": len(duplicate_ids),
        "affected_sample_count": affected,
        "examples": repaired_examples,
    }


def build_choices(record: Dict[str, Any], config: Dict[str, Any], question: str) -> List[Dict[str, str]]:
    mode = str(config.get("mode") or "none")
    configured_labels = [str(label) for label in config.get("labels") or []]
    if mode == "none":
        return [{"label": label, "text": ""} for label in configured_labels]
    if mode == "boolean":
        return [{"label": "yes", "text": "yes"}, {"label": "no", "text": "no"}]
    if mode == "inline_mcq":
        choices = parse_inline_choices(question)
        if not choices:
            raise ValueError("inline_mcq was selected but no choices could be parsed")
        return choices
    if mode == "fields":
        fields = list(config.get("fields") or [])
        return [
            {
                "label": str(
                    configured_labels[index]
                    if index < len(configured_labels)
                    else chr(ord("A") + index)
                ),
                "text": text_value(record.get(str(field))),
            }
            for index, field in enumerate(fields)
        ]
    raw = record.get(str(config.get("field") or ""))
    values = sequence_values(raw)
    if not values and configured_labels:
        return [{"label": label, "text": ""} for label in configured_labels]
    if configured_labels and len(configured_labels) < len(values):
        raise ValueError("choices.labels does not cover every annotated choice")
    choices = []
    for index, value in enumerate(values):
        if isinstance(value, dict):
            label = (
                value.get("label")
                or (configured_labels[index] if index < len(configured_labels) else None)
                or chr(ord("A") + index)
            )
            text = value.get("text") or value.get("value") or label
        else:
            label = (
                configured_labels[index]
                if index < len(configured_labels)
                else chr(ord("A") + index)
            )
            text = text_value(value)
        choices.append({"label": str(label), "text": str(text)})
    return choices


def parse_inline_choices(question: str) -> List[Dict[str, str]]:
    match = re.search(
        r"\b(?:choices?|options?)\b(?:\s+(?:are|follow))?\s*[:.]\s*",
        question,
        flags=re.IGNORECASE,
    )
    segment = question[match.end() :] if match else question
    raw_markers = list(
        re.finditer(r"(?<![A-Za-z0-9])([A-H])\s*(?:[.)]|:)\s*", segment)
    )
    markers = []
    for start_index, start_marker in enumerate(raw_markers):
        if start_marker.group(1) != "A":
            continue
        candidate = []
        expected = ord("A")
        for marker in raw_markers[start_index:]:
            if ord(marker.group(1)) == expected:
                candidate.append(marker)
                expected += 1
        if len(candidate) > len(markers):
            markers = candidate
    choices = []
    for index, marker in enumerate(markers):
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(segment)
        text = " ".join(segment[start:end].strip().split())
        text = re.split(
            r"(?:^|\s+)(?:Please\s+answer|Answer\s+directly|Respond\s+with|Output\s+only)\b",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip()
        choices.append({"label": marker.group(1), "text": text})
    return choices


def infer_answer_type(answer: Any, choices: List[Dict[str, str]]) -> str:
    if choices:
        labels = {choice["label"].lower() for choice in choices}
        return "boolean" if labels == {"yes", "no"} else "choice"
    if isinstance(answer, bool):
        return "boolean"
    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        return "number"
    if (
        isinstance(answer, (list, tuple))
        and len(answer) == 4
        and all(isinstance(value, (int, float)) for value in answer)
    ):
        return "bbox"
    return "text"


def normalize_answer(
    answer: Any,
    answer_type: str,
    choices: List[Dict[str, str]],
    *,
    choice_index_base: Any = None,
) -> Any:
    if answer_type == "boolean":
        text = text_value(answer).strip().lower()
        if text in {"yes", "true", "1"}:
            return "yes"
        if text in {"no", "false", "0"}:
            return "no"
        return text
    if answer_type == "choice":
        numeric = scalar_value(answer)
        if isinstance(numeric, (int, np.integer)) and not isinstance(numeric, bool):
            if not choices:
                raise ValueError(
                    "numeric choice answer requires annotated choices or explicit choices.labels"
                )
            if (
                all(not str(choice.get("text") or "").strip() for choice in choices)
                and choice_index_base not in {0, 1, "0", "1"}
            ):
                raise ValueError(
                    "numeric choice answer with label-only choices requires "
                    "an explicit choice_index_base"
                )
            base = int(choice_index_base) if choice_index_base in {0, 1, "0", "1"} else 0
            index = int(numeric) - base
            if 0 <= index < len(choices):
                return choices[index]["label"]
        text = text_value(answer).strip()
        labels = {choice["label"] for choice in choices}
        if text in labels:
            return text
        upper = text.upper().strip("(). ")
        if upper in labels:
            return upper
        normalized_text = normalize_choice_text(text)
        matching_labels = [
            str(choice["label"])
            for choice in choices
            if normalize_choice_text(choice.get("text")) == normalized_text
        ]
        return matching_labels[0] if len(matching_labels) == 1 else text
    if answer_type == "number":
        value = scalar_value(answer)
        return float(value) if isinstance(value, (int, float)) else text_value(value)
    if answer_type == "points":
        return text_value(answer)
    return (
        json_safe_metadata(answer)
        if answer_type in {"bbox", "mask"}
        else text_value(answer)
    )


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while limit > 0:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def normalize_choice_text(value: Any) -> str:
    return " ".join(text_value(value).strip().rstrip(".").casefold().split())


def sequence_values(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def conversation_text(value: Any, *, role: str) -> str:
    chunks = []
    for message in sequence_values(value):
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role") or "").strip().lower() != role:
            continue
        for part in sequence_values(message.get("content")):
            if isinstance(part, Mapping):
                if str(part.get("type") or "").strip().lower() != "text":
                    continue
                text = text_value(part.get("text"))
            else:
                text = text_value(part)
            if text.strip():
                chunks.append(text.strip())
    return "\n".join(chunks)


def conversation_image_paths(value: Any) -> List[str]:
    paths = []
    for message in sequence_values(value):
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role") or "").strip().lower() not in {"user", "human"}:
            continue
        for part in sequence_values(message.get("content")):
            if not isinstance(part, Mapping):
                continue
            if str(part.get("type") or "").strip().lower() not in {"image", "image_url"}:
                continue
            path = part.get("image") or part.get("path") or part.get("image_url")
            if isinstance(path, Mapping):
                path = path.get("url")
            path_text = text_value(path).strip()
            if path_text and not path_text.startswith(("http://", "https://", "data:")):
                paths.append(path_text)
    return paths


def all_sources_missing_required_media(
    source_extraction: Mapping[str, Mapping[str, Any]],
    warnings: List[str],
) -> bool:
    if not source_extraction:
        return False
    if any(
        int(stats.get("record_count") or 0) <= 0
        or int(stats.get("converted_count") or 0) != 0
        or int(stats.get("skipped_count") or 0) != int(stats.get("record_count") or 0)
        for stats in source_extraction.values()
    ):
        return False
    extraction_failures = [
        warning for warning in warnings if warning.startswith("skipped ")
    ]
    return bool(extraction_failures) and all(
        "no input media could be extracted" in warning
        for warning in extraction_failures
    )


def scalar_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def text_value(value: Any) -> str:
    value = scalar_value(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        values = value.tolist()
        return text_value(values[0]) if values else ""
    if isinstance(value, (list, tuple)):
        return text_value(value[0]) if value else ""
    if value is None:
        return ""
    return str(value)


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return value.strip("_") or "default"


def json_safe_metadata(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [json_safe_metadata(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return {"bytes_length": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if isinstance(value, dict):
        return {str(key): json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_metadata(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def ingestion_repair_diagnostics(result: IngestionResult) -> Dict[str, Any]:
    validation = result.manifest.get("validation") or {}
    errors = [str(error) for error in validation.get("errors") or []]
    error_patterns = Counter(re.sub(r"^row \d+:\s*", "", error) for error in errors)
    warnings = [str(warning) for warning in result.manifest.get("warnings") or []]
    return {
        "decision": result.decision,
        "reason": str(result.manifest.get("reason") or ""),
        "converted_sample_count": int(validation.get("sample_count") or 0),
        "validation_errors": [
            {"message": message, "reported_count": count}
            for message, count in error_patterns.most_common(10)
        ],
        "mapping_or_extraction_warnings": warnings[:20],
    }


def build_ingestion_repair_prompt(
    original_prompt: str,
    previous_response: str,
    diagnostics: Dict[str, Any],
) -> str:
    return (
        original_prompt
        + "\n\nINGESTION MAPPING REPAIR\n"
        + "The deterministic executor rejected the first declarative mapping. Correct the mapping once. "
        + "Do not write Python, invent fields, change task semantics, expose evaluation targets as model "
        + "inputs, or claim an official metric without supplied evidence. Use choices.mode=fields when "
        + "options occupy separate columns. Return only a complete replacement JSON mapping.\n"
        + "Previous response:\n"
        + previous_response[:16000]
        + "\nDeterministic diagnostics:\n"
        + json.dumps(diagnostics, ensure_ascii=False, indent=2)
    )


def parse_json_response(response: str) -> Dict[str, Any]:
    text = str(response or "").strip()
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE))
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("no JSON object found")


def write_ingestion_outputs(result: IngestionResult, output_root: str | Path) -> Dict[str, str]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "unified_data": root / f"{result.benchmark_name}.unified.jsonl",
        "package": root / "benchmark.yaml",
        "mapping": root / f"{result.benchmark_name}.ingestion_mapping.yaml",
        "manifest": root / f"{result.benchmark_name}.ingestion_manifest.json",
        "prompt": root / f"{result.benchmark_name}.ingestion_prompt.txt",
        "raw_response": root / f"{result.benchmark_name}.ingestion_response.txt",
        "attempts": root / f"{result.benchmark_name}.ingestion_attempts.json",
    }
    with paths["unified_data"].open("w", encoding="utf-8") as handle:
        for item in result.items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_package_manifest(
        paths["package"],
        benchmark_name=result.benchmark_name,
        data_file=paths["unified_data"].name,
        benchmark_root=".",
    )
    paths["mapping"].write_text(
        yaml.safe_dump(result.mapping, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    paths["manifest"].write_text(
        json.dumps(result.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["prompt"].write_text(result.prompt, encoding="utf-8")
    paths["raw_response"].write_text(result.raw_response, encoding="utf-8")
    paths["attempts"].write_text(
        json.dumps(result.attempts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {name: str(path) for name, path in paths.items()}
