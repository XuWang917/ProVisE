from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Mapping

from ..evaluation.metrics import normalize_metric_name


def infer_answer_schema(item: Mapping[str, Any]) -> str:
    choices = item.get("choices") or item.get("options") or []
    if choices:
        texts = []
        for choice in choices:
            text = choice.get("text") if isinstance(choice, Mapping) else choice
            texts.append(
                re.sub(r"[^\w]+", " ", str(text or "").lower(), flags=re.UNICODE).strip()
            )
        boolean_terms = {
            "yes",
            "no",
            "true",
            "false",
            "correct",
            "incorrect",
            "possible",
            "impossible",
            "是",
            "否",
            "对",
            "错",
            "可以",
            "不可以",
        }
        if len(texts) == 2 and set(texts).issubset(boolean_terms):
            return "binary_boolean"
        return "choice_selection"

    answer_type = str(item.get("answer_type") or "").strip().lower()
    metric = normalize_metric_name((item.get("evaluation") or {}).get("metric"))
    if metric in {"point_in_mask", "point_distance"} or any(
        term in answer_type for term in ("point", "coordinate")
    ):
        return "localization"
    if metric == "bbox_iou" or any(term in answer_type for term in ("bbox", "box")):
        return "bounding_box"
    if metric in {"mask_iou", "mask_precision"} or any(
        term in answer_type for term in ("mask", "region", "segmentation")
    ):
        return "region"
    if metric in {
        "qspatial_ratio",
        "numeric_absolute_error",
        "numeric_relative_error",
        "mra",
        "angle_error",
    } or any(
        term in answer_type for term in ("number", "numeric", "scalar", "float", "integer")
    ):
        return "numeric"
    if metric == "dfd" or any(term in answer_type for term in ("path", "trajectory")):
        return "trajectory"
    return "structured_or_text"


def task_contract(item: Mapping[str, Any]) -> Dict[str, Any]:
    evaluation = dict(item.get("evaluation") or {})
    return {
        "answer_schema": infer_answer_schema(item),
        "metric": normalize_metric_name(evaluation.get("metric")),
        "metric_config": _json_safe(evaluation.get("metric_config") or {}),
    }


def partition_heterogeneous_tasks(
    items: Iterable[Dict[str, Any]],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Split task labels only when one label carries incompatible score contracts."""

    rows = list(items)
    grouped: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for item in rows:
        grouped[str(item.get("task") or "default")].append(item)

    aliases: dict[tuple[str, str], str] = {}
    partition_manifest = []
    for original_task, task_items in sorted(grouped.items()):
        contracts: dict[str, Dict[str, Any]] = {}
        counts: Counter[str] = Counter()
        for item in task_items:
            contract = task_contract(item)
            key = _contract_key(contract)
            contracts[key] = contract
            counts[key] += 1
        if len(contracts) <= 1:
            continue

        proposed = Counter(
            _partition_name(original_task, contract) for contract in contracts.values()
        )
        partitions = []
        for key in sorted(contracts):
            contract = contracts[key]
            name = _partition_name(original_task, contract)
            if proposed[name] > 1:
                name = f"{name}__{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}"
            aliases[(original_task, key)] = name
            partitions.append({"task": name, "sample_count": counts[key], **contract})
        partition_manifest.append(
            {
                "original_task": original_task,
                "reason": "heterogeneous_answer_or_metric_contract",
                "partition_count": len(partitions),
                "partitions": partitions,
            }
        )

    if not aliases:
        return rows, []

    partitioned = []
    for item in rows:
        original_task = str(item.get("task") or "default")
        contract = task_contract(item)
        alias = aliases.get((original_task, _contract_key(contract)))
        if alias is None:
            partitioned.append(item)
            continue
        converted = dict(item)
        converted["task"] = alias
        metadata = dict(converted.get("metadata") or {})
        metadata["original_task"] = original_task
        metadata["agentic_task_contract"] = contract
        converted["metadata"] = metadata
        partitioned.append(converted)
    return partitioned, partition_manifest


def _partition_name(original_task: str, contract: Mapping[str, Any]) -> str:
    original = _slug(original_task, "default")
    schema = _slug(contract.get("answer_schema"), "answer")
    metric = _slug(contract.get("metric"), "unverified")
    return f"{original}__{schema}__{metric}"


def _contract_key(contract: Mapping[str, Any]) -> str:
    return json.dumps(contract, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _slug(value: Any, default: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return text or default


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)
