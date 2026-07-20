from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

from ..benchmark.schema import canonicalize_unified_item
from ..protocols import create_protocol
from ..reporting import ProgressReporter, concise_output_enabled, display_task_name
from .results import classify_sample_detail, summarize_details


RETRYABLE_GENERATION_ERROR_TYPES = {
    "ChunkedEncodingError",
    "ConnectionError",
    "ConnectTimeout",
    "ProxyError",
    "ReadTimeout",
    "Timeout",
    "timeout",
}
RETRYABLE_GENERATION_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def is_retryable_generation_error(error_type: str, error_message: str = "") -> bool:
    if error_type in RETRYABLE_GENERATION_ERROR_TYPES:
        return True
    if error_type != "api_http_error":
        return False
    match = re.search(r"\bHTTP\s+(\d{3})\b", str(error_message or ""), flags=re.IGNORECASE)
    return bool(match and int(match.group(1)) in RETRYABLE_GENERATION_HTTP_STATUS_CODES)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_frozen_runtime_path(value: str, config_path: str) -> str:
    """Resolve a versioned artifact path relative to its build directory."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    config = Path(config_path).expanduser().resolve()
    artifact_root = config.parent.parent if config.parent.name == "configs" else config.parent
    return str((artifact_root / path).resolve())


def load_protocol_pool(protocol_spec_dir: str) -> Dict[str, Dict[str, Any]]:
    pool: Dict[str, Dict[str, Any]] = {}
    root = Path(protocol_spec_dir)
    if not root.exists():
        return pool
    for path in sorted(root.glob("*.yaml")):
        cfg = load_config(str(path))
        if not cfg:
            continue
        name = str(cfg.get("name") or path.stem)
        pool[name] = cfg
    return pool


def load_benchmark_runtime(args: argparse.Namespace) -> tuple[Dict[str, Any], Dict[str, Any], str]:
    """Return benchmark config, task config mapping, and benchmark name."""
    legacy_config = str(getattr(args, "config", "") or "").strip()
    if legacy_config:
        data_file = str(getattr(args, "data_file", "") or "").strip()
        if not data_file:
            raise ValueError("--data-file is required when using legacy --config")
        cfg = load_config(legacy_config)
        benchmark_cfg = {
            "benchmark": Path(legacy_config).stem,
            "data_file": data_file,
            "benchmark_root": str(getattr(args, "benchmark_root", "") or "benchmarks"),
        }
        return benchmark_cfg, cfg, str(benchmark_cfg["benchmark"])

    benchmark_config = str(getattr(args, "benchmark_config", "") or "").strip()
    if not benchmark_config:
        raise ValueError("Please pass --benchmark-config <yaml>; use `provise build` to construct one.")
    config_path = Path(benchmark_config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Benchmark config does not exist: {config_path}")

    args.benchmark_config = str(config_path)
    benchmark_cfg = load_config(str(config_path))
    tasks_cfg = benchmark_cfg.get("tasks", {})
    benchmark_name = str(benchmark_cfg.get("benchmark") or Path(config_path).stem)
    return benchmark_cfg, tasks_cfg, benchmark_name


def load_data(path: str, task: str, limit: int | None, task_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    task_field = task_cfg.get("task_field", "task")
    task_value = task_cfg.get("task_value", task)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = canonicalize_unified_item(json.loads(line))
            if not task_field or item.get(task_field) == task_value:
                items.append(item)
                if limit and len(items) >= limit:
                    break
    return items


def allocate_task_sample_limits(
    path: str,
    tasks: List[str],
    tasks_cfg: Mapping[str, Mapping[str, Any]],
    *,
    total_budget: int,
    per_task_limit: int | None,
) -> Dict[str, int | None]:
    """Balance a total evaluation budget without silently dropping small tasks."""

    if total_budget <= 0:
        return {task: per_task_limit for task in tasks}

    capacities = {
        task: len(load_data(path, task, per_task_limit, dict(tasks_cfg[task])))
        for task in tasks
    }
    limits = {task: 0 for task in tasks}
    remaining = min(total_budget, sum(capacities.values()))
    while remaining:
        progressed = False
        for task in tasks:
            if limits[task] >= capacities[task]:
                continue
            limits[task] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return limits


def resolve_task_config(
    task: str,
    task_cfg: Dict[str, Any],
    protocol_pool: Dict[str, Dict[str, Any]],
    protocol_override: str = "",
) -> tuple[str, Dict[str, Any], str]:
    proto_name = protocol_override or task_cfg["protocol"]
    proto_def = protocol_pool.get(proto_name, {})

    prompt_variant = task_cfg.get("prompt_variant", "default")
    prompt_entry = (proto_def.get("prompts") or {}).get(prompt_variant, {})
    has_inline_prompt = bool(str(task_cfg.get("prompt") or "").strip())
    if prompt_variant and proto_def and not prompt_entry and not has_inline_prompt:
        raise ValueError(
            f"Task {task} requests protocol={proto_name} prompt_variant={prompt_variant}, "
            f"but that variant is not defined in protocol pool."
        )

    proto_cfg: Dict[str, Any] = {}
    proto_cfg = deep_merge(proto_cfg, proto_def.get("default_config") or {})
    parser_ops = proto_def.get("parser_ops") or {}
    if isinstance(parser_ops, dict) and parser_ops.get("pipeline"):
        proto_cfg["parser_pipeline"] = parser_ops["pipeline"]
    proto_cfg = deep_merge(proto_cfg, prompt_entry.get("config") or {})

    input_cfg = dict(task_cfg.get("input") or {})
    if input_cfg:
        if "mode" in input_cfg and "input_mode" not in input_cfg:
            input_cfg["input_mode"] = input_cfg.pop("mode")
        proto_cfg = deep_merge(proto_cfg, input_cfg)
    proto_cfg = deep_merge(proto_cfg, task_cfg.get("protocol_config") or {})
    for key in ("metric", "metric_config"):
        if task_cfg.get(key) is not None:
            proto_cfg[key] = task_cfg[key]

    prompt_template = str(task_cfg.get("prompt") or prompt_entry.get("template") or "").strip()
    if not prompt_template:
        raise ValueError(f"Missing prompt template for task={task}, protocol={proto_name}")
    return proto_name, proto_cfg, prompt_template


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_model(model_key: str):
    if model_key.startswith("mock-"):
        return MockProtocolModel(model_key)
    from ..models.generative import create_model

    model = create_model(model_key)
    model.load_model()
    return model


def maybe_generate(
    model,
    input_paths: List[str],
    prompt: str,
    save_path: str,
    reuse: bool,
    *,
    reuse_only: bool = False,
) -> Dict[str, Any]:
    if reuse and os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return {
            "generation_attempted": False,
            "generation_success": True,
            "generation_mode": "reused",
            "generation_error_type": "",
            "generation_error_message": "",
        }

    if reuse_only:
        return {
            "generation_attempted": False,
            "generation_success": False,
            "generation_mode": "failed",
            "generation_error_type": "reused_output_missing",
            "generation_error_message": (
                "A second parser pass was requested, but the first pass did not produce an image."
            ),
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if hasattr(model, "clear_last_error"):
        model.clear_last_error()
    if len(input_paths) == 1:
        ok = bool(model.generate(input_paths[0], prompt, save_path))
    elif hasattr(model, "generate_multi"):
        ok = bool(model.generate_multi(input_paths, prompt, save_path))
    else:
        ok = bool(model.generate(input_paths[0], prompt, save_path))

    exists = os.path.exists(save_path) and os.path.getsize(save_path) > 0
    error_type = str(getattr(model, "last_error_type", "") or "").strip()
    error_message = str(getattr(model, "last_error_message", "") or "").strip()
    if ok and not exists:
        ok = False
        error_type = error_type or "output_missing_or_empty"
        error_message = error_message or "Model reported success but output file is missing or empty."

    return {
        "generation_attempted": True,
        "generation_success": ok,
        "generation_mode": "generated" if ok else "failed",
        "generation_error_type": "" if ok else error_type,
        "generation_error_message": "" if ok else error_message,
    }


def _base_detail(
    *,
    item: Dict[str, Any],
    task: str,
    protocol_name: str,
    input_paths: List[str],
    save_path: str,
    prompt: str = "",
) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "task": task,
        "capability": item.get("capability", ""),
        "protocol": protocol_name,
        "input_paths": input_paths,
        "generated_image": save_path,
        "prompt": prompt,
        "ground_truth": item.get("answer"),
        "prediction": None,
        "input_available": True,
        "missing_input": False,
        "generation_attempted": False,
        "generation_success": False,
        "generation_mode": "",
        "generation_error_type": "",
        "generation_error_message": "",
        "parse_attempted": False,
        "parse_success": False,
        "parse_error_type": "",
        "parse_error_message": "",
        "score_computed": False,
        "score_error_type": "",
        "score_error_message": "",
        "is_correct": False,
        "score": 0.0,
        "sample_status": "",
        "failure_category": "",
    }


def _detail_with_status(detail: Dict[str, Any]) -> Dict[str, Any]:
    return classify_sample_detail(detail)


def _overall_summary_from_task_results(task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_details: List[Dict[str, Any]] = []
    capability_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in task_results:
        for detail in result.get("detailed_results", []):
            all_details.append(detail)
            capability = str(detail.get("capability") or result.get("capability") or "")
            if capability:
                capability_groups[capability].append(detail)

    overall = summarize_details(all_details)
    capabilities = []
    for capability in sorted(capability_groups):
        capabilities.append({"capability": capability, **summarize_details(capability_groups[capability])})
    return {"overall": overall, "capabilities": capabilities}


def run_task(
    task: str,
    task_cfg: Dict[str, Any],
    protocol_pool: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    model,
    output_root: Path,
    *,
    sample_limit: int | None = None,
) -> Dict[str, Any]:
    reporter = getattr(args, "reporter", None)
    proto_name, proto_cfg, prompt_template = resolve_task_config(
        task, task_cfg, protocol_pool, protocol_override=args.protocol
    )
    protocol = create_protocol(proto_name, proto_cfg)
    effective_limit = args.limit if sample_limit is None else sample_limit
    data = load_data(args.data_file, task, effective_limit, task_cfg)
    out_dir = output_root / task
    out_dir.mkdir(parents=True, exist_ok=True)
    concise = concise_output_enabled()

    if not concise:
        print("")
        print("=" * 72)
        print(f"Task: {task} | protocol={proto_name} | samples={len(data)}")
        print("=" * 72)
    if args.print_prompt:
        print(prompt_template)

    details = []
    for idx, item in enumerate(data, 1):
        sample_id = safe_name(str(item.get("id", f"{task}_{idx}")))
        prefix = f"[{idx}/{len(data)}]"
        input_paths = protocol.input_paths(item, args.benchmark_root)
        save_path = str(out_dir / f"{sample_id}_generated.png")
        detail = _base_detail(
            item=item,
            task=task,
            protocol_name=proto_name,
            input_paths=input_paths,
            save_path=save_path,
        )
        missing = [p for p in input_paths if not os.path.exists(p)]
        if missing:
            if not concise:
                print(f"{prefix} ERROR missing_input={missing[:2]}")
            detail["input_available"] = False
            detail["missing_input"] = True
            detail["generation_error_type"] = "missing_input"
            detail["generation_error_message"] = "; ".join(missing[:2])
            details.append(_detail_with_status(detail))
            continue

        prompt = protocol.render_prompt(prompt_template, item, args.benchmark_root)
        detail["prompt"] = prompt
        max_generation_retries = max(0, int(getattr(args, "generation_retries", 0) or 0))
        generation_retry_backoff = max(
            0.0, float(getattr(args, "generation_retry_backoff", 0.0) or 0.0)
        )
        generation_attempt = 0
        while True:
            reusing_output = (
                generation_attempt == 0
                and not args.no_reuse
                and os.path.exists(save_path)
                and os.path.getsize(save_path) > 0
            )
            operation_label = str(
                getattr(args, "operation_label", "Generating image") or "Generating image"
            )
            operation = (
                f"Reusing image {idx}/{len(data)}"
                if reusing_output
                else f"{operation_label} {idx}/{len(data)}"
            )
            if generation_attempt:
                operation += f" retry {generation_attempt}/{max_generation_retries}"
            if reporter is not None:
                with reporter.waiting(
                    operation,
                    event="image_reuse" if reusing_output else "image_generation",
                    task=task,
                    sample_id=str(item.get("id") or sample_id),
                    model=args.model,
                ):
                    gen = maybe_generate(
                        model,
                        input_paths,
                        prompt,
                        save_path,
                        reuse=reusing_output,
                        reuse_only=bool(getattr(args, "reuse_only", False)),
                    )
            else:
                gen = maybe_generate(
                    model,
                    input_paths,
                    prompt,
                    save_path,
                    reuse=reusing_output,
                    reuse_only=bool(getattr(args, "reuse_only", False)),
                )
            if gen["generation_success"]:
                break
            error_type = str(gen.get("generation_error_type") or "generation_failed")
            error_message = str(gen.get("generation_error_message") or "")
            if (
                generation_attempt >= max_generation_retries
                or not is_retryable_generation_error(error_type, error_message)
            ):
                break
            generation_attempt += 1
            delay = generation_retry_backoff * (2 ** (generation_attempt - 1))
            if reporter is not None:
                reporter.emit(
                    f"Retrying generation after {error_type} "
                    f"in {delay:g}s ({generation_attempt}/{max_generation_retries})",
                    event="image_generation_retry_started",
                    task=task,
                    sample_id=str(item.get("id") or sample_id),
                    error_type=error_type,
                    retry=generation_attempt,
                    max_retries=max_generation_retries,
                )
            time.sleep(delay)
        detail["generation_retry_count"] = generation_attempt
        detail.update(gen)
        if not gen["generation_success"]:
            error_type = str(gen.get("generation_error_type") or "generation_failed")
            if reporter is not None and error_type != "reused_output_missing":
                reporter.emit(
                    f"Image generation failed: {error_type}",
                    event="image_generation_result",
                    status="failed",
                    task=task,
                    sample_id=str(item.get("id") or sample_id),
                    error_type=error_type,
                    error_message=str(gen.get("generation_error_message") or ""),
                )
            if not concise:
                print(f"{prefix} ERROR generation_failed")
            details.append(_detail_with_status(detail))
            continue

        try:
            if reporter is not None:
                with reporter.waiting(
                    "Parsing generated image",
                    event="image_parse",
                    task=task,
                    sample_id=str(item.get("id") or sample_id),
                ):
                    parsed = protocol.parse(save_path, item, args.benchmark_root)
            else:
                parsed = protocol.parse(save_path, item, args.benchmark_root)
            detail["parse_attempted"] = True
            detail["parse_success"] = bool(parsed.parse_success)
            detail["prediction"] = parsed.prediction
            detail.update(parsed.extra)
            if not parsed.parse_success:
                detail["parse_error_type"] = str(parsed.extra.get("error_type") or "parse_failed")
                detail["parse_error_message"] = str(parsed.extra.get("error") or "")
                if reporter is not None:
                    reporter.emit(
                        f"Parser rejected generated image: {detail['parse_error_type']}",
                        event="image_parse_result",
                        status="failed",
                        task=task,
                        sample_id=str(item.get("id") or sample_id),
                        error_type=detail["parse_error_type"],
                        error_message=detail["parse_error_message"],
                    )
        except Exception as exc:
            detail["parse_attempted"] = True
            detail["parse_success"] = False
            detail["parse_error_type"] = type(exc).__name__
            detail["parse_error_message"] = str(exc)
            details.append(_detail_with_status(detail))
            if not concise:
                print(f"{prefix} ERROR parse_exception={type(exc).__name__}")
            continue

        try:
            scored = protocol.score(parsed, item, args.benchmark_root)
            detail["score_computed"] = True
            detail["is_correct"] = bool(scored.is_correct)
            detail["score"] = float(scored.score)
            detail.update(scored.extra)
            if detail.get("metric_unverified"):
                detail["score_computed"] = False
                detail["is_correct"] = False
        except Exception as exc:
            detail["score_error_type"] = type(exc).__name__
            detail["score_error_message"] = str(exc)
            details.append(_detail_with_status(detail))
            if not concise:
                print(f"{prefix} ERROR score_exception={type(exc).__name__}")
            continue

        detail = _detail_with_status(detail)
        if detail.get("metric_unverified"):
            mark = "UNSCORED"
        else:
            mark = "PASS" if scored.is_correct else "FAIL"
        pred = detail.get("prediction")
        gt = detail.get("ground_truth", item.get("answer"))
        if not concise:
            if mark == "UNSCORED":
                print(f"{prefix} {mark} pred={pred} parse={parsed.parse_success} metric=unverified")
            else:
                print(
                    f"{prefix} {mark} pred={pred} gt={gt} score={scored.score:.3f} "
                    f"parse={parsed.parse_success}"
                )
        if reporter is not None:
            reporter.emit(
                f"Sample complete: parse={parsed.parse_success} score={scored.score:.3f}",
                event="sample_completed",
                status="completed",
                task=task,
                sample_id=str(item.get("id") or sample_id),
                parse_success=bool(parsed.parse_success),
                score=float(scored.score),
            )
        details.append(detail)

    results = protocol.aggregate(details, task)
    capability = ""
    if data:
        capability = str(data[0].get("capability", ""))
    results["capability"] = capability
    results["model"] = args.model
    results["data_file"] = args.data_file
    results["benchmark_root"] = args.benchmark_root
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    if not concise:
        print(
            f"\n  => {task}: score={results.get('mean_score', 0) * 100:.1f}% "
            f"valid={results.get('valid_parse_rate', 0):.1f}% "
            f"generated={results.get('generated_rate', 0):.1f}%"
        )
    elif reporter is None:
        print(f"[Task: {display_task_name(task)}] {evaluation_result_message(results)}")
    return results


def run_protocol_eval(args: argparse.Namespace) -> int:
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ.setdefault("JOYAI_CUDA_DEVICE", "0")
    os.environ.setdefault("PROVISE_CLIP_DEVICE", "cpu")

    benchmark_cfg, tasks_cfg, benchmark_name = load_benchmark_runtime(args)
    if args.protocol_dir:
        args.protocol_spec_dir = args.protocol_dir
    protocol_pool = load_protocol_pool(args.protocol_spec_dir)

    args.data_file = args.data_file or str(benchmark_cfg.get("data_file", ""))
    args.benchmark_root = args.benchmark_root or str(benchmark_cfg.get("benchmark_root", ""))
    if (benchmark_cfg.get("protocol_build") or {}).get("frozen") is True:
        if args.data_file:
            args.data_file = resolve_frozen_runtime_path(
                args.data_file,
                args.benchmark_config,
            )
        if args.benchmark_root:
            args.benchmark_root = resolve_frozen_runtime_path(
                args.benchmark_root,
                args.benchmark_config,
            )
    if not args.data_file:
        raise ValueError("data_file is required, either in benchmark config or via --data-file")
    if not args.benchmark_root:
        raise ValueError("benchmark_root is required, either in benchmark config or via --benchmark-root")

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    if not tasks:
        tasks = list(tasks_cfg.keys())
    missing_cfg = [task for task in tasks if task not in tasks_cfg]
    if missing_cfg:
        raise ValueError(f"Missing task config(s): {missing_cfg}")
    blocked_tasks = formal_evaluation_blocked_tasks(tasks, tasks_cfg)
    if blocked_tasks:
        raise ValueError(
            "Formal evaluation is blocked because the benchmark metric is unverified for "
            f"task(s): {blocked_tasks}. Use the Agentic smoke workflow to validate protocols."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output or f"outputs/{benchmark_name}/{args.model}/{timestamp}")
    output_root.mkdir(parents=True, exist_ok=True)
    progress_events = str(getattr(args, "progress_events", "") or "").strip()
    args.reporter = (
        ProgressReporter(
            progress_events,
            heartbeat_seconds=float(getattr(args, "heartbeat_seconds", 1.0) or 1.0),
        )
        if progress_events
        else None
    )
    args.operation_label = "Generating evaluation image"
    if not concise_output_enabled():
        print("=" * 72)
        print("ProVisE Protocol Evaluation")
        print(f"benchmark:      {benchmark_name}")
        print(f"model:          {args.model}")
        print(f"tasks:          {tasks}")
        print(f"data_file:      {args.data_file}")
        print(f"benchmark_root: {args.benchmark_root}")
        print(f"protocol_specs: {args.protocol_spec_dir}")
        print(f"output:         {output_root}")
        if args.gpu:
            print(
                f"gpu:            {args.gpu} "
                f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})"
            )
        print("=" * 72)

    model = None if bool(getattr(args, "reuse_only", False)) else ensure_model(args.model)
    task_limits = allocate_task_sample_limits(
        args.data_file,
        tasks,
        tasks_cfg,
        total_budget=max(0, int(getattr(args, "max_samples", 0) or 0)),
        per_task_limit=args.limit,
    )
    evaluated_tasks = [task for task in tasks if task_limits[task] != 0]
    summaries = []
    for task in evaluated_tasks:
        result = run_task(
            task,
            tasks_cfg[task],
            protocol_pool,
            args,
            model,
            output_root,
            sample_limit=task_limits[task],
        )
        summaries.append(result)
        if args.reporter is not None:
            args.reporter.emit(
                evaluation_result_message(result),
                event="evaluation_task_completed",
                status="completed",
                task=task,
                score=float(result.get("mean_score", 0.0)),
                valid_parse_rate=float(result.get("valid_parse_rate", 0.0)),
                generated_rate=float(result.get("generated_rate", 0.0)),
                failure_category_counts=result.get("failure_category_counts", {}),
            )

    summary = {
        "model": args.model,
        "benchmark": benchmark_name,
        "tasks": evaluated_tasks,
        "requested_tasks": tasks,
        "sample_budget": max(0, int(getattr(args, "max_samples", 0) or 0)),
        "task_sample_limits": task_limits,
        "output": str(output_root),
        "summaries": [
            {
                "task": r.get("task"),
                "protocol": r.get("protocol"),
                "total_samples": r.get("total_samples", 0),
                "valid_parse_rate": r.get("valid_parse_rate", 0),
                "generated_rate": r.get("generated_rate", 0),
                "invalid_output_rate": r.get("invalid_output_rate", 0),
                "correct_among_valid": r.get("correct_among_valid", 0),
                "accuracy": r.get("accuracy", 0),
                "mean_score": r.get("mean_score", 0),
                "mean_dfd": r.get("mean_dfd"),
                "mean_precision": r.get("mean_precision"),
                "status_counts": r.get("status_counts", {}),
                "failure_category_counts": r.get("failure_category_counts", {}),
            }
            for r in summaries
        ],
    }
    summary.update(_overall_summary_from_task_results(summaries))
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if not concise_output_enabled():
        print("")
        print(f"Summary saved to: {output_root / 'summary.json'}")
    return 0


def formal_evaluation_blocked_tasks(
    tasks: List[str], tasks_cfg: Mapping[str, Mapping[str, Any]]
) -> List[str]:
    return [task for task in tasks if tasks_cfg.get(task, {}).get("formal_evaluation") is False]


def evaluation_result_message(results: Mapping[str, Any]) -> str:
    message = (
        f"Evaluation: score={float(results.get('mean_score', 0.0)) * 100:.1f}% "
        f"valid={float(results.get('valid_parse_rate', 0.0)):.1f}% "
        f"generated={float(results.get('generated_rate', 0.0)):.1f}%"
    )
    failures = results.get("failure_category_counts") or {}
    if failures:
        reason = max(failures, key=lambda name: int(failures[name]))
        message += f" reason={reason}"
    return message


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:180]


class MockProtocolModel:
    """Tiny deterministic model for smoke-testing parser/runner plumbing."""

    def __init__(self, model_key: str):
        self.model_key = model_key
        self.model_name = model_key

    def load_model(self):
        return None

    def generate(self, image_path: str, prompt: str, save_path: str) -> bool:
        return self.generate_multi([image_path], prompt, save_path)

    def generate_multi(self, image_paths: List[str], prompt: str, save_path: str) -> bool:
        from PIL import Image, ImageDraw

        source = Image.open(image_paths[0]).convert("RGB")
        image = source.copy()
        draw = ImageDraw.Draw(image)
        width, height = image.size
        size = max(16, min(width, height) // 8)
        margin = max(8, min(width, height) // 32)

        if self.model_key == "mock-label-a":
            image = Image.new("RGB", source.size, "white")
            draw = ImageDraw.Draw(image)
            if self._uses_bottom_strip(prompt):
                slots = self._prompt_slot_count(prompt)
                strip_h = max(1, int(height * 0.18))
                cell_w = max(1, width // max(1, slots))
                marker = max(16, min(cell_w, strip_h) // 2)
                x1 = max(0, (cell_w - marker) // 2)
                y1 = height - strip_h + max(0, (strip_h - marker) // 2)
                box = (x1, y1, x1 + marker, y1 + marker)
            else:
                box = (margin, margin, margin + size, margin + size)
            draw.rectangle(box, fill=(255, 0, 255))
        elif self.model_key == "mock-copy":
            pass
        else:
            raise ValueError(f"Unknown mock model: {self.model_key}")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path)
        return True

    def _uses_bottom_strip(self, prompt: str) -> bool:
        prompt_l = prompt.lower()
        return "bottom code strip" in prompt_l or "bottom edge" in prompt_l

    def _prompt_slot_count(self, prompt: str) -> int:
        match = re.search(r"into\s+(\d+)\s+equal\s+slots", prompt, flags=re.IGNORECASE)
        if match:
            return max(1, int(match.group(1)))
        return 1
