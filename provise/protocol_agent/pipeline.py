#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from provise.evaluation.runner import ensure_model, load_protocol_pool, run_task
from provise.models.vlm import create_eval_vlm
from provise.paths import protocol_spec_dir, runtime_root
from provise.protocol_agent.builder import (
    AgenticProtocolBuildResult,
    AgenticProtocolBuilder,
    extract_task_row,
    group_items_by_task,
    join_task_artifacts,
    load_items,
    normalize_agent_payload,
    parse_agent_json_response,
    select_representative_items,
    write_build_outputs,
)
from provise.reporting import ProgressReporter, concise_output_enabled


PROJECT_ROOT = runtime_root()
load_dotenv(PROJECT_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build task-specific agentic visual protocols for a unified benchmark.")
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--input", required=True, help="Unified benchmark JSONL.")
    parser.add_argument("--tasks", default="", help="Optional comma-separated task subset for staged construction.")
    parser.add_argument("--data-file", default="", help="Path written into benchmark YAML. Defaults to --input.")
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--benchmark-config-dir", default="outputs/protocol_build/configs")
    parser.add_argument("--generated-protocol-dir", default="outputs/protocol_build/generated")
    parser.add_argument("--protocol-spec-dir", default=str(protocol_spec_dir()))
    parser.add_argument("--max-examples-per-task", type=int, default=3)
    parser.add_argument("--max-media-per-task", type=int, default=8)
    parser.add_argument(
        "--router-model",
        default=os.getenv("PROVISE_AGENT_MODEL", "gpt-5.4"),
        help="Defaults to PROVISE_AGENT_MODEL.",
    )
    parser.add_argument("--router-timeout", type=int, default=60)
    parser.add_argument("--router-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--agent-response-file",
        "--mock-agent-response-file",
        dest="agent_response_file",
        default="",
        help="Reuse a saved agent response artifact instead of calling the construction VLM.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument(
        "--smoke-model",
        default=os.getenv("PROVISE_PROTOCOL_MODEL", "gpt-image-2"),
    )
    parser.add_argument("--smoke-limit", type=int, default=3)
    parser.add_argument("--smoke-output", default="")
    parser.add_argument("--generation-retries", type=int, default=1)
    parser.add_argument("--generation-retry-backoff", type=float, default=2.0)
    parser.add_argument(
        "--reuse-smoke-images",
        action="store_true",
        help="Reuse existing smoke images. Use only with the same saved protocol response.",
    )
    parser.add_argument(
        "--mock-parse-response",
        default="",
        help="Testing helper: inject a fixed agentic parser JSON response during smoke validation only.",
    )
    parser.add_argument("--min-parse-success-rate", type=float, default=66.0)
    parser.add_argument("--min-generation-rate", type=float, default=66.0)
    parser.add_argument("--min-parser-agreement-rate", type=float, default=66.0)
    parser.add_argument("--min-spatial-evidence-rate", type=float, default=66.0)
    parser.add_argument("--progress-events", default="")
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=1.0)
    parser.add_argument("--max-revisions", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reporter = ProgressReporter(
        args.progress_events or None,
        enabled=not args.quiet_progress,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    config_path = Path(args.benchmark_config_dir) / f"{args.benchmark_name}.yaml"
    protocol_path = Path(args.generated_protocol_dir) / f"{args.benchmark_name}.agentic_protocols.yaml"
    manifest_path = Path(args.benchmark_config_dir) / f"{args.benchmark_name}.agentic_manifest.json"
    prompt_path = Path(args.benchmark_config_dir) / f"{args.benchmark_name}.agentic_prompt.txt"
    raw_response_path = Path(args.benchmark_config_dir) / f"{args.benchmark_name}.agentic_response.txt"

    if config_path.exists() and not args.force:
        print(f"Benchmark config already exists: {config_path}")
        print("Pass --force to rebuild it.")
        return 0

    raw_response = ""
    vlm = None
    if args.agent_response_file:
        reporter.emit(
            f"Loading saved protocol-agent response: {args.agent_response_file}",
            event="protocol_response_reused",
        )
        raw_response = Path(args.agent_response_file).read_text(encoding="utf-8")
    else:
        reporter.emit(
            f"Initializing protocol agent: {args.router_model}",
            event="protocol_agent_loading",
        )
        vlm = create_eval_vlm(
            timeout=args.router_timeout,
            max_tokens=args.router_max_tokens,
            model_name=args.router_model,
        )
        vlm.load_model()

    items = load_items(args.input)
    reporter.emit(
        f"Loaded {len(items)} unified sample(s)",
        event="unified_samples_loaded",
        status="completed",
        sample_count=len(items),
    )
    requested_tasks = {task.strip() for task in args.tasks.split(",") if task.strip()}
    if requested_tasks:
        available_tasks = {str(item.get("task") or "default") for item in items}
        unknown_tasks = sorted(requested_tasks - available_tasks)
        if unknown_tasks:
            raise ValueError(f"Unknown task(s) requested: {unknown_tasks}")
        items = [item for item in items if str(item.get("task") or "default") in requested_tasks]
    result, smoke, paths = build_tasks_sequentially(
        args,
        items,
        vlm=vlm,
        raw_response=raw_response,
        reporter=reporter,
        output_paths={
            "benchmark_config_path": config_path,
            "protocol_path": protocol_path,
            "manifest_path": manifest_path,
            "prompt_path": prompt_path,
            "raw_response_path": raw_response_path,
        },
    )

    if not concise_output_enabled():
        print("=" * 72)
        print("Agentic Protocol Builder")
        print(f"benchmark: {args.benchmark_name}")
        print(
            f"tasks:     {len(result.benchmark_config.get('tasks', {}))} active / "
            f"{result.manifest['task_count']} total"
        )
        print(f"decisions: {result.manifest.get('decision_counts', {})}")
        print("Outputs:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        if smoke:
            print("Smoke validation:")
            for task, row in smoke.get("tasks", {}).items():
                print(
                    f"  {task}: {row.get('status')} "
                    f"parse={row.get('valid_parse_rate', 0):.1f}%"
                )
    formal_tasks = [
        task
        for task, task_cfg in (result.benchmark_config.get("tasks") or {}).items()
        if task_cfg.get("formal_evaluation") is not False
    ]
    if formal_tasks and not concise_output_enabled():
        print(f"Evaluate with: provise evaluate --protocol {config_path}")
    elif result.benchmark_config.get("tasks") and not concise_output_enabled():
        print("Accepted protocols are smoke-only because their benchmark metric is unverified.")
    elif not result.benchmark_config.get("tasks") and not concise_output_enabled():
        print("No task passed protocol construction and smoke validation; evaluation config is intentionally empty.")
    return 0


REPAIRABLE_ROUTE_SOURCES = {
    "agent_missing",
    "contract_compiler",
    "fallback_validation",
    "framework_validation",
}


def build_tasks_sequentially(
    args: argparse.Namespace,
    items: list[dict],
    *,
    vlm,
    raw_response: str,
    reporter: ProgressReporter,
    output_paths: dict[str, Path],
) -> tuple[AgenticProtocolBuildResult, dict, dict[str, str]]:
    groups = group_items_by_task(items)
    task_names = sorted(groups)
    completed: dict[str, AgenticProtocolBuildResult] = {}
    prompts: dict[str, str] = {}
    responses: dict[str, str] = {}
    smoke_runtime: dict[str, object] = {}

    result, smoke = combine_task_results(
        args,
        items,
        task_names,
        completed,
        prompts,
        responses,
    )
    paths = write_build_outputs(result, **output_paths)

    for task_index, task in enumerate(task_names, 1):
        reporter.emit(
            "Protocol workflow started",
            event="task_workflow_started",
            task=task,
            task_index=task_index,
            task_count=len(task_names),
            sample_count=len(groups[task]),
        )
        task_result, task_prompt, task_response = build_one_task(
            args,
            task,
            groups[task],
            vlm=vlm,
            raw_response=raw_response,
            reporter=reporter,
            smoke_runtime=smoke_runtime,
        )
        completed[task] = task_result
        prompts[task] = task_prompt
        responses[task] = task_response
        result, smoke = combine_task_results(
            args,
            items,
            task_names,
            completed,
            prompts,
            responses,
        )
        paths = write_build_outputs(result, **output_paths)

        route = next(
            (row for row in task_result.manifest.get("route_rows", []) if row.get("task") == task),
            {},
        )
        active = task in (task_result.benchmark_config.get("tasks") or {})
        if active:
            decision = str(route.get("decision") or "ready")
            reporter.emit(
                f"Ready via {display_route(decision, route.get('build_mode'))}",
                event="task_workflow_completed",
                status="completed",
                task=task,
                task_index=task_index,
                task_count=len(task_names),
                decision=decision,
                outcome="ready",
            )
        elif route.get("decision") == "deferred":
            reporter.emit(
                "Deferred because the generation service failed",
                event="task_workflow_completed",
                status="stopped",
                task=task,
                task_index=task_index,
                task_count=len(task_names),
                decision="deferred",
                outcome="deferred",
            )
        else:
            reporter.emit(
                "Unresolved after protocol construction and fallback",
                event="task_workflow_completed",
                status="failed",
                task=task,
                task_index=task_index,
                task_count=len(task_names),
                decision=str(route.get("decision") or "unresolved"),
                outcome="unresolved",
            )

    return result, smoke, paths


def build_one_task(
    args: argparse.Namespace,
    task: str,
    task_items: list[dict],
    *,
    vlm,
    raw_response: str,
    reporter: ProgressReporter,
    smoke_runtime: dict[str, object],
) -> tuple[AgenticProtocolBuildResult, str, str]:
    builder = AgenticProtocolBuilder(
        task_items,
        benchmark_name=args.benchmark_name,
        data_file=args.data_file or args.input,
        benchmark_root=args.benchmark_root,
        max_examples_per_task=args.max_examples_per_task,
        max_media_per_task=args.max_media_per_task,
        protocol_spec_dir=args.protocol_spec_dir,
        reporter=reporter,
    )
    result = builder.build(vlm=vlm, raw_response=raw_response)
    response_rows, response_text = response_rows_from_artifact(result.raw_response, [task])
    revision_counts: dict[str, int] = {}
    revision_history: list[dict] = []

    route = task_route(result, task)
    if (
        vlm is not None
        and args.max_revisions > 0
        and route
        and not route.get("active")
        and route.get("source") in REPAIRABLE_ROUTE_SOURCES
    ):
        try:
            response = builder.revise_task(
                task=task,
                vlm=vlm,
                previous_response=response_text.get(task, result.raw_response),
                diagnostics={"phase": "compile", "route": route},
            )
            parsed = extract_task_row(parse_agent_json_response(response), task)
            if parsed is None:
                raise ValueError("revision omitted task")
            response_rows[task] = parsed
            response_text[task] = response
            revision_counts[task] = 1
            revision_history.append(
                {"task": task, "phase": "compile", "previous_diagnostics": route}
            )
            result = builder.build(
                raw_response=combined_agent_response(args.benchmark_name, response_rows)
            )
        except Exception as exc:
            reporter.emit(
                f"Compile revision failed: {type(exc).__name__}: {exc}",
                event="protocol_revision_failed",
                status="failed",
                task=task,
            )

    activate_compile_fallbacks(builder, result, reporter=reporter)
    result.manifest["revision_counts"] = dict(sorted(revision_counts.items()))
    result.manifest["revision_history"] = revision_history
    smoke = {}

    if not args.no_smoke:
        smoke = run_smoke_validation(
            args,
            result.benchmark_config,
            reporter=reporter,
            smoke_runtime=smoke_runtime,
        )
        smoke_row = (smoke.get("tasks") or {}).get(task)
        can_revise_smoke = (
            vlm is not None
            and args.max_revisions > 0
            and smoke_row
            and smoke_row.get("status") != "passed"
            and revision_counts.get(task, 0) < args.max_revisions
            and task in response_rows
        )
        if can_revise_smoke and smoke_failure_is_external(smoke_row):
            reporter.emit(
                "Skipping protocol revision because failure is generation/API-only",
                event="protocol_revision_skipped_external_failure",
                status="stopped",
                task=task,
            )
        elif can_revise_smoke:
            diagnostics, generated_paths = smoke_revision_context(smoke_row)
            try:
                response = builder.revise_task(
                    task=task,
                    vlm=vlm,
                    previous_response=response_text.get(task, json.dumps(response_rows[task])),
                    diagnostics={"phase": "smoke", **diagnostics},
                    generated_paths=generated_paths,
                )
                parsed = extract_task_row(parse_agent_json_response(response), task)
                if parsed is None:
                    raise ValueError("revision omitted task")
                response_rows[task] = parsed
                response_text[task] = response
                revision_counts[task] = revision_counts.get(task, 0) + 1
                revision_history.append(
                    {"task": task, "phase": "smoke", "previous_diagnostics": diagnostics}
                )
                result = builder.build(
                    raw_response=combined_agent_response(args.benchmark_name, response_rows)
                )
                activate_compile_fallbacks(builder, result, reporter=reporter)
                result.manifest["revision_counts"] = dict(sorted(revision_counts.items()))
                result.manifest["revision_history"] = revision_history
                if task in (result.benchmark_config.get("tasks") or {}):
                    revision_args = argparse.Namespace(**vars(args))
                    base_smoke_output = Path(
                        args.smoke_output
                        or f"outputs/agentic_smoke_{args.benchmark_name}_{args.smoke_model}"
                    )
                    revision_args.smoke_output = str(
                        base_smoke_output.parent / f"{base_smoke_output.name}_revision1"
                    )
                    revision_args.reuse_smoke_images = False
                    revision_args.smoke_phase = "revision"
                    revised_smoke = run_smoke_validation(
                        revision_args,
                        result.benchmark_config,
                        reporter=reporter,
                        smoke_runtime=smoke_runtime,
                    )
                    smoke.setdefault("tasks", {}).update(revised_smoke.get("tasks") or {})
                    smoke.setdefault("revision_runs", []).append(revised_smoke)
            except Exception as exc:
                reporter.emit(
                    f"Smoke revision failed: {type(exc).__name__}: {exc}",
                    event="protocol_revision_failed",
                    status="failed",
                    task=task,
                )

        run_automatic_fallback_smoke(
            args,
            builder,
            result,
            smoke,
            reporter=reporter,
            smoke_runtime=smoke_runtime,
        )
        result.manifest["revision_counts"] = dict(sorted(revision_counts.items()))
        result.manifest["revision_history"] = revision_history
        result.manifest["smoke_validation"] = smoke
        apply_smoke_gate(result.benchmark_config, result.manifest, smoke)

    prompt = task_artifact_body(result.prompt, task)
    response = response_text.get(task) or result.raw_response
    return result, prompt, response


def combine_task_results(
    args: argparse.Namespace,
    items: list[dict],
    task_names: list[str],
    completed: dict[str, AgenticProtocolBuildResult],
    prompts: dict[str, str],
    responses: dict[str, str],
) -> tuple[AgenticProtocolBuildResult, dict]:
    tasks_config: dict[str, dict] = {}
    protocols: list[dict] = []
    route_rows: list[dict] = []
    task_contexts: dict[str, dict] = {}
    task_contracts: dict[str, dict] = {}
    warnings: list[str] = []
    revision_counts: dict[str, int] = {}
    revision_history: list[dict] = []
    fallback_history: list[dict] = []
    fallback_tasks: set[str] = set()
    disabled_tasks: set[str] = set()
    deferred_tasks: set[str] = set()
    smoke = {
        "model": args.smoke_model,
        "limit": args.smoke_limit,
        "smoke_data_file": str(
            Path(
                args.smoke_output
                or f"outputs/agentic_smoke_{args.benchmark_name}_{args.smoke_model}"
            )
            / "smoke_samples.jsonl"
        ),
        "sample_ids": {},
        "tasks": {},
    }

    for task in task_names:
        task_result = completed.get(task)
        if task_result is None:
            continue
        tasks_config.update(task_result.benchmark_config.get("tasks") or {})
        protocols.extend(task_result.generated_protocols.get("protocols") or [])
        manifest = task_result.manifest
        route_rows.extend(manifest.get("route_rows") or [])
        task_contexts.update(manifest.get("task_contexts") or {})
        task_contracts.update(manifest.get("task_contracts") or {})
        warnings.extend(manifest.get("warnings") or [])
        revision_counts.update(manifest.get("revision_counts") or {})
        revision_history.extend(manifest.get("revision_history") or [])
        fallback_history.extend(manifest.get("fallback_history") or [])
        fallback_tasks.update(manifest.get("automatic_vlm_fallback_tasks") or [])
        disabled_tasks.update(manifest.get("disabled_tasks") or [])
        deferred_tasks.update(manifest.get("deferred_external_failure_tasks") or [])
        task_smoke = manifest.get("smoke_validation") or {}
        smoke["sample_ids"].update(task_smoke.get("sample_ids") or {})
        smoke["tasks"].update(task_smoke.get("tasks") or {})
        for key in ("revision_runs", "fallback_runs"):
            if task_smoke.get(key):
                smoke.setdefault(key, []).extend(task_smoke[key])

    active_sample_count = sum(
        int(route.get("sample_count", 0)) for route in route_rows if route.get("active")
    )
    manifest = {
        "benchmark": args.benchmark_name,
        "builder": "agentic_protocol_builder_v2",
        "total_samples": len(items),
        "task_count": len(task_names),
        "completed_task_count": len(completed),
        "active_task_count": len(tasks_config),
        "active_sample_count": active_sample_count,
        "decision_counts": dict(
            sorted(Counter(route.get("decision") for route in route_rows).items())
        ),
        "route_rows": route_rows,
        "task_contexts": task_contexts,
        "task_contracts": task_contracts,
        "revision_counts": dict(sorted(revision_counts.items())),
        "revision_history": revision_history,
        "automatic_vlm_fallback_tasks": sorted(fallback_tasks),
        "fallback_history": fallback_history,
        "disabled_tasks": sorted(disabled_tasks),
        "deferred_external_failure_tasks": sorted(deferred_tasks),
        "warnings": deduplicate(warnings),
    }
    if not args.no_smoke:
        manifest["smoke_validation"] = smoke
    result = AgenticProtocolBuildResult(
        benchmark_name=args.benchmark_name,
        benchmark_config={
            "benchmark": args.benchmark_name,
            "data_file": args.data_file or args.input,
            "benchmark_root": args.benchmark_root,
            "protocol_build": {
                "schema_version": "provise.protocol.v1",
                "frozen": True,
                "agent_model": getattr(args, "router_model", "")
                or os.getenv("PROVISE_AGENT_MODEL", "gpt-5.4"),
                "parser_model": os.getenv("PROVISE_PARSER_MODEL")
                or getattr(args, "router_model", "")
                or os.getenv("PROVISE_AGENT_MODEL", "gpt-5.4"),
                "validation_model": args.smoke_model,
            },
            "tasks": tasks_config,
        },
        generated_protocols={"protocols": protocols},
        manifest=manifest,
        prompt=join_task_artifacts(prompts),
        raw_response=json.dumps(
            {"benchmark": args.benchmark_name, "task_responses": responses},
            ensure_ascii=False,
            indent=2,
        ),
    )
    return result, smoke if not args.no_smoke else {}


def task_route(result: AgenticProtocolBuildResult, task: str) -> dict:
    return next(
        (row for row in result.manifest.get("route_rows", []) if row.get("task") == task),
        {},
    )


def task_artifact_body(value: str, task: str) -> str:
    marker = f"===== TASK: {task} =====\n"
    text = str(value or "")
    return text[len(marker) :] if text.startswith(marker) else text


def display_route(decision: str, build_mode: object = "") -> str:
    value = str(decision or "").lower()
    if value == "build":
        return {
            "recipe": "build (recipe)",
            "parser_ops": "build (Parser Ops)",
        }.get(str(build_mode or "").lower(), "build")
    return {
        "fallback": "VLM fallback",
        "reuse": "reuse",
    }.get(value, str(decision or "ready"))


def deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def response_rows_from_artifact(
    raw_response: str,
    tasks: list[str],
) -> tuple[dict[str, dict], dict[str, str]]:
    rows: dict[str, dict] = {}
    raw_by_task: dict[str, str] = {}
    try:
        payload = parse_agent_json_response(raw_response)
    except ValueError:
        return rows, raw_by_task
    task_responses = payload.get("task_responses") if isinstance(payload, dict) else None
    if isinstance(task_responses, dict):
        for task in tasks:
            raw = str(task_responses.get(task) or "")
            if not raw:
                continue
            try:
                row = extract_task_row(parse_agent_json_response(raw), task)
            except ValueError:
                row = None
            if row is not None:
                rows[task] = row
                raw_by_task[task] = raw
        return rows, raw_by_task
    normalized = normalize_agent_payload(payload)
    for row in normalized.get("tasks") or []:
        if not isinstance(row, dict):
            continue
        task = str(row.get("task") or "")
        if task in tasks:
            rows[task] = row
            raw_by_task[task] = json.dumps(row, ensure_ascii=False)
    return rows, raw_by_task


def combined_agent_response(benchmark_name: str, rows: dict[str, dict]) -> str:
    return json.dumps(
        {
            "benchmark": benchmark_name,
            "tasks": [rows[task] for task in sorted(rows)],
        },
        ensure_ascii=False,
        indent=2,
    )


HARD_UNSUPPORTED_SOURCES = {
    "framework_input_validation",
    "framework_metric_validation",
}


def activate_compile_fallbacks(
    builder: AgenticProtocolBuilder,
    result,
    *,
    reporter: ProgressReporter | None = None,
) -> list[str]:
    reporter = reporter or ProgressReporter(enabled=False)
    activated = []
    for route in list(result.manifest.get("route_rows") or []):
        if route.get("active"):
            continue
        source = str(route.get("source") or "")
        if source in HARD_UNSUPPORTED_SOURCES:
            continue
        task = str(route.get("task") or "")
        reason = (
            "No validated deterministic visual readout remained after protocol construction: "
            + str(route.get("reason") or source or "unknown construction failure")
        )
        ok, errors = builder.activate_automatic_vlm_fallback(
            result,
            task=task,
            origin=f"compile:{source or 'unknown'}",
            reason=reason,
        )
        if ok:
            activated.append(task)
            reporter.emit(
                "Activated task-level VLM fallback after protocol construction",
                event="automatic_vlm_fallback_activated",
                status="completed",
                task=task,
                origin=f"compile:{source or 'unknown'}",
            )
        else:
            reporter.emit(
                "Could not activate task-level VLM fallback: " + "; ".join(errors),
                event="automatic_vlm_fallback_rejected",
                status="failed",
                task=task,
                origin=f"compile:{source or 'unknown'}",
            )
    return activated


def run_automatic_fallback_smoke(
    args: argparse.Namespace,
    builder: AgenticProtocolBuilder,
    result,
    smoke: dict,
    *,
    reporter: ProgressReporter | None = None,
    smoke_runtime: dict[str, object] | None = None,
) -> list[str]:
    reporter = reporter or ProgressReporter(enabled=False)
    if int(getattr(args, "max_revisions", 0) or 0) <= 0:
        return []
    activated = []
    for task, smoke_row in list((smoke.get("tasks") or {}).items()):
        if smoke_row.get("status") == "passed" or smoke_failure_is_external(smoke_row):
            continue
        current = (result.benchmark_config.get("tasks") or {}).get(task) or {}
        if current.get("protocol") == "agentic_vlm_protocol":
            continue
        ok, errors = builder.activate_automatic_vlm_fallback(
            result,
            task=task,
            origin="smoke:deterministic_protocol_failed",
            reason=(
                "The deterministic or fixed-model visual readout failed mechanical smoke checks: "
                + ", ".join(smoke_row.get("failed_checks") or [smoke_row.get("status", "failed")])
            ),
        )
        if ok:
            activated.append(task)
        else:
            reporter.emit(
                "Could not activate smoke fallback: " + "; ".join(errors),
                event="automatic_vlm_fallback_rejected",
                status="failed",
                task=task,
                origin="smoke",
            )
    if not activated:
        return []

    fallback_config = json.loads(json.dumps(result.benchmark_config, ensure_ascii=False))
    fallback_config["tasks"] = {
        task: fallback_config["tasks"][task]
        for task in activated
        if task in fallback_config.get("tasks", {})
    }
    fallback_args = argparse.Namespace(**vars(args))
    base_output = Path(
        args.smoke_output
        or f"outputs/agentic_smoke_{args.benchmark_name}_{args.smoke_model}"
    )
    fallback_output = base_output.parent / f"{base_output.name}_vlm_fallback"
    fallback_args.smoke_output = str(fallback_output)
    fallback_args.smoke_phase = "fallback"
    reused_images = seed_fallback_smoke_images(
        base_output,
        fallback_output,
        activated,
        result.benchmark_config,
    )
    fallback_args.reuse_smoke_images = reused_images > 0
    if reused_images:
        reporter.emit(
            f"Reusing {reused_images} generated visual response(s) for VLM fallback readout",
            event="automatic_vlm_fallback_images_reused",
            status="completed",
            tasks=sorted(activated),
            image_count=reused_images,
        )
    reporter.emit(
        "Running task-level VLM fallback smoke for: " + ", ".join(sorted(activated)),
        event="automatic_vlm_fallback_smoke_started",
        tasks=sorted(activated),
    )
    smoke_kwargs = {"reporter": reporter}
    if smoke_runtime is not None:
        smoke_kwargs["smoke_runtime"] = smoke_runtime
    fallback_smoke = run_smoke_validation(
        fallback_args,
        fallback_config,
        **smoke_kwargs,
    )
    smoke.setdefault("tasks", {}).update(fallback_smoke.get("tasks") or {})
    smoke.setdefault("fallback_runs", []).append(fallback_smoke)
    reporter.emit(
        "Task-level VLM fallback smoke completed",
        event="automatic_vlm_fallback_smoke_completed",
        status="completed",
        tasks=sorted(activated),
    )
    return activated


def seed_fallback_smoke_images(
    source_root: Path,
    target_root: Path,
    tasks: list[str],
    benchmark_cfg: dict,
) -> int:
    copied = 0
    task_configs = benchmark_cfg.get("tasks") or {}
    for task in tasks:
        protocol_config = (task_configs.get(task) or {}).get("protocol_config") or {}
        if not protocol_config.get("fallback_preserved_generation"):
            continue
        source_dir = source_root / task
        if not source_dir.is_dir():
            continue
        target_dir = target_root / task
        target_dir.mkdir(parents=True, exist_ok=True)
        for source_path in source_dir.glob("*_generated.png"):
            if not source_path.is_file() or source_path.stat().st_size <= 0:
                continue
            shutil.copy2(source_path, target_dir / source_path.name)
            copied += 1
    return copied


def smoke_revision_context(smoke_row: dict) -> tuple[dict, list[str]]:
    diagnostics = {
        "failed_checks": list(smoke_row.get("failed_checks") or []),
        "failure_category_counts": dict(smoke_row.get("failure_category_counts") or {}),
        "parser_disagreements": list(smoke_row.get("parser_disagreements") or [])[:10],
        "failed_samples": [],
    }
    generated_paths = []
    results_path = Path(str(smoke_row.get("results_pass1") or ""))
    if not results_path.exists():
        return diagnostics, generated_paths
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return diagnostics, generated_paths
    for row in results.get("detailed_results") or []:
        if row.get("generation_success") and row.get("parse_success"):
            continue
        image_path = str(row.get("generated_image") or "")
        if image_path and Path(image_path).exists():
            generated_paths.append(image_path)
        diagnostics["failed_samples"].append(
            {
                "id": row.get("id"),
                "generation_success": bool(row.get("generation_success")),
                "generation_error_type": row.get("generation_error_type"),
                "generation_error_message": row.get("generation_error_message"),
                "parse_success": bool(row.get("parse_success")),
                "parse_error_type": row.get("parse_error_type"),
                "parse_error_message": row.get("parse_error_message"),
                "parser_ops": row.get("parser_ops"),
            }
        )
    return diagnostics, generated_paths


def write_smoke_samples(
    path: Path,
    items: list[dict],
    *,
    replace_tasks: set[str],
) -> None:
    retained: list[dict] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("task") or "default") not in replace_tasks:
                retained.append(row)
    rows = retained + items
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def run_smoke_validation(
    args: argparse.Namespace,
    benchmark_cfg: dict,
    *,
    reporter: ProgressReporter | None = None,
    smoke_runtime: dict[str, object] | None = None,
) -> dict:
    reporter = reporter or ProgressReporter(enabled=False)
    smoke_phase = str(getattr(args, "smoke_phase", "initial") or "initial")
    smoke_label = {
        "initial": "Protocol smoke",
        "revision": "Revision smoke",
        "fallback": "Fallback smoke",
    }.get(smoke_phase, "Protocol smoke")
    if not benchmark_cfg.get("tasks"):
        return {"model": args.smoke_model, "limit": args.smoke_limit, "tasks": {}}
    spec_dir = str(getattr(args, "protocol_spec_dir", "") or protocol_spec_dir())
    if smoke_runtime is None:
        protocol_pool = load_protocol_pool(spec_dir)
        model = ensure_model(args.smoke_model)
    else:
        if "protocol_pool" not in smoke_runtime:
            smoke_runtime["protocol_pool"] = load_protocol_pool(spec_dir)
        if "model" not in smoke_runtime:
            smoke_runtime["model"] = ensure_model(args.smoke_model)
        protocol_pool = smoke_runtime["protocol_pool"]
        model = smoke_runtime["model"]
    output_root = Path(args.smoke_output or f"outputs/agentic_smoke_{args.benchmark_name}_{args.smoke_model}")
    output_root.mkdir(parents=True, exist_ok=True)
    grouped_items = group_items_by_task(load_items(benchmark_cfg["data_file"]))
    smoke_items = []
    smoke_ids = {}
    for task in benchmark_cfg.get("tasks", {}):
        selected = select_representative_items(grouped_items.get(task, []), args.smoke_limit)
        smoke_items.extend(selected)
        smoke_ids[task] = [str(item.get("id") or "") for item in selected]
    smoke_data_path = output_root / "smoke_samples.jsonl"
    write_smoke_samples(
        smoke_data_path,
        smoke_items,
        replace_tasks=set(benchmark_cfg.get("tasks") or {}),
    )
    smoke_args = argparse.Namespace(
        data_file=str(smoke_data_path),
        benchmark_root=benchmark_cfg["benchmark_root"],
        limit=None,
        no_reuse=True,
        protocol="",
        print_prompt=False,
        model=args.smoke_model,
        reporter=reporter,
        operation_label="Generating smoke image",
        generation_retries=max(0, int(getattr(args, "generation_retries", 1))),
        generation_retry_backoff=max(
            0.0, float(getattr(args, "generation_retry_backoff", 2.0))
        ),
    )
    task_rows = {}
    for task, task_cfg in benchmark_cfg.get("tasks", {}).items():
        try:
            reporter.emit(
                f"Smoke task started with {len(smoke_ids.get(task, []))} sample(s)",
                event="smoke_task_started",
                task=task,
                sample_ids=smoke_ids.get(task, []),
            )
            task_cfg_for_smoke = json.loads(json.dumps(task_cfg, ensure_ascii=False))
            if args.mock_parse_response and task_cfg_for_smoke.get("protocol") == "agentic_vlm_protocol":
                task_cfg_for_smoke.setdefault("protocol_config", {})["mock_parse_response"] = args.mock_parse_response
            first_args = argparse.Namespace(**vars(smoke_args))
            first_args.no_reuse = not bool(getattr(args, "reuse_smoke_images", False))
            first_args.reuse_only = False
            first_attempt = run_task(
                task, task_cfg_for_smoke, protocol_pool, first_args, model, output_root
            )
            task_output = output_root / task
            initial_generation_failures = int(first_attempt.get("generation_failed_count", 0))
            generation_retry_count = 0
            first = first_attempt
            if initial_generation_failures:
                (task_output / "results_generation_attempt1.json").write_text(
                    json.dumps(first_attempt, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                reporter.emit(
                    f"Retrying {initial_generation_failures} failed smoke generation(s) once",
                    event="smoke_generation_retry_started",
                    task=task,
                    failed_count=initial_generation_failures,
                )
                retry_args = argparse.Namespace(**vars(smoke_args))
                retry_args.no_reuse = False
                retry_args.reuse_only = False
                first = run_task(
                    task, task_cfg_for_smoke, protocol_pool, retry_args, model, output_root
                )
                generation_retry_count = 1
            (task_output / "results_pass1.json").write_text(
                json.dumps(first, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            if int(first.get("generated_count", 0)) > 0:
                second_args = argparse.Namespace(**vars(smoke_args))
                second_args.no_reuse = False
                second_args.reuse_only = True
                second = run_task(
                    task,
                    task_cfg_for_smoke,
                    protocol_pool,
                    second_args,
                    model,
                    output_root,
                )
            else:
                second = json.loads(json.dumps(first, ensure_ascii=False))
            (task_output / "results_pass2.json").write_text(
                json.dumps(second, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            parse_rate_pass1 = float(first.get("valid_parse_rate", 0.0))
            parse_rate_pass2 = float(second.get("valid_parse_rate", 0.0))
            parse_rate = min(parse_rate_pass1, parse_rate_pass2)
            readout_rate_pass1 = readout_operational_rate(first)
            readout_rate_pass2 = readout_operational_rate(second)
            readout_rate = min(readout_rate_pass1, readout_rate_pass2)
            generation_rate = float(first.get("generated_rate", 0.0))
            agreement_rate, disagreements = parser_agreement(first, second)
            failure_reasons = smoke_failure_reasons(first, second)
            if disagreements and not failure_reasons:
                failure_reasons = ["parser_disagreement"]
            evidence_required = task_cfg_for_smoke.get("protocol") == "agentic_vlm_protocol"
            evidence_rate_pass1 = spatial_evidence_rate(first, required=evidence_required)
            evidence_rate_pass2 = spatial_evidence_rate(second, required=evidence_required)
            evidence_rate = min(evidence_rate_pass1, evidence_rate_pass2)
            metric_compatibility_rate = min(
                metric_compatibility_rate_among_valid(first),
                metric_compatibility_rate_among_valid(second),
            )
            failed_checks = []
            if generation_rate < args.min_generation_rate:
                failed_checks.append("generation_rate")
            if readout_rate < args.min_parse_success_rate:
                failed_checks.append("parse_success_rate")
            if evidence_required and parse_rate <= 0.0:
                failed_checks.append("no_valid_protocol_example")
            if agreement_rate < args.min_parser_agreement_rate:
                failed_checks.append("parser_agreement_rate")
            if evidence_rate < args.min_spatial_evidence_rate:
                failed_checks.append("spatial_evidence_rate")
            if metric_compatibility_rate < 100.0:
                failed_checks.append("metric_compatibility_rate")
            status = "passed" if not failed_checks else "failed"
            task_rows[task] = {
                "status": status,
                "phase": smoke_phase,
                "valid_parse_rate": parse_rate,
                "valid_parse_rate_pass1": parse_rate_pass1,
                "valid_parse_rate_pass2": parse_rate_pass2,
                "readout_operational_rate": readout_rate,
                "readout_operational_rate_pass1": readout_rate_pass1,
                "readout_operational_rate_pass2": readout_rate_pass2,
                "generated_rate": generation_rate,
                "parser_agreement_rate": agreement_rate,
                "spatial_evidence_rate": evidence_rate,
                "spatial_evidence_rate_pass1": evidence_rate_pass1,
                "spatial_evidence_rate_pass2": evidence_rate_pass2,
                "metric_compatibility_rate": metric_compatibility_rate,
                "mean_score": float(first.get("mean_score", 0.0)),
                "correct_count": int(first.get("correct_count", 0)),
                "initial_generation_failure_count": initial_generation_failures,
                "generation_retry_count": generation_retry_count,
                "failed_checks": failed_checks,
                "parser_disagreements": disagreements,
                "failure_category_counts": first.get("failure_category_counts", {}),
                "failure_reasons": failure_reasons,
                "results_pass1": str(task_output / "results_pass1.json"),
                "results_pass2": str(task_output / "results_pass2.json"),
            }
            reason_suffix = (
                f" reason={','.join(failure_reasons[:2])}"
                if status != "passed" and failure_reasons
                else ""
            )
            reporter.emit(
                f"{smoke_label} {status}: generation={generation_rate:.1f}% "
                f"parse={parse_rate:.1f}% agreement={agreement_rate:.1f}%"
                f"{reason_suffix}",
                event="smoke_task_completed",
                status=status,
                task=task,
                generation_rate=generation_rate,
                parse_rate=parse_rate,
                agreement_rate=agreement_rate,
                failed_checks=failed_checks,
            )
        except Exception as exc:
            task_rows[task] = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
            reporter.emit(
                f"{smoke_label} error: {type(exc).__name__}: {exc}",
                event="smoke_task_failed",
                status="failed",
                task=task,
            )
    return {
        "model": args.smoke_model,
        "limit": args.smoke_limit,
        "smoke_data_file": str(smoke_data_path),
        "sample_ids": smoke_ids,
        "tasks": task_rows,
    }


def apply_smoke_gate(benchmark_cfg: dict, manifest: dict, smoke: dict) -> None:
    tasks = benchmark_cfg.get("tasks", {})
    smoke_tasks = (smoke or {}).get("tasks", {})
    disabled_tasks = []
    deferred_tasks = []
    for task, row in smoke_tasks.items():
        if row.get("status") == "passed":
            continue
        if task not in tasks:
            continue
        tasks.pop(task, None)
        if smoke_failure_is_external(row):
            deferred_tasks.append(task)
        else:
            disabled_tasks.append(task)
    affected_tasks = set(disabled_tasks) | set(deferred_tasks)
    if not affected_tasks:
        return
    for route in manifest.get("route_rows", []):
        task = route.get("task")
        if task in affected_tasks:
            route["pre_smoke_decision"] = route.get("decision")
            route["active"] = False
            if task in deferred_tasks:
                route["decision"] = "deferred"
                route["source"] = "smoke_external_failure"
                route["smoke_status"] = "deferred_external_failure"
                route["reason"] = "protocol smoke was not completed because generation/API failed"
            else:
                route["decision"] = "unsupported"
                route["source"] = "smoke_gate"
                route["smoke_status"] = "failed"
                failed = smoke_tasks.get(task, {}).get("failed_checks", [])
                route["reason"] = "protocol failed smoke validation: " + ", ".join(failed)
    manifest["active_task_count"] = len(tasks)
    manifest["active_sample_count"] = sum(
        int(route.get("sample_count", 0)) for route in manifest.get("route_rows", []) if route.get("active")
    )
    manifest["decision_counts"] = dict(
        sorted(Counter(route.get("decision") for route in manifest.get("route_rows", [])).items())
    )
    manifest["disabled_tasks"] = sorted(disabled_tasks)
    manifest["deferred_external_failure_tasks"] = sorted(deferred_tasks)
    if disabled_tasks:
        manifest.setdefault("warnings", []).append(
            "Smoke validation failed, including any eligible automatic VLM fallback; tasks were "
            "disabled: " + ", ".join(sorted(disabled_tasks))
        )
    if deferred_tasks:
        manifest.setdefault("warnings", []).append(
            "Smoke validation was deferred because the generation service failed: "
            + ", ".join(sorted(deferred_tasks))
        )


def smoke_failure_is_external(smoke_row: dict) -> bool:
    counts = {
        str(name): int(count)
        for name, count in (smoke_row.get("failure_category_counts") or {}).items()
        if int(count) > 0
    }
    return bool(counts) and set(counts) <= {"generation_failure"}


def parser_agreement(first: dict, second: dict) -> tuple[float, list[dict]]:
    first_rows = {str(row.get("id")): row for row in first.get("detailed_results", [])}
    second_rows = {str(row.get("id")): row for row in second.get("detailed_results", [])}
    ids = sorted(set(first_rows) | set(second_rows))
    if not ids:
        return 0.0, []
    stable = 0
    disagreements = []
    for sample_id in ids:
        row_a = first_rows.get(sample_id, {})
        row_b = second_rows.get(sample_id, {})
        prediction_a = normalized_prediction(row_a.get("prediction"))
        prediction_b = normalized_prediction(row_b.get("prediction"))
        readout_a = normalized_readout(row_a)
        readout_b = normalized_readout(row_b)
        agrees = bool(readout_a and readout_a == readout_b)
        if agrees:
            stable += 1
        else:
            disagreements.append(
                {"id": sample_id, "prediction_pass1": prediction_a, "prediction_pass2": prediction_b}
            )
    return 100.0 * stable / len(ids), disagreements


def smoke_failure_reasons(*results: dict) -> list[str]:
    reasons = []
    for result in results:
        for row in result.get("detailed_results") or []:
            if not row.get("generation_success"):
                reason = str(row.get("generation_error_type") or "generation_failed")
                if reason == "reused_output_missing":
                    continue
            elif not row.get("parse_success"):
                reason = str(row.get("parse_error_type") or row.get("error_type") or "parse_failed")
            elif row.get("score_error_type"):
                reason = str(row.get("score_error_type"))
            else:
                continue
            if reason and reason not in reasons:
                reasons.append(reason)
    return reasons


def spatial_evidence_rate(results: dict, *, required: bool) -> float:
    operational = [
        row for row in results.get("detailed_results", []) if readout_is_operational(row)
    ]
    if not operational:
        return 0.0
    if not required:
        return 100.0
    evidence_rows = 0
    for row in operational:
        evidence = str(row.get("agentic_evidence") or "").strip()
        if evidence and not re.fullmatch(
            r"(?i)(?:option|answer|label)?\s*[A-Z0-9]+[.!]?", evidence
        ):
            evidence_rows += 1
    return 100.0 * evidence_rows / len(operational)


def metric_compatibility_rate_among_valid(results: dict) -> float:
    operational = [
        row for row in results.get("detailed_results", []) if readout_is_operational(row)
    ]
    if not operational:
        return 0.0
    return 100.0 * sum(
        bool(row.get("score_computed") or row.get("metric_unverified")) for row in operational
    ) / len(operational)


def readout_is_operational(row: dict) -> bool:
    return bool(row.get("parse_success") or row.get("model_protocol_noncompliance"))


def readout_operational_rate(results: dict) -> float:
    rows = results.get("detailed_results", [])
    if not rows:
        return 0.0
    return 100.0 * sum(readout_is_operational(row) for row in rows) / len(rows)


def normalized_readout(row: dict) -> str:
    if row.get("parse_success"):
        prediction = normalized_prediction(row.get("prediction"))
        return f"valid:{prediction}" if prediction else ""
    if row.get("model_protocol_noncompliance"):
        status = str(row.get("agentic_status") or "invalid").strip().lower()
        return f"noncompliant:{status}"
    return ""


def normalized_prediction(value) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value).strip().lower()


if __name__ == "__main__":
    raise SystemExit(main())
