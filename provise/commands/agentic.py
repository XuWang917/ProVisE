#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

from ..benchmark import AgenticBenchmarkIngestor, write_ingestion_outputs
from ..benchmark.tasks import task_contract
from ..benchmark.package import (
    BenchmarkPackage,
    infer_benchmark_name,
    probe_benchmark_package,
)
from ..benchmark.schema import load_unified_items
from ..models.vlm import create_eval_vlm
from ..paths import protocol_spec_dir, runtime_root
from ..reporting import (
    CONCISE_OUTPUT_ENV,
    ProgressReporter,
    compact_path,
    concise_output_enabled,
)


PROJECT_ROOT = runtime_root()
PROTOCOL_SPEC_DIR = protocol_spec_dir()
load_dotenv(PROJECT_ROOT / ".env")


def parse_args(
    argv: list[str] | None = None,
    *,
    command: str = "run",
) -> argparse.Namespace:
    build_only = command == "build"
    parser = argparse.ArgumentParser(
        prog=f"provise {command}",
        description=(
            "Normalize a benchmark and construct versioned visual protocols."
            if build_only
            else "Normalize a benchmark, build visual protocols, then evaluate an image-generation model."
        )
    )
    parser.add_argument(
        "--source",
        "--source-root",
        dest="source_root",
        required=True,
        help="Raw benchmark directory or a normalized ProVisE benchmark package.",
    )
    parser.add_argument(
        "--model",
        dest="evaluation_model",
        default=os.getenv("PROVISE_EVALUATION_MODEL", "gpt-image-2"),
        help=(
            argparse.SUPPRESS
            if build_only
            else "Image generation model to evaluate (default: gpt-image-2)."
        ),
    )
    parser.add_argument(
        "--protocol-model",
        "--smoke-model",
        dest="protocol_model",
        default=os.getenv("PROVISE_PROTOCOL_MODEL", "gpt-image-2"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--benchmark-name",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--workspace",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--agent-model",
        default=os.getenv("PROVISE_AGENT_MODEL", "gpt-5.4"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--parser-model",
        default=os.getenv("PROVISE_PARSER_MODEL", ""),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--agent-timeout", type=int, default=180, help=argparse.SUPPRESS)
    parser.add_argument("--agent-max-tokens", type=int, default=4096, help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-examples-per-source", type=int, default=3, help=argparse.SUPPRESS
    )
    parser.add_argument("--max-sources", type=int, default=30, help=argparse.SUPPRESS)
    parser.add_argument(
        "--ingestion-response-file",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--protocol-response-file",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--ingest-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--build-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-limit", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument(
        "--generation-retries",
        type=int,
        default=int(os.getenv("PROVISE_GENERATION_RETRIES", "1")),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--generation-retry-backoff",
        type=float,
        default=float(os.getenv("PROVISE_GENERATION_RETRY_BACKOFF", "2")),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no-smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--tasks",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reuse-smoke-images",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--evaluate-limit",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--evaluate-samples",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            argparse.SUPPRESS
            if build_only
            else "Run every sample in each accepted task after smoke validation."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show artifact paths and sample-level diagnostics in the terminal.",
    )
    parser.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--quiet-progress", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--heartbeat-seconds", type=float, default=1.0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        choices=(0, 1),
        default=1,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mock-parse-response",
        default="",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if build_only:
        args.build_only = True
    return args


def main(argv: list[str] | None = None, *, command: str = "run") -> int:
    args = parse_args(argv, command=command)
    previous_output_mode = os.environ.get(CONCISE_OUTPUT_ENV)
    os.environ[CONCISE_OUTPUT_ENV] = "0" if args.verbose else "1"
    try:
        return _run(args)
    finally:
        if previous_output_mode is None:
            os.environ.pop(CONCISE_OUTPUT_ENV, None)
        else:
            os.environ[CONCISE_OUTPUT_ENV] = previous_output_mode


def _run(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root).expanduser().resolve()
    package_probe = probe_benchmark_package(
        source_root,
        benchmark_name=str(args.benchmark_name or "").strip(),
    )
    if package_probe.package is not None:
        args.benchmark_name = package_probe.package.benchmark_name
    elif not args.benchmark_name:
        args.benchmark_name = infer_benchmark_name(source_root)
    if args.build_only:
        args.evaluate_limit = 0
        args.evaluate_samples = 0
    elif args.full:
        args.evaluate_limit = -1
        args.evaluate_samples = 0
    elif args.evaluate_samples > 0 and args.evaluate_limit is None:
        args.evaluate_limit = -1
    elif args.evaluate_limit is None:
        args.evaluate_limit = 5
    parser_model = str(args.parser_model or args.agent_model).strip()
    child_env = evaluation_subprocess_env(parser_model)
    total_stages = 5 if args.build_only else 6
    workspace = resolve_workspace(args)
    workspace.mkdir(parents=True, exist_ok=True)
    if args.force:
        (workspace / "action_required.json").unlink(missing_ok=True)
    reporter = ProgressReporter(
        workspace / f"{args.benchmark_name}.progress.jsonl",
        enabled=not args.quiet_progress,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    reporter.emit(
        f"Agentic benchmark run started: {args.benchmark_name}",
        event="run_started",
        stage=1,
        total_stages=total_stages,
        source_root=str(source_root),
        workspace=str(workspace),
        protocol_model=args.protocol_model,
        evaluation_model=args.evaluation_model,
    )
    ingestion_manifest = workspace / f"{args.benchmark_name}.ingestion_manifest.json"
    total_task_count = 0
    if ingestion_manifest.exists() and not args.force:
        reporter.emit(
            f"Ingestion artifact already exists: {ingestion_manifest}",
            event="run_stopped_existing_artifact",
            status="stopped",
        )
        print("Pass --force to rebuild it.", flush=True)
        return 0

    if package_probe.status == "invalid":
        action_path = write_action_required(
            workspace,
            benchmark_name=args.benchmark_name,
            stage="benchmark_package_validation",
            reason=package_probe.reason,
            details={
                "source": str(source_root),
                "candidates": package_probe.candidates,
                "validation": package_probe.validation,
            },
            required=(
                "Fix the normalized package fields or media paths reported in validation.",
                "Keep benchmark.yaml, data.jsonl, and referenced assets under one package root.",
            ),
        )
        reporter.emit(
            f"Normalized benchmark package is invalid: {package_probe.reason}",
            event="benchmark_package_invalid",
            status="failed",
            action_required=str(action_path),
        )
        print_action_path(action_path)
        return 2

    if package_probe.package is not None:
        package = package_probe.package
        source_mode = "normalized_package"
        reporter.emit(
            "Ingestion agent skipped: normalized package detected",
            event="ingestion_agent_skipped",
            status="completed",
            stage=2,
            total_stages=total_stages,
        )
        paths = register_normalized_package(package, workspace)
        benchmark_root = str(package.benchmark_root)
        sample_count = len(package.items)
        task_counts = dict(
            sorted(Counter(str(item.get("task") or "default") for item in package.items).items())
        )
        total_task_count = len(task_counts)
        reporter.emit(
            f"Using validated benchmark package: {sample_count} samples, {len(task_counts)} tasks",
            event="normalized_package_loaded",
            status="completed",
            stage=3,
            total_stages=total_stages,
            sample_count=sample_count,
            task_counts=task_counts,
            data_file=str(package.data_file),
            benchmark_root=benchmark_root,
        )
        if not concise_output_enabled():
            print("=" * 72)
            print("ProVisE Benchmark Package")
            print(f"benchmark: {args.benchmark_name}")
            print("source:    normalized package (ingestion agent skipped)")
            print(f"samples:   {sample_count}")
            print(f"tasks:     {task_counts}")
    else:
        source_mode = "agentic_ingestion"
        if not source_root.is_dir():
            action_path = write_action_required(
                workspace,
                benchmark_name=args.benchmark_name,
                stage="benchmark_ingestion",
                reason="A raw benchmark source must be a directory.",
                details={"source": str(source_root)},
                required=(
                    "Pass the downloaded benchmark directory with --source.",
                    "Alternatively provide a valid genbench.v1 JSONL package.",
                ),
            )
            print_action_path(action_path)
            return 2
        raw_response = ""
        vlm = None
        if args.ingestion_response_file:
            reporter.emit(
                "Loading saved ingestion-agent response",
                event="ingestion_response_reused",
                status="completed",
                stage=2,
                total_stages=total_stages,
            )
            raw_response = Path(args.ingestion_response_file).read_text(encoding="utf-8")
        else:
            reporter.emit(
                f"Initializing ingestion agent: {args.agent_model}",
                event="agent_model_loading",
                stage=2,
                total_stages=total_stages,
            )
            vlm = create_eval_vlm(
                timeout=args.agent_timeout,
                max_tokens=args.agent_max_tokens,
                model_name=args.agent_model,
            )
            vlm.load_model()

        ingestor = AgenticBenchmarkIngestor(
            source_root=source_root,
            benchmark_name=args.benchmark_name,
            output_root=workspace,
            max_examples_per_source=args.max_examples_per_source,
            max_sources=args.max_sources,
            max_revisions=args.max_revisions,
            reporter=reporter,
        )
        with reporter.waiting(
            "Inspecting benchmark and building ingestion mapping",
            event="benchmark_ingestion",
            model=args.agent_model,
        ):
            result = ingestor.build(vlm=vlm, raw_response=raw_response)
        paths = write_ingestion_outputs(result, workspace)
        benchmark_root = str(workspace)
        total_task_count = len(result.manifest.get("task_counts", {}))
        converted_count = int((result.manifest.get("validation") or {}).get("sample_count") or 0)
        if not concise_output_enabled():
            print("=" * 72)
            print("Agentic Benchmark Ingestion")
            print(f"benchmark: {args.benchmark_name}")
            print(f"decision:  {result.decision}")
            if result.decision != "ingest" and converted_count:
                print(f"samples:   {len(result.items)} accepted / {converted_count} converted")
            else:
                print(f"samples:   {len(result.items)}")
            print(f"tasks:     {result.manifest.get('task_counts', {})}")
            for name, path in paths.items():
                print(f"  {name}: {path}")
        if result.decision != "ingest":
            stop_message = f"Ingestion stopped safely: {result.manifest.get('reason', '')}"
            action_path = write_action_required(
                workspace,
                benchmark_name=args.benchmark_name,
                stage="benchmark_ingestion",
                reason=str(result.manifest.get("reason") or "raw benchmark could not be normalized"),
                details={
                    "source": str(source_root),
                    "ingestion_manifest": paths["manifest"],
                    "attempts": paths.get("attempts", ""),
                    "validation": result.manifest.get("validation") or {},
                },
                required=(
                    "Ensure annotations expose question, ground truth, and resolvable input media.",
                    "Add or correct a declarative ingestion mapping when the automatic mapping is insufficient.",
                ),
            )
            if args.quiet_progress:
                print(stop_message)
            reporter.emit(
                stop_message,
                event="ingestion_stopped",
                status="stopped",
                action_required=str(action_path),
            )
            print_action_path(action_path)
            return 2
        reporter.emit(
            f"Ingestion validated: {len(result.items)} samples, "
            f"{len(result.manifest.get('task_counts', {}))} tasks",
            event="ingestion_validated",
            status="completed",
            stage=3,
            total_stages=total_stages,
            sample_count=len(result.items),
            task_counts=result.manifest.get("task_counts", {}),
        )

    if args.ingest_only:
        return 0

    if not args.tasks and args.max_tasks > 0:
        selected_tasks = select_representative_tasks(
            paths["unified_data"],
            args.max_tasks,
        )
        args.tasks = ",".join(selected_tasks)
        reporter.emit(
            f"Selected {len(selected_tasks)}/{total_task_count} representative task(s) for this pilot",
            event="pilot_tasks_selected",
            status="completed",
            selected_tasks=selected_tasks,
            total_task_count=total_task_count,
        )

    protocol_name = f"{args.benchmark_name}_agentic"
    config_dir = workspace / "configs"
    generated_dir = workspace / "generated"
    smoke_dir = workspace / "smoke"
    protocol_command = [
        sys.executable,
        "-m",
        "provise.protocol_agent.pipeline",
        "--benchmark-name",
        protocol_name,
        "--input",
        paths["unified_data"],
        "--benchmark-root",
        benchmark_root,
        "--benchmark-config-dir",
        str(config_dir),
        "--generated-protocol-dir",
        str(generated_dir),
        "--protocol-spec-dir",
        str(PROTOCOL_SPEC_DIR),
        "--smoke-output",
        str(smoke_dir),
        "--router-model",
        args.agent_model,
        "--smoke-model",
        args.protocol_model,
        "--smoke-limit",
        str(args.smoke_limit),
        "--force",
        "--progress-events",
        str(reporter.event_path),
        "--heartbeat-seconds",
        str(args.heartbeat_seconds),
        "--max-revisions",
        str(args.max_revisions),
        "--generation-retries",
        str(max(0, args.generation_retries)),
        "--generation-retry-backoff",
        str(max(0.0, args.generation_retry_backoff)),
    ]
    if args.no_smoke:
        protocol_command.append("--no-smoke")
    if args.reuse_smoke_images:
        protocol_command.append("--reuse-smoke-images")
    if args.protocol_response_file:
        protocol_command.extend(["--agent-response-file", args.protocol_response_file])
    if args.mock_parse_response:
        protocol_command.extend(["--mock-parse-response", args.mock_parse_response])
    if args.tasks:
        protocol_command.extend(["--tasks", args.tasks])
    child_env["PYTHONUNBUFFERED"] = "1"
    reporter.emit(
        "Constructing protocols and running smoke validation",
        event="protocol_subprocess_started",
        stage=4,
        total_stages=total_stages,
    )
    try:
        subprocess.run(protocol_command, cwd=PROJECT_ROOT, env=child_env, check=True)
    except subprocess.CalledProcessError as exc:
        action_path = write_action_required(
            workspace,
            benchmark_name=args.benchmark_name,
            stage="protocol_construction",
            reason=f"protocol construction process failed with exit code {exc.returncode}",
            details={"command": protocol_command, "progress": str(reporter.event_path)},
            required=(
                "Inspect the final protocol-construction error in the progress log.",
                "Retry after resolving an external API failure; extend a registered contract only for a real framework gap.",
            ),
        )
        reporter.emit(
            "Protocol construction process failed",
            event="protocol_subprocess_failed",
            status="failed",
            action_required=str(action_path),
        )
        print_action_path(action_path)
        return int(exc.returncode or 2)

    config_path = config_dir / f"{protocol_name}.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    active_tasks, formal_tasks, metric_unverified_tasks = evaluation_task_sets(config)
    requested_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    selected_task_count = len(requested_tasks) if requested_tasks else total_task_count
    run_manifest = {
        "benchmark": args.benchmark_name,
        "status": "protocol_ready" if active_tasks else "no_active_tasks",
        "source_mode": source_mode,
        "source": str(source_root),
        "protocol_model": args.protocol_model,
        "evaluation_model": args.evaluation_model,
        "workspace": str(workspace),
        "ingestion_manifest": paths["manifest"],
        "protocol_config": str(config_path),
        "parser_model": parser_model,
        "active_tasks": active_tasks,
        "selected_tasks": [
            task.strip() for task in args.tasks.split(",") if task.strip()
        ],
        "selected_task_count": selected_task_count,
        "total_task_count": total_task_count,
        "formal_evaluation_tasks": formal_tasks,
        "metric_unverified_tasks": metric_unverified_tasks,
        "evaluation_started": False,
        "evaluation_sample_budget": max(0, int(args.evaluate_samples)),
    }
    action_path = None
    if not active_tasks:
        action_path = write_action_required(
            workspace,
            benchmark_name=args.benchmark_name,
            stage="protocol_smoke_validation",
            reason="No task retained a valid visual protocol after deterministic and VLM fallback smoke.",
            details={
                "protocol_config": str(config_path),
                "protocol_manifest": str(
                    config_dir / f"{protocol_name}.agentic_manifest.json"
                ),
            },
            required=(
                "Inspect task-level smoke failure attribution in the protocol manifest.",
                "Retry external generation failures or add a reusable Parser Op for a genuine unsupported contract.",
            ),
        )
        run_manifest["action_required"] = str(action_path)
    if metric_unverified_tasks:
        reporter.emit(
            "Formal scoring blocked for metric-unverified task(s): "
            + ", ".join(metric_unverified_tasks),
            event="formal_evaluation_blocked",
            status="stopped",
            tasks=metric_unverified_tasks,
        )
        if args.evaluate_limit != 0 and not formal_tasks and active_tasks:
            action_path = write_action_required(
                workspace,
                benchmark_name=args.benchmark_name,
                stage="metric_verification",
                reason="Protocols passed smoke, but no task has a verified benchmark metric.",
                details={
                    "tasks": metric_unverified_tasks,
                    "ingestion_manifest": paths["manifest"],
                },
                required=(
                    "Set the official registered metric and its parameters in the normalized benchmark package.",
                ),
            )
            run_manifest["action_required"] = str(action_path)
    if args.evaluate_limit != 0 and formal_tasks:
        if args.evaluate_samples > 0:
            eval_name = f"pilot_{args.evaluate_samples}_total"
        else:
            eval_name = "full" if args.evaluate_limit < 0 else f"pilot_{args.evaluate_limit}"
        eval_output = workspace / eval_name
        eval_command = [
            sys.executable,
            "-m",
            "provise.cli",
            "evaluate",
            "--model",
            args.evaluation_model,
            "--protocol",
            str(config_path),
            "--output",
            str(eval_output),
            "--tasks",
            ",".join(formal_tasks),
            "--progress-events",
            str(reporter.event_path),
            "--heartbeat-seconds",
            str(args.heartbeat_seconds),
            "--generation-retries",
            str(max(0, args.generation_retries)),
            "--generation-retry-backoff",
            str(max(0.0, args.generation_retry_backoff)),
        ]
        if args.evaluate_limit < 0:
            eval_command.append("--full")
        elif args.evaluate_limit > 0:
            eval_command.extend(["--limit", str(args.evaluate_limit)])
        if args.evaluate_samples > 0:
            eval_command.extend(["--max-samples", str(args.evaluate_samples)])
        reporter.emit(
            f"Starting formal evaluation for {len(formal_tasks)} verified task(s)",
            event="evaluation_started",
            stage=5,
            total_stages=total_stages,
            active_tasks=formal_tasks,
        )
        try:
            subprocess.run(eval_command, cwd=PROJECT_ROOT, env=child_env, check=True)
        except subprocess.CalledProcessError as exc:
            action_path = write_action_required(
                workspace,
                benchmark_name=args.benchmark_name,
                stage="formal_evaluation",
                reason=f"evaluation process failed with exit code {exc.returncode}",
                details={"command": eval_command, "output": str(eval_output)},
                required=(
                    "Inspect the evaluation output and retry external model failures.",
                ),
            )
            run_manifest.update(
                status="evaluation_failed",
                evaluation_error=f"exit code {exc.returncode}",
                action_required=str(action_path),
            )
        else:
            evaluation_summary = load_evaluation_summary(eval_output / "summary.json")
            run_manifest.update(
                status="completed",
                evaluation_started=True,
                evaluation_output=str(eval_output),
                evaluation_summary=evaluation_summary,
            )
    run_manifest_path = workspace / f"{args.benchmark_name}.agentic_run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    final_status = str(run_manifest.get("status") or "completed")
    reporter.emit(
        final_run_message(
            final_status,
            active_task_count=len(active_tasks),
            selected_task_count=selected_task_count,
            total_task_count=total_task_count,
        ),
        event="run_completed",
        status="completed" if final_status in {"completed", "protocol_ready"} else "stopped",
        stage=total_stages,
        total_stages=total_stages,
        active_tasks=active_tasks,
        evaluation_started=run_manifest["evaluation_started"],
    )
    if concise_output_enabled():
        evaluation_summary = run_manifest.get("evaluation_summary") or {}
        if evaluation_summary:
            print(evaluation_summary_message(evaluation_summary))
        print(f"Results: {compact_path(workspace, PROJECT_ROOT)}")
    else:
        print("=" * 72)
        print(f"Active tasks: {active_tasks}")
        print(f"Run manifest: {run_manifest_path}")
    if action_path:
        print_action_path(action_path)
    return final_exit_code(final_status, active_task_count=len(active_tasks))


def evaluation_subprocess_env(parser_model: str) -> dict[str, str]:
    env = os.environ.copy()
    if parser_model:
        env["PROVISE_PARSER_MODEL"] = parser_model
    if concise_output_enabled():
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        env["TRANSFORMERS_VERBOSITY"] = "error"
        env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def print_action_path(path: str | Path) -> None:
    print(f"Action required: {compact_path(path, PROJECT_ROOT)}", flush=True)


def load_evaluation_summary(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    overall = payload.get("overall") or {}
    return {
        "total_samples": int(overall.get("total_samples", 0)),
        "correct_count": int(overall.get("correct_count", 0)),
        "accuracy": float(overall.get("accuracy", 0.0)),
        "valid_parse_count": int(overall.get("valid_parse_count", 0)),
        "valid_parse_rate": float(overall.get("valid_parse_rate", 0.0)),
        "generation_failed_count": int(overall.get("generation_failed_count", 0)),
        "parser_failure_count": int(overall.get("parser_failure_count", 0)),
    }


def final_run_message(
    status: str,
    *,
    active_task_count: int,
    selected_task_count: int,
    total_task_count: int,
) -> str:
    selected_total = max(selected_task_count, active_task_count)
    benchmark_total = max(total_task_count, selected_total)
    if selected_total < benchmark_total:
        task_summary = (
            f"{active_task_count}/{selected_total} selected tasks active; "
            f"benchmark coverage {active_task_count}/{benchmark_total}"
        )
    else:
        task_summary = f"{active_task_count}/{benchmark_total} tasks active"
    if status in {"completed", "protocol_ready"}:
        return f"Run completed: {task_summary}"
    return f"Run finished with status={status}: {task_summary}"


def evaluation_summary_message(summary: dict) -> str:
    total = int(summary.get("total_samples", 0))
    correct = int(summary.get("correct_count", 0))
    accuracy = float(summary.get("accuracy", 0.0))
    valid = int(summary.get("valid_parse_count", 0))
    parts = [f"Pilot: {correct}/{total} correct ({accuracy:.1f}%)", f"{valid} valid"]
    generation_failed = int(summary.get("generation_failed_count", 0))
    parser_failed = int(summary.get("parser_failure_count", 0))
    if generation_failed:
        parts.append(f"{generation_failed} generation failures")
    if parser_failed:
        parts.append(f"{parser_failed} parser failures")
    return "; ".join(parts)


def final_exit_code(status: str, *, active_task_count: int) -> int:
    if active_task_count <= 0:
        return 3
    if status == "evaluation_failed":
        return 4
    return 0


def resolve_workspace(args: argparse.Namespace) -> Path:
    if str(args.workspace or "").strip():
        return Path(args.workspace).expanduser().resolve()
    runs_root = Path(
        os.getenv("PROVISE_RUNS_DIR", str(PROJECT_ROOT / "outputs" / "agentic_runs"))
    ).expanduser()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return (runs_root / args.benchmark_name / run_id).resolve()


def register_normalized_package(
    package: BenchmarkPackage,
    workspace: Path,
) -> dict[str, str]:
    task_counts = dict(
        sorted(Counter(str(item.get("task") or "default") for item in package.items).items())
    )
    manifest_path = workspace / f"{package.benchmark_name}.ingestion_manifest.json"
    manifest = {
        "benchmark": package.benchmark_name,
        "decision": "ingest",
        "mode": "normalized_package",
        "reason": "validated normalized benchmark package; ingestion agent was skipped",
        "data_file": str(package.data_file),
        "benchmark_root": str(package.benchmark_root),
        "package_manifest": str(package.manifest_path or ""),
        "validation": package.validation,
        "task_counts": task_counts,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "unified_data": str(package.data_file),
        "package": str(package.manifest_path or package.data_file),
        "manifest": str(manifest_path),
    }


def write_action_required(
    workspace: Path,
    *,
    benchmark_name: str,
    stage: str,
    reason: str,
    details: dict,
    required: tuple[str, ...],
) -> Path:
    path = workspace / "action_required.json"
    payload = {
        "status": "action_required",
        "benchmark": benchmark_name,
        "stage": stage,
        "reason": reason,
        "required_input": list(required),
        "details": details,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def evaluation_task_sets(config: dict) -> tuple[list[str], list[str], list[str]]:
    tasks = config.get("tasks") or {}
    active = list(tasks)
    formal = [task for task, task_cfg in tasks.items() if bool(task_cfg.get("formal_evaluation"))]
    metric_unverified = sorted(set(active) - set(formal))
    return active, formal, metric_unverified


def select_representative_tasks(data_file: str | Path, limit: int) -> list[str]:
    """Select high-coverage tasks while preserving answer/metric diversity."""

    max_tasks = max(0, int(limit))
    if max_tasks == 0:
        return []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in load_unified_items(data_file):
        grouped[str(item.get("task") or "default")].append(item)

    candidates = []
    for task, items in grouped.items():
        signatures = {
            json.dumps(
                task_contract(item),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for item in items
        }
        candidates.append(
            {
                "task": task,
                "sample_count": len(items),
                "signature": "|".join(sorted(signatures)),
            }
        )
    candidates.sort(key=lambda row: (-row["sample_count"], row["task"]))

    selected: list[str] = []
    seen_signatures: set[str] = set()
    for row in candidates:
        if row["signature"] in seen_signatures:
            continue
        selected.append(row["task"])
        seen_signatures.add(row["signature"])
        if len(selected) >= max_tasks:
            return selected
    for row in candidates:
        if row["task"] not in selected:
            selected.append(row["task"])
        if len(selected) >= max_tasks:
            break
    return selected


if __name__ == "__main__":
    raise SystemExit(main())
