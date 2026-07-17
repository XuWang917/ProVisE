from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml
from dotenv import load_dotenv

from ..paths import benchmark_suite_path, runtime_root
from ..reporting import style_terminal


PROJECT_ROOT = runtime_root()
DEFAULT_SUITE = benchmark_suite_path()
SUITE_SCHEMA_VERSION = "provise.suite.v1"
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class BenchmarkEntry:
    benchmark_id: str
    source: str
    family: str = ""
    enabled: bool = True
    source_url: str = ""
    source_env: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="provise suite",
        description=(
            "Build versioned protocols and run a uniform pilot across spatial benchmarks."
        ),
    )
    parser.add_argument(
        "--suite",
        default=str(DEFAULT_SUITE),
        help="Benchmark-suite YAML (default: validated_spatial).",
    )
    parser.add_argument(
        "--benchmark-root",
        default=os.getenv("PROVISE_BENCHMARK_ROOT", ""),
        help="Directory containing the downloaded benchmark directories.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("PROVISE_EVALUATION_MODEL", "gpt-image-2"),
        help="Image generation model evaluated with every protocol.",
    )
    parser.add_argument(
        "--agent-model",
        default=os.getenv("PROVISE_AGENT_MODEL", "gpt-5.4"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--parser-model",
        default=os.getenv("PROVISE_PARSER_MODEL", "gpt-5.4"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--protocol-model",
        default=os.getenv("PROVISE_PROTOCOL_MODEL", "gpt-image-2"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--benchmarks",
        default="",
        help="Optional comma-separated benchmark ids from the suite.",
    )
    parser.add_argument(
        "--max-tasks-per-benchmark",
        type=int,
        default=0,
        help="Representative task cap for each benchmark (default: 0, all tasks).",
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=3,
        help="Smoke samples per selected task (default: 3).",
    )
    parser.add_argument(
        "--evaluate-limit",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--samples-per-benchmark",
        type=int,
        default=24,
        help="Total pilot sample budget per benchmark (default: 24).",
    )
    parser.add_argument("--full", action="store_true", help="Evaluate every accepted sample.")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build and smoke protocols without evaluating the target model.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Suite output directory. A timestamped directory is used by default.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed benchmark runs in the output directory (default: enabled).",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail the suite when a benchmark directory is missing instead of skipping it.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild benchmark workspaces.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--verbose", action="store_true", help="Show detailed ProVisE output.")
    parser.add_argument(
        "--timeout-minutes",
        type=float,
        default=0.0,
        help="Optional timeout for each benchmark; 0 disables the timeout.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    try:
        return run_suite(args)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(style_terminal(f"Suite error: {exc}", tone="error"), file=sys.stderr)
        return 2


def run_suite(args: argparse.Namespace) -> int:
    suite_path = Path(args.suite).expanduser().resolve()
    suite_name, entries = load_suite(suite_path)
    benchmark_root = Path(args.benchmark_root).expanduser().resolve() if args.benchmark_root else None
    if benchmark_root is None:
        raise ValueError(
            "--benchmark-root is required (or set PROVISE_BENCHMARK_ROOT)"
        )
    selected_ids = {
        value.strip() for value in str(args.benchmarks or "").split(",") if value.strip()
    }
    known_ids = {entry.benchmark_id for entry in entries}
    unknown = sorted(selected_ids - known_ids)
    if unknown:
        raise ValueError(f"unknown benchmark id(s) in --benchmarks: {unknown}")
    selected = [
        entry
        for entry in entries
        if entry.enabled and (not selected_ids or entry.benchmark_id in selected_ids)
    ]
    if not selected:
        raise ValueError("the suite contains no selected benchmarks")

    output_root = resolve_output_root(args.output, suite_name)
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    rows: list[dict[str, Any]] = []
    write_suite_state(
        output_root,
        suite_name=suite_name,
        suite_path=suite_path,
        benchmark_root=benchmark_root,
        args=args,
        started_at=started_at,
        rows=rows,
        status="running",
    )

    print(style_terminal(f"Spatial benchmark suite: {suite_name}", bold=True))
    print(f"Benchmarks: {len(selected)} | target model: {args.model}")
    print(f"Results: {output_root}")

    for index, entry in enumerate(selected, 1):
        source = resolve_source(benchmark_root, entry)
        workspace = output_root / "benchmarks" / entry.benchmark_id
        prefix = f"[Benchmark {index}/{len(selected)}: {entry.benchmark_id}]"
        print("\n" + style_terminal(prefix, bold=True))

        if not source.is_dir():
            message = f"Missing source: {source}"
            print(style_terminal(message, tone="error" if args.strict_missing else "warning"))
            rows.append(
                base_result_row(
                    entry,
                    source,
                    workspace,
                    status="missing",
                    return_code=2 if args.strict_missing else 0,
                    error=message,
                )
            )
            write_suite_state(
                output_root,
                suite_name=suite_name,
                suite_path=suite_path,
                benchmark_root=benchmark_root,
                args=args,
                started_at=started_at,
                rows=rows,
                status="running",
            )
            continue

        existing = load_run_manifest(workspace, entry.benchmark_id)
        if args.resume and not args.force and run_is_complete(existing, build_only=args.build_only):
            row = summarize_benchmark_run(entry, source, workspace, 0, 0.0, existing)
            row["status"] = "resumed"
            rows.append(row)
            print(style_terminal("Reused completed run", tone="success"))
            write_suite_state(
                output_root,
                suite_name=suite_name,
                suite_path=suite_path,
                benchmark_root=benchmark_root,
                args=args,
                started_at=started_at,
                rows=rows,
                status="running",
            )
            continue

        command = build_benchmark_command(args, entry, source, workspace)
        print(f"Command: {shlex.join(command)}")
        if args.dry_run:
            rows.append(
                base_result_row(entry, source, workspace, status="planned", return_code=0)
            )
            continue

        started = time.monotonic()
        return_code, error = run_command(
            command,
            cwd=PROJECT_ROOT,
            timeout_seconds=max(0.0, float(args.timeout_minutes)) * 60.0,
        )
        elapsed = time.monotonic() - started
        manifest = load_run_manifest(workspace, entry.benchmark_id)
        row = summarize_benchmark_run(
            entry,
            source,
            workspace,
            return_code,
            elapsed,
            manifest,
            error=error,
        )
        rows.append(row)
        tone = (
            "success"
            if row["status"] in {"completed", "protocol_ready"}
            else "warning"
            if row["status"] == "blocked"
            else "error"
        )
        print(style_terminal(format_benchmark_result(row), tone=tone))
        write_suite_state(
            output_root,
            suite_name=suite_name,
            suite_path=suite_path,
            benchmark_root=benchmark_root,
            args=args,
            started_at=started_at,
            rows=rows,
            status="running",
        )

    final_status = suite_status(rows, strict_missing=args.strict_missing)
    write_suite_state(
        output_root,
        suite_name=suite_name,
        suite_path=suite_path,
        benchmark_root=benchmark_root,
        args=args,
        started_at=started_at,
        rows=rows,
        status=final_status,
    )
    totals = aggregate_suite(rows)
    print("\n" + style_terminal("Suite complete", bold=True))
    print(
        f"Ready: {totals['ready_benchmarks']}/{totals['available_benchmarks']} available benchmarks | "
        f"blocked: {totals['blocked_benchmarks']} | "
        f"active tasks: {totals['active_tasks']}/{totals['selected_tasks']}"
    )
    print(f"Summary: {output_root / 'summary.md'}")
    return 0 if final_status in {
        "completed",
        "completed_with_blocks",
        "completed_with_skips",
        "planned",
    } else 1


def load_suite(path: str | Path) -> tuple[str, list[BenchmarkEntry]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"suite YAML does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if payload.get("schema_version") != SUITE_SCHEMA_VERSION:
        raise ValueError(
            f"suite schema_version must be {SUITE_SCHEMA_VERSION!r}: {source}"
        )
    name = str(payload.get("name") or source.stem).strip()
    raw_entries = payload.get("benchmarks")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("suite benchmarks must be a non-empty list")
    entries = []
    seen = set()
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise ValueError(f"benchmarks[{index}] must be an object")
        benchmark_id = str(raw.get("id") or "").strip()
        benchmark_source = str(raw.get("source") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", benchmark_id):
            raise ValueError(f"invalid benchmark id at benchmarks[{index}]: {benchmark_id!r}")
        if benchmark_id in seen:
            raise ValueError(f"duplicate benchmark id: {benchmark_id}")
        if not benchmark_source:
            raise ValueError(f"benchmarks[{index}].source is required")
        seen.add(benchmark_id)
        entries.append(
            BenchmarkEntry(
                benchmark_id=benchmark_id,
                source=benchmark_source,
                family=str(raw.get("family") or "").strip(),
                enabled=bool(raw.get("enabled", True)),
                source_url=str(raw.get("source_url") or "").strip(),
                source_env=str(raw.get("source_env") or "").strip(),
            )
        )
    return name, entries


def resolve_source(benchmark_root: Path, entry: BenchmarkEntry) -> Path:
    value = os.getenv(entry.source_env, "").strip() if entry.source_env else ""
    value = value or entry.source
    path = Path(os.path.expandvars(value)).expanduser()
    return path.resolve() if path.is_absolute() else (benchmark_root / path).resolve()


def resolve_output_root(value: str, suite_name: str) -> Path:
    if str(value or "").strip():
        return Path(value).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (PROJECT_ROOT / "outputs" / "suites" / suite_name / timestamp).resolve()


def build_benchmark_command(
    args: argparse.Namespace,
    entry: BenchmarkEntry,
    source: Path,
    workspace: Path,
) -> list[str]:
    command_name = "build" if args.build_only else "run"
    command = [
        sys.executable,
        "-m",
        "provise.cli",
        command_name,
        "--source",
        str(source),
        "--benchmark-name",
        entry.benchmark_id,
        "--workspace",
        str(workspace),
        "--agent-model",
        str(args.agent_model),
        "--parser-model",
        str(args.parser_model),
        "--protocol-model",
        str(args.protocol_model),
        "--max-tasks",
        str(max(0, int(args.max_tasks_per_benchmark))),
        "--smoke-limit",
        str(max(1, int(args.smoke_limit))),
    ]
    if not args.build_only:
        command.extend(["--model", str(args.model)])
        if args.full:
            command.append("--full")
        else:
            command.extend(
                ["--evaluate-samples", str(max(1, int(args.samples_per_benchmark)))]
            )
            if args.evaluate_limit > 0:
                command.extend(["--evaluate-limit", str(args.evaluate_limit)])
    if args.verbose:
        command.append("--verbose")
    if args.force or workspace.exists():
        command.append("--force")
    return command


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> tuple[int, str]:
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    try:
        return process.wait(timeout=timeout_seconds or None), ""
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return 124, f"benchmark timed out after {timeout_seconds / 60.0:g} minute(s)"


def run_manifest_path(workspace: Path, benchmark_id: str) -> Path:
    return workspace / f"{benchmark_id}.agentic_run_manifest.json"


def load_run_manifest(workspace: Path, benchmark_id: str) -> dict[str, Any]:
    path = run_manifest_path(workspace, benchmark_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def run_is_complete(manifest: dict[str, Any], *, build_only: bool) -> bool:
    status = str(manifest.get("status") or "")
    if build_only:
        return status in {"protocol_ready", "completed"}
    return status == "completed" and bool(manifest.get("evaluation_started"))


def load_protocol_manifest(workspace: Path, benchmark_id: str) -> dict[str, Any]:
    path = workspace / "configs" / f"{benchmark_id}_agentic.agentic_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_action_required(workspace: Path) -> dict[str, Any]:
    try:
        payload = json.loads((workspace / "action_required.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def base_result_row(
    entry: BenchmarkEntry,
    source: Path,
    workspace: Path,
    *,
    status: str,
    return_code: int,
    error: str = "",
) -> dict[str, Any]:
    return {
        "benchmark": entry.benchmark_id,
        "family": entry.family,
        "source": str(source),
        "source_url": entry.source_url,
        "workspace": str(workspace),
        "status": status,
        "return_code": int(return_code),
        "elapsed_seconds": 0.0,
        "total_tasks": 0,
        "selected_tasks": 0,
        "active_tasks": 0,
        "formal_tasks": 0,
        "deterministic_tasks": 0,
        "fallback_tasks": 0,
        "disabled_tasks": 0,
        "deferred_tasks": 0,
        "evaluated_samples": 0,
        "accuracy": None,
        "valid_parse_rate": None,
        "generation_failed_count": 0,
        "parser_failure_count": 0,
        "error": error,
    }


def summarize_benchmark_run(
    entry: BenchmarkEntry,
    source: Path,
    workspace: Path,
    return_code: int,
    elapsed_seconds: float,
    run_manifest: dict[str, Any],
    *,
    error: str = "",
) -> dict[str, Any]:
    action_required = load_action_required(workspace)
    manifest_status = str(run_manifest.get("status") or "")
    blocked = bool(action_required) and (
        return_code != 0 or not run_manifest or manifest_status in {"no_active_tasks"}
    )
    row = base_result_row(
        entry,
        source,
        workspace,
        status=(
            "blocked"
            if blocked
            else str(run_manifest.get("status") or ("failed" if return_code else "unknown"))
        ),
        return_code=return_code,
        error=(
            str(action_required.get("reason") or "")
            if blocked
            else error or str(run_manifest.get("evaluation_error") or "")
        ),
    )
    protocol_manifest = load_protocol_manifest(workspace, entry.benchmark_id)
    routes = list(protocol_manifest.get("route_rows") or [])
    selected_count = int(run_manifest.get("selected_task_count") or 0)
    total_count = int(run_manifest.get("total_task_count") or protocol_manifest.get("task_count") or 0)
    active_count = len(run_manifest.get("active_tasks") or [])
    deterministic = sum(
        1
        for route in routes
        if route.get("active")
        and str(route.get("parser_backend") or "") != "vlm_fallback"
    )
    fallback = sum(
        1
        for route in routes
        if route.get("active")
        and (
            str(route.get("decision") or "") == "fallback"
            or str(route.get("parser_backend") or "") == "vlm_fallback"
        )
    )
    evaluation = dict(run_manifest.get("evaluation_summary") or {})
    row.update(
        elapsed_seconds=round(float(elapsed_seconds), 3),
        total_tasks=total_count,
        selected_tasks=selected_count or min(total_count, len(routes)),
        active_tasks=active_count,
        formal_tasks=len(run_manifest.get("formal_evaluation_tasks") or []),
        deterministic_tasks=deterministic,
        fallback_tasks=fallback,
        disabled_tasks=len(protocol_manifest.get("disabled_tasks") or []),
        deferred_tasks=len(protocol_manifest.get("deferred_external_failure_tasks") or []),
        evaluated_samples=int(evaluation.get("total_samples") or 0),
        accuracy=(float(evaluation["accuracy"]) if "accuracy" in evaluation else None),
        valid_parse_rate=(
            float(evaluation["valid_parse_rate"])
            if "valid_parse_rate" in evaluation
            else None
        ),
        generation_failed_count=int(evaluation.get("generation_failed_count") or 0),
        parser_failure_count=int(evaluation.get("parser_failure_count") or 0),
    )
    if return_code and row["status"] in {"completed", "protocol_ready"}:
        row["status"] = "failed"
    return row


def format_benchmark_result(row: dict[str, Any]) -> str:
    selected = int(row.get("selected_tasks") or 0)
    active = int(row.get("active_tasks") or 0)
    text = f"{row['status']}: protocols={active}/{selected}"
    if row.get("evaluated_samples"):
        text += (
            f" samples={row['evaluated_samples']}"
            f" accuracy={float(row.get('accuracy') or 0.0):.1f}%"
        )
    return text


def aggregate_suite(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row.get("status") != "missing"]
    ready_statuses = {"completed", "protocol_ready", "resumed"}
    return {
        "benchmark_count": len(rows),
        "available_benchmarks": len(available),
        "missing_benchmarks": sum(1 for row in rows if row.get("status") == "missing"),
        "ready_benchmarks": sum(1 for row in available if row.get("status") in ready_statuses),
        "blocked_benchmarks": sum(1 for row in available if row.get("status") == "blocked"),
        "failed_benchmarks": sum(
            1
            for row in available
            if row.get("status") not in ready_statuses | {"blocked", "planned"}
        ),
        "selected_tasks": sum(int(row.get("selected_tasks") or 0) for row in available),
        "active_tasks": sum(int(row.get("active_tasks") or 0) for row in available),
        "deterministic_tasks": sum(
            int(row.get("deterministic_tasks") or 0) for row in available
        ),
        "fallback_tasks": sum(int(row.get("fallback_tasks") or 0) for row in available),
        "evaluated_samples": sum(int(row.get("evaluated_samples") or 0) for row in available),
    }


def suite_status(rows: list[dict[str, Any]], *, strict_missing: bool) -> str:
    if rows and all(row.get("status") == "planned" for row in rows):
        return "planned"
    failures = [
        row
        for row in rows
        if row.get("status")
        not in {"completed", "protocol_ready", "resumed", "missing", "planned"}
    ]
    missing = [row for row in rows if row.get("status") == "missing"]
    blocked = [row for row in rows if row.get("status") == "blocked"]
    failures = [row for row in failures if row.get("status") != "blocked"]
    if failures or (strict_missing and missing):
        return "failed"
    if blocked:
        return "completed_with_blocks"
    return "completed_with_skips" if missing else "completed"


def write_suite_state(
    output_root: Path,
    *,
    suite_name: str,
    suite_path: Path,
    benchmark_root: Path,
    args: argparse.Namespace,
    started_at: str,
    rows: list[dict[str, Any]],
    status: str,
) -> None:
    totals = aggregate_suite(rows)
    payload = {
        "schema_version": "provise.suite_result.v1",
        "suite": suite_name,
        "status": status,
        "started_at": started_at,
        "updated_at": utc_now(),
        "suite_file": str(suite_path),
        "benchmark_root": str(benchmark_root),
        "evaluation_model": str(args.model),
        "agent_model": str(args.agent_model),
        "parser_model": str(args.parser_model),
        "protocol_model": str(args.protocol_model),
        "max_tasks_per_benchmark": int(args.max_tasks_per_benchmark),
        "smoke_limit": int(args.smoke_limit),
        "evaluate_limit": None if args.full else int(args.evaluate_limit),
        "samples_per_benchmark": None if args.full else int(args.samples_per_benchmark),
        "build_only": bool(args.build_only),
        "totals": totals,
        "benchmarks": rows,
    }
    (output_root / "suite_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_csv(output_root / "summary.csv", rows)
    (output_root / "summary.md").write_text(
        render_summary_markdown(payload), encoding="utf-8"
    )


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(base_result_row(
        BenchmarkEntry("benchmark", "source"),
        Path("source"),
        Path("workspace"),
        status="status",
        return_code=0,
    ))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render_summary_markdown(payload: dict[str, Any]) -> str:
    totals = payload["totals"]
    lines = [
        f"# {payload['suite']}",
        "",
        f"- Status: `{payload['status']}`",
        f"- Evaluation model: `{payload['evaluation_model']}`",
        f"- Available benchmarks ready: {totals['ready_benchmarks']}/{totals['available_benchmarks']}",
        f"- Blocked by data or interface: {totals['blocked_benchmarks']}",
        f"- Active protocols: {totals['active_tasks']}/{totals['selected_tasks']} selected tasks",
        f"- Deterministic / VLM fallback: {totals['deterministic_tasks']} / {totals['fallback_tasks']}",
        f"- Evaluated samples: {totals['evaluated_samples']}",
        "",
        "| Benchmark | Family | Status | Protocols | Deterministic | Fallback | Samples | Accuracy | Valid parse |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["benchmarks"]:
        accuracy = "-" if row.get("accuracy") is None else f"{float(row['accuracy']):.1f}%"
        valid = (
            "-"
            if row.get("valid_parse_rate") is None
            else f"{float(row['valid_parse_rate']):.1f}%"
        )
        lines.append(
            f"| {row['benchmark']} | {row.get('family') or '-'} | {row['status']} | "
            f"{row.get('active_tasks', 0)}/{row.get('selected_tasks', 0)} | "
            f"{row.get('deterministic_tasks', 0)} | {row.get('fallback_tasks', 0)} | "
            f"{row.get('evaluated_samples', 0)} | {accuracy} | {valid} |"
        )
    lines.append("")
    return "\n".join(lines)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
