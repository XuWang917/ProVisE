from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import yaml

from ..evaluation.runner import run_protocol_eval, safe_name
from ..paths import protocol_spec_dir, runtime_root
from ..reporting import CONCISE_OUTPUT_ENV, compact_path


PROJECT_ROOT = runtime_root()


@dataclass(frozen=True)
class FrozenProtocol:
    config_path: Path
    manifest_path: Path
    config: dict


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = resolve_frozen_protocol(args.protocol)
    build = dict(artifact.config.get("protocol_build") or {})
    parser_model = str(build.get("parser_model") or "").strip()
    tasks = _evaluation_tasks(artifact.config, args.tasks)
    limit = None if args.full else args.limit
    output = Path(args.output).expanduser().resolve() if args.output else _default_output(
        artifact.config_path, args.model
    )
    runtime_args = argparse.Namespace(
        benchmark_config=str(artifact.config_path),
        benchmark_root="",
        config="",
        data_file="",
        generation_retries=max(0, args.generation_retries),
        generation_retry_backoff=max(0.0, args.generation_retry_backoff),
        gpu=args.gpu,
        heartbeat_seconds=args.heartbeat_seconds,
        limit=limit,
        max_samples=0 if args.full else max(0, int(args.max_samples)),
        model=args.model,
        no_reuse=False,
        output=str(output),
        print_prompt=False,
        progress_events=args.progress_events,
        protocol="",
        protocol_dir="",
        protocol_spec_dir=str(protocol_spec_dir()),
        reuse_only=args.reuse_only,
        tasks=",".join(tasks),
    )

    previous_output_mode = os.environ.get(CONCISE_OUTPUT_ENV)
    previous_parser_model = os.environ.get("PROVISE_PARSER_MODEL")
    os.environ[CONCISE_OUTPUT_ENV] = "0" if args.verbose else "1"
    if parser_model:
        os.environ["PROVISE_PARSER_MODEL"] = parser_model
    try:
        code = run_protocol_eval(runtime_args)
    finally:
        if previous_output_mode is None:
            os.environ.pop(CONCISE_OUTPUT_ENV, None)
        else:
            os.environ[CONCISE_OUTPUT_ENV] = previous_output_mode
        if previous_parser_model is None:
            os.environ.pop("PROVISE_PARSER_MODEL", None)
        else:
            os.environ["PROVISE_PARSER_MODEL"] = previous_parser_model
    if code == 0:
        print(f"Results: {compact_path(output, PROJECT_ROOT)}")
    return code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="provise evaluate",
        description="Evaluate a model with one versioned ProVisE protocol artifact.",
    )
    parser.add_argument(
        "--protocol",
        required=True,
        help="Protocol build directory or its benchmark YAML.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("PROVISE_EVALUATION_MODEL", "gpt-image-2"),
        help="Registered image generation model to evaluate.",
    )
    parser.add_argument("--tasks", default="", help="Optional comma-separated task subset.")
    parser.add_argument("--limit", type=int, default=5, help="Samples per task (default: 5).")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Total sample budget, balanced across selected tasks (0 means no total cap).",
    )
    parser.add_argument("--full", action="store_true", help="Evaluate every sample.")
    parser.add_argument("--output", default="", help="Optional result directory.")
    parser.add_argument("--gpu", default="", help=argparse.SUPPRESS)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress-events", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reuse-only",
        action="store_true",
        help="Parse and score cached generated images without loading a generation model.",
    )
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
    return parser.parse_args(list(argv) if argv is not None else None)


def resolve_frozen_protocol(value: str | Path) -> FrozenProtocol:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        candidates = sorted((path / "configs").glob("*_agentic.yaml"))
        if not candidates:
            candidates = sorted(path.glob("*_agentic.yaml"))
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one protocol YAML under {path}, found {len(candidates)}"
            )
        config_path = candidates[0]
    else:
        config_path = path
    if not config_path.is_file():
        raise FileNotFoundError(f"Protocol config does not exist: {config_path}")

    manifest_path = config_path.with_name(
        f"{config_path.stem}.agentic_manifest.json"
    )
    if not manifest_path.is_file():
        raise ValueError(
            "The protocol is not a versioned Agentic artifact: matching manifest is missing"
        )
    config_text = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text) or {}
    build = config.get("protocol_build") or {}
    if build.get("schema_version") != "provise.protocol.v1" or build.get("frozen") is not True:
        raise ValueError("The benchmark YAML is not marked as a versioned ProVisE protocol")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = str((manifest.get("protocol_artifact") or {}).get("config_sha256") or "")
    actual = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    if not expected or expected != actual:
        raise ValueError(
            "Protocol integrity check failed; rebuild instead of editing the artifact"
        )
    return FrozenProtocol(config_path, manifest_path, config)


def _evaluation_tasks(config: dict, requested: str) -> list[str]:
    tasks = dict(config.get("tasks") or {})
    if requested:
        selected = [value.strip() for value in requested.split(",") if value.strip()]
        unknown = sorted(set(selected) - set(tasks))
        if unknown:
            raise ValueError(f"Unknown protocol task(s): {unknown}")
    else:
        selected = [
            task
            for task, task_config in tasks.items()
            if task_config.get("formal_evaluation") is not False
        ]
    if not selected:
        raise ValueError("The protocol has no formally evaluable task")
    return selected


def _default_output(config_path: Path, model: str) -> Path:
    workspace = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return workspace / "evaluations" / safe_name(model) / timestamp
