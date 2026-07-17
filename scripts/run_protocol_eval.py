#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from provise.evaluation.runner import run_protocol_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Protocol-based generative benchmark evaluation")
    parser.add_argument("--model", default="mock-label-a", help="Generative model key, e.g. joyai-image")
    parser.add_argument("--tasks", default="", help="Comma-separated task list. Default: all tasks in benchmark config")
    parser.add_argument(
        "--benchmark-config",
        default="",
        metavar="YAML",
        help="Benchmark mapping YAML. Example: tests/fixtures/smoke_choice/benchmark.yaml",
    )
    # The remaining options are kept for reproducibility/debugging, but hidden
    # from the everyday CLI surface so the evaluator has one simple entry point.
    advanced = argparse.SUPPRESS
    parser.add_argument(
        "--protocol-spec-dir",
        default="configs/protocol_specs",
        help=advanced,
    )
    parser.add_argument(
        "--protocol-dir",
        default="",
        help=advanced,
    )
    parser.add_argument("--data-file", default="", help="Optional benchmark JSONL override.")
    parser.add_argument("--config", default="", help=advanced)
    parser.add_argument("--benchmark-root", default="", help="Optional benchmark asset-root override.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Sample limit per task")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Total sample budget balanced across selected tasks",
    )
    parser.add_argument("--gpu", default="", help=advanced)
    parser.add_argument("--protocol", default="", help=advanced)
    parser.add_argument("--no-reuse", action="store_true", help=advanced)
    parser.add_argument("--print-prompt", action="store_true", help=advanced)
    parser.add_argument("--progress-events", default="", help=advanced)
    parser.add_argument("--heartbeat-seconds", type=float, default=1.0, help=advanced)
    parser.add_argument("--generation-retries", type=int, default=1, help=advanced)
    parser.add_argument("--generation-retry-backoff", type=float, default=2.0, help=advanced)
    return parser.parse_args()


def main() -> int:
    return run_protocol_eval(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
