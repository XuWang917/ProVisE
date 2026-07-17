from __future__ import annotations

import sys
from typing import Sequence

from .commands.agentic import main as run_agentic_benchmark
from .commands.baseline import main as run_text_baseline
from .commands.evaluate import main as evaluate_frozen_protocol
from .commands.suite import main as run_spatial_suite


USAGE = """usage:
  provise build --source BENCHMARK
  provise evaluate --protocol BUILD --model MODEL [--full]
  provise run --source BENCHMARK --model MODEL [--full]
  provise suite --benchmark-root DIRECTORY [--model MODEL]
  provise baseline --suite-output DIRECTORY --model MODEL

Commands:
  build      Normalize a benchmark and build one validated protocol artifact.
  evaluate   Evaluate a model with an existing protocol artifact.
  run        Build a protocol, then evaluate one model with it.
  suite      Run one uniform pilot across a spatial benchmark suite.
  baseline   Evaluate a direct text-output VLM on the same pilot samples.

Protocol construction uses the fixed PROVISE_AGENT_MODEL,
PROVISE_PARSER_MODEL, and PROVISE_PROTOCOL_MODEL settings. The --model option
only selects the model under evaluation.
"""


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(USAGE)
        return 0
    command = args.pop(0)
    if command == "build":
        return run_agentic_benchmark(args, command="build")
    if command == "evaluate":
        return evaluate_frozen_protocol(args)
    if command == "suite":
        return run_spatial_suite(args)
    if command == "baseline":
        return run_text_baseline(args)
    if command != "run":
        print(f"Unknown command: {command}\n\n{USAGE}", file=sys.stderr)
        return 2
    return run_agentic_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
