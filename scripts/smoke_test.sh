#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python}"

"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

missing = [
    name
    for name, module in {
        "cv2": "opencv-python-headless",
        "numpy": "numpy",
        "PIL": "pillow",
        "yaml": "pyyaml",
    }.items()
    if importlib.util.find_spec(name) is None
]
if missing:
    print("Missing smoke-test dependencies:", ", ".join(missing), file=sys.stderr)
    print('Install them with: python -m pip install -e ".[dev]"', file=sys.stderr)
    raise SystemExit(1)
PY

"$PYTHON_BIN" scripts/run_protocol_eval.py \
  --model mock-label-a \
  --tasks choice_qa \
  --limit 1 \
  --benchmark-config tests/fixtures/smoke_choice/benchmark.yaml \
  --output outputs/smoke_choice_label_code \
  --no-reuse

"$PYTHON_BIN" scripts/run_protocol_eval.py \
  --model mock-label-a \
  --tasks choice_qa \
  --limit 1 \
  --benchmark-config tests/fixtures/smoke_choice/generic_choice.yaml \
  --output outputs/smoke_choice_dynamic_label_code \
  --no-reuse

echo ""
echo "Smoke test finished. Inspect:"
echo "  outputs/smoke_choice_label_code/summary.json"
echo "  outputs/smoke_choice_dynamic_label_code/summary.json"
