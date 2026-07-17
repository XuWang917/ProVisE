#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python}"
MODEL_KEY="${PROVISE_API_SMOKE_MODEL:-gpt-image-2}"
BENCHMARK_CONFIG="${PROVISE_API_SMOKE_BENCHMARK_CONFIG:-tests/fixtures/smoke_choice/benchmark.yaml}"
if [[ -n "${PROVISE_API_SMOKE_TASKS:-}" ]]; then
  TASKS="${PROVISE_API_SMOKE_TASKS}"
else
  TASKS=""
fi
LIMIT="${PROVISE_API_SMOKE_LIMIT:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${PROVISE_API_SMOKE_OUTPUT:-outputs/api_smoke_${MODEL_KEY}_${STAMP}}"

export PROVISE_CLIP_DEVICE="${PROVISE_CLIP_DEVICE:-cpu}"

echo "OpenAI-compatible API smoke test"
echo "  model:  ${MODEL_KEY}"
echo "  config: ${BENCHMARK_CONFIG}"
echo "  tasks:  ${TASKS}"
echo "  limit:  ${LIMIT}"
echo "  output: ${OUTPUT_DIR}"

CMD=(
  "$PYTHON_BIN" scripts/run_protocol_eval.py
  --model "$MODEL_KEY"
  --benchmark-config "$BENCHMARK_CONFIG"
  --limit "$LIMIT"
  --output "$OUTPUT_DIR"
  --no-reuse
)

if [[ -n "${TASKS}" ]]; then
  CMD+=(--tasks "$TASKS")
fi

"${CMD[@]}"

echo ""
echo "Smoke test finished. Inspect:"
echo "  ${OUTPUT_DIR}/summary.json"
