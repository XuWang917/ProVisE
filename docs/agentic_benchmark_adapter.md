# Agentic Benchmark Adapter

This document describes the automatic path from a downloaded spatial benchmark
to a versioned visual protocol and model scores.

## Inputs

The minimal user input is a benchmark directory:

```bash
provise build --source /path/to/benchmark
```

The directory may contain nested annotation files, images, and official metric
code. There is no required raw directory layout. Automatic ingestion currently
recognizes common JSON, JSONL, Parquet, and TFRecord structures. Native video or
simulator-only tasks remain outside the image-generation runtime.

A normalized package can be passed instead. It must contain `benchmark.yaml`
and a valid `genbench.v1` JSONL file. Normalized packages skip the ingestion
Agent, which makes repeated protocol construction faster and more reproducible.

## Build Stages

The terminal reports six top-level stages. Stage 4 contains the per-task loop.

### 1. Start

ProVisE resolves the source, benchmark name, output workspace, and fixed build
models. No target model is involved in protocol construction.

### 2. Inspect or Load

For raw data, the ingestion Agent receives a bounded inventory of annotation
sources, representative records, media candidates, and metric-code evidence. It
returns a declarative mapping. Python applies that mapping; the Agent does not
write or execute conversion code.

For a normalized package, this stage validates and loads the package directly.

### 3. Normalize and Validate

Python converts records to `genbench.v1`, resolves media paths, normalizes
choices and answers, and partitions incompatible answer schemas or metrics into
separate task units. It then verifies required fields, media existence, and
metric compatibility.

For declarative image path templates, Python may repair an Agent-selected value
transform only when the original transform resolves zero probe paths and exactly
one alternative registered transform resolves every probe path. The repaired
mapping and evidence are recorded in the ingestion manifest.

If normalization cannot preserve ground truth or input media, the run stops with
an `action_required.json` artifact instead of silently dropping data.

### 4. Build Protocols

Tasks are completed one at a time. For each task:

1. Select up to three representative samples with diverse input and answer
   schemas. Ground-truth answers are excluded from the Agent prompt.
2. Attach representative source images and summarize the task contract.
3. Ask the protocol Agent for one route: `reuse`, `build`, `fallback`, or
   `unsupported`.
4. Validate the response with Python. Unknown protocols, unregistered operators,
   answer leakage, answer-code layouts, incompatible metrics, and unsupported
   media are rejected mechanically.
5. Compile `build_mode=recipe` from one registered visual recipe, or
   `build_mode=parser_ops` from a typed graph of registered readout operators.
6. Run smoke generation and parsing twice on a small sample.
7. Check generation rate, readout operation, parser agreement, spatial evidence,
   and metric compatibility.
8. Allow at most one diagnostic revision. Correctness and ground truth are not
   shown to the revision Agent.
9. If deterministic readout still fails for a framework reason, validate a
   generated-image-only VLM fallback.
10. Version the accepted task or record a precise disabled/deferred reason.

This task-major order gives every task a complete result before the next task
starts and checkpoints the manifest after each completion.

#### Visual Contract

A Visual Contract is the constrained intermediate representation between the
protocol Agent and executable ProVisE code. It states what spatial evidence the
generation model should draw, which registered recipe or Parser Ops graph reads
that evidence, the structured output kind, and the compatible benchmark metric.
Python validates and compiles the contract; the Agent cannot supply or execute
arbitrary parser code.

### 5. Evaluate

`provise build` stops after protocol construction. `provise run` then evaluates
the target model. A standalone evaluation uses:

```bash
provise evaluate --protocol /path/to/build --model MODEL
```

Only tasks with a registered benchmark metric enter formal scoring. Tasks marked
with `metric: unverified` remain available for smoke validation and visible in
the manifest.

### 6. Finish

The run manifest reports benchmark coverage, active tasks, fallback tasks,
evaluation status, and compact result paths. Detailed paths and sample events
are available with `--verbose` and in the JSONL progress log.

## Protocol Decisions

### `reuse`

Select a registered protocol whose visual expression, parser output, input mode,
and metric all match the task.

### `build`

Compile a deterministic protocol:

- `build_mode=recipe`: configure a registered high-level recipe such as a point
  marker, relation-validity region, region mask, trajectory, grounded dimension,
  or state-image match.
- `build_mode=parser_ops`: provide a generation prompt and a typed graph using
  only whitelisted Parser Ops.

The compiler verifies operator input/output types and metric compatibility. The
Agent never writes Python parser code.

### `fallback`

Use a task-specific VLM readout only when deterministic operators, OCR, geometry,
and fixed similarity models are insufficient. The generation prompt must request
concrete spatial evidence and cannot encode an answer through a corner, slot,
color code, bare label, check/X, or other generic verdict symbol.

The parser sees:

- the generated image;
- the fixed visual evidence and readout contract;
- the required answer format and candidate choices;
- invalid-output conditions.

It does not see the source image or original question. The returned JSON contains
`status`, `prediction`, `evidence`, and `confidence`.

### `unsupported`

Reserved for a real runtime boundary, such as native video input, mixed
incompatible metrics in one task unit, missing ground truth, or an output type
that no available evaluator can consume. A semantic parser gap normally enters
fallback rather than unsupported.

## Parser Ops

Parser Ops are typed, registered visual readout operators. The current library
covers color segmentation, connected components, source-aware differencing,
geometry, point-in-region membership, points, masks, paths, measurements, OCR,
choice mapping, and fixed CLIP similarity. Each graph is validated before
execution and stored in the generated protocol artifact.

This library is the main extensibility boundary: adding a tested operator or
recipe expands deterministic task coverage without allowing the Agent to execute
arbitrary code.

## Smoke Gates

Each accepted protocol records:

- `generated_rate`: successful image responses over attempted samples.
- `valid_parse_rate`: valid predictions over attempted samples.
- `readout_operational_rate`: parser produced either a prediction or a
  meaningful model-noncompliance result.
- `parser_agreement_rate`: repeated parsing gave the same result.
- `spatial_evidence_rate`: fallback evidence names concrete visual marks.
- `metric_compatibility_rate`: valid parser outputs can be consumed by the
  benchmark metric.

Generation/API-only failures are marked `deferred`; they do not trigger prompt
revision. Parser or protocol failures may trigger the single allowed revision.

## Failure Attribution

Per-sample outcomes separate:

- `generation_failure`
- `parser_failure`
- `scoring_failure`
- `model_protocol_noncompliance`
- `incorrect_prediction`
- `input_missing`

This distinction prevents an API outage or parser defect from being reported as
model spatial reasoning failure.

## Fairness and Versioning

The build YAML records:

```yaml
protocol_build:
  schema_version: provise.protocol.v1
  frozen: true
  agent_model: ...
  parser_model: ...
  validation_model: ...
```

Its manifest stores a SHA-256 fingerprint. `provise evaluate` checks the
fingerprint and restores the recorded parser model. Changing the evaluated
model does not revise, reroute, or re-smoke the protocol.
