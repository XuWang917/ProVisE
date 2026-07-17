# UnifiedSample JSONL Schema v1

This schema is ProVisE's stable internal boundary. Users may pass a raw
benchmark directory to `provise build`; the ingestion stage creates this package
automatically. Benchmarks distributed with ProVisE should publish the normalized
package so subsequent runs skip ingestion and reproduce the same task contract.

Each line is one sample.

```json
{
  "schema_version": "genbench.v1",
  "id": "sample_001",
  "benchmark": "vsr",
  "task": "spatial_relation",
  "split": "test",
  "input": {
    "type": "image",
    "media": [
      {
        "type": "image",
        "path": "images/000001.jpg",
        "role": "primary"
      }
    ]
  },
  "question": "Is the dog under the table?",
  "answer": true,
  "answer_type": "boolean",
  "choices": [
    {"label": "true", "text": "True"},
    {"label": "false", "text": "False"}
  ],
  "evaluation": {
    "metric": "accuracy"
  },
  "metadata": {
    "source": "VSR",
    "relation": "under"
  }
}
```

## Package Layout

A directly reusable benchmark package has this minimal layout:

```text
benchmark/
├── benchmark.yaml
├── data.jsonl
└── assets/
```

`benchmark.yaml` is deliberately small:

```yaml
schema_version: provise.benchmark.v1
benchmark: example_benchmark
data_file: data.jsonl
benchmark_root: .
```

Metric definitions remain in each sample's `evaluation` contract because tasks
inside one benchmark may use different official metrics.

## Required Fields

`schema_version`: Use `"genbench.v1"`.

`id`: Unique sample id inside the benchmark.

`benchmark`: Benchmark name, such as `vsr`, `spatialsense`, or `nlvr2`.

`task`: Task name used by the benchmark config mapping.

`input`: Media input specification.

`question`: Original text question or statement.

`answer`: Canonical ground-truth answer used by metrics. Keep this simple when
possible, e.g. `"A"`, `true`, `3`, or a trajectory list/string.

## Input Media

`input.type` can be:

`image`: one image.

`multi_image`: multiple images, such as NLVR2 or navigation candidates.

`video_frames`: sampled video frames represented as image paths.

`video`: raw video input. This is reserved for future video-generation/video-VLM
support; current protocols generally use `video_frames`.

`input.media` is a list of media objects:

```json
{
  "type": "image",
  "path": "relative/or/absolute/path.jpg",
  "role": "primary",
  "label": "A"
}
```

`path`: Relative to `benchmark_root` in the benchmark YAML unless absolute.

`role`: Recommended roles include `primary`, `view`, `frame`, `option`,
`reference`, and `mask`.

`label`: Optional option label, e.g. `A`, `B`, `true`, `false`.

## Choices

Choices are optional but recommended for discrete tasks:

```json
[
  {"label": "A", "text": "left"},
  {"label": "B", "text": "right"}
]
```

Generic discrete-choice evaluation can use the `label_code` protocol with
`prompt_variant: choice_hstrip_from_choices`. In that mode labels are read from
`choices`; if a choice has no explicit `label`, labels fall back to `A`, `B`,
`C`, and so on.

For visual candidates, a choice can include media:

```json
{
  "label": "A",
  "text": "candidate A",
  "media": {"type": "image", "path": "options/a.jpg"}
}
```

## Evaluation

`evaluation` contains protocol-specific target assets or metric hints.

When the official ground truth exists only as an evaluation mask, `answer` uses
an internal target reference instead of serialized mask bytes:

```json
{
  "answer": {
    "type": "evaluation_target",
    "metric": "point_in_mask",
    "path": "targets/0001.png"
  },
  "answer_type": "points",
  "evaluation": {
    "metric": "point_in_mask",
    "mask_path": "targets/0001.png"
  }
}
```

The target path is used only by scoring and must not appear in `input.media`.

Examples:

```json
{
  "metric": "accuracy",
  "metric_config": {}
}
```

The ingestion agent checks its selected metric against benchmark code or
documentation while constructing the normalized package. Those temporary evidence
references are not persisted in `data.jsonl`. Use `metric: "unverified"` when the
official scoring rule cannot be recovered; ProVisE can smoke-test that task but
does not emit a formal benchmark score for it.

```json
{
  "metric": "mask_precision",
  "mask_path": "interaction/affordance/masks/0001.png"
}
```

```json
{
  "metric": "depth_ab",
  "coordinates": [[0.2, 0.4], [0.7, 0.5]]
}
```

## Developer Workflow

For a newly downloaded benchmark, run `provise build --source <path>`. The
ingestion Agent creates a declarative
mapping, the deterministic executor writes UnifiedSample JSONL, and the protocol
agent builds and smoke-validates task routes. A supported image task receives a
task-level VLM readout fallback when no deterministic Parser Ops route survives.

Write a benchmark-local converter only when the official release uses an
unsupported container or mandates preprocessing that cannot be represented by
the ingestion mapping contract. Do not add benchmark-name conditionals to the
core executor. See `docs/agentic_benchmark_adapter.md` for the extension and
acceptance contract.
