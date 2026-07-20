# Protocol Pool

This directory publishes one independently versioned ProVisE protocol artifact
per benchmark. Each artifact contains only the audited benchmark configuration,
versioned build manifest, and generated protocol definitions. Benchmark records,
media, smoke images, raw Agent prompts, and credentials are intentionally excluded.

| Artifact | Protocol composition |
|---|---|
| `spatialgen_bench` | Manual x11 + Agentic x3 |
| `embspatial` | Build x1 |
| `omnispatial` | Fallback x5 |
| `q_spatial_plus` | Build x3 |
| `robospatial_home` | Build x2 + Reuse x1 |
| `sat` | Build x5 + Fallback x3 |
| `roboafford` | Reuse x3 |

## Data Layout

The published YAML uses portable paths relative to its artifact directory. Put
or symlink the corresponding normalized ProVisE package at:

```text
configs/protocols/<artifact>/benchmark/
  data.jsonl
  assets/
```

The normalized package must match the benchmark and metric contract recorded in
the protocol. Data preparation is deliberately separate from the protocol so
dataset licenses and large media are not redistributed here.

Evaluate a model with the fixed protocol:

```bash
provise evaluate \
  --protocol configs/protocols/embspatial \
  --model gpt-image-2
```

The manifest stores the SHA-256 hash of the benchmark YAML. ProVisE rejects
edited YAML rather than silently changing the protocol between model runs.
