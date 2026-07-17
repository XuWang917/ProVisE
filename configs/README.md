# Configuration Layout

Only reusable or published configuration belongs under `configs/`. Runtime
builds, Agent responses, smoke outputs, and temporary benchmark mappings belong
under `outputs/`.

| Directory | Purpose |
|---|---|
| `benchmark_suites/` | Validated benchmarks that can be run as one batch. Each entry resolves to an independent build workspace. |
| `protocol_specs/` | Global protocol catalog: benchmark-independent reusable templates and runtime adapters loaded by the builder and evaluator. |
| `protocols/` | Published benchmark-specific protocol artifacts. Each benchmark keeps its selected routes, versioned config, manifest, and generated definitions together. |

Test-only benchmark mappings and media live under `tests/fixtures/`; downloaded
benchmark data is not stored in this repository.

The two protocol directories have different ownership boundaries. A spec under
`protocol_specs/` defines a general visual response and readout implementation;
it must not contain benchmark samples, task-specific build outputs, or run
results. An artifact under `protocols/<benchmark>/` records how those global
building blocks and any benchmark-specific generated protocols are assembled
for one fixed evaluation.
