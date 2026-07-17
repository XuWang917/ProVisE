# Protocol Catalog

This directory contains global, benchmark-independent protocol specifications
loaded by the ProVisE builder and evaluation runtime. It is the catalog of
available protocol building blocks, not the storage location for protocols
generated for one benchmark.

Two kinds of specification belong here:

- **Reusable protocols** provide a stable visual contract, one or more prompt
  variants, parser configuration, and an output that can be matched to a
  benchmark metric. `agentic_point_marker` is one example.
- **Runtime adapters** provide a shared execution envelope for task-specific
  contracts. `agentic_parser_ops_protocol` executes the prompt and typed Parser
  Ops pipeline embedded in an Agent-generated benchmark artifact; it is not an
  independently reusable task protocol. These entries declare
  `catalog_role: runtime_adapter` and are not shown to the Agent as reuse
  candidates.

Benchmark-specific outputs belong under `configs/protocols/<benchmark>/`:

```text
configs/protocols/<benchmark>/
  configs/      selected routes and evaluation contract
  generated/    task-specific Agent-built protocol definitions
```

## Catalog Rules

- Keep specifications independent of benchmark records and local file paths.
- Do not store raw Agent responses, smoke images, results, or credentials here.
- Pair reusable protocols with a registered runtime implementation and focused
  parser tests.
- Keep task-specific prompts and Parser Ops pipelines in the owning benchmark
  artifact unless they are deliberately generalized and reviewed for reuse.
