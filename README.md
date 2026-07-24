<div align="center">
  <h1><img src="assets/provise-mark.svg" alt="ProVisE logo" height="50" align="absmiddle">&nbsp; Show, Don't Tell</h1>
  <p><img src="assets/provise-subtitle.svg" alt="Evaluating Spatial Cognition in Generative Pixels Rather Than LLM Text" width="900"></p>
  <p>
    <a href="https://arxiv.org/abs/2607.21072"><img src="https://img.shields.io/badge/arXiv-2607.21072-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="arXiv:2607.21072"></a>
    <a href="https://zju-omniai.github.io/ProVisE/"><img src="https://img.shields.io/badge/-Project_Page-0F5354?style=for-the-badge&amp;logo=googlechrome&amp;logoColor=white" alt="Project page"></a>
    <a href="https://huggingface.co/datasets/wx91726/SpatialGen-Bench"><img src="https://img.shields.io/badge/HF-SpatialGen--Bench-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=FFD21E&amp;labelColor=3A3B45" alt="SpatialGen-Bench on Hugging Face"></a>
    <a href="https://github.com/ZJU-OmniAI/ProVisE"><img src="https://img.shields.io/badge/-Code-171B1F?style=for-the-badge&amp;logo=github&amp;logoColor=white" alt="Code"></a>
  </p>
  <p>
    <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&amp;logoColor=white" alt="Python 3.10+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-2F855A" alt="Apache-2.0 license"></a>
  </p>
  <p><img src="assets/provise-quote.svg" alt="When words fall short, images give form to spatial intent. From Xici Zhuan, The Book of Changes." width="720"></p>
</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/199f556c-d1be-402a-88ae-2ddb91965c8c" controls width="80%"></video>
</div>

## 📖 Overview

Existing spatial benchmarks usually require coordinates, option labels, or textual descriptions.
This creates an answer-interface mismatch for image-generation models: they can express spatial judgments by pointing, marking, masking, or drawing in pixel space, but those visual answers fall outside the original evaluator.

**ProVisE** changes only the response interface.
A task-aware router assigns a visual protocol whose guidance prompt and parser are fixed before generation.
The model produces a protocol-constrained visual answer, the parser converts it into the required structured prediction, and the original benchmark metric scores the result.
Text-output VLMs continue to answer in the original answer space, enabling comparison under shared task semantics and metrics.

```text
benchmark task
  -> task-aware protocol routing
  -> protocol-constrained visual response
  -> structured prediction
  -> original benchmark metric
```

Building on ProVisE, **SpatialGen-Bench** contains 470 curated samples across 14 subtasks and four capability levels: perception, understanding, reasoning, and interaction.
The study finds complementary strengths: image-generation models are competitive when judgments can be externalized in pixels, while text-output VLMs remain stronger in complex spatial reasoning.
ProVisE further automates adaptation to new benchmarks through protocol reuse, construction from Parser Ops, and an explicitly labeled fallback when deterministic readout is unavailable.

## 🧪 Supported Benchmarks

Each validated benchmark has an independent artifact in the [Protocol Pool](configs/protocols/README.md).

| Benchmark | Tasks | Protocol routes | Official resource | Protocol |
|:---:|:---:|:---:|:---:|:---:|
| [SpatialGen-Bench](https://huggingface.co/datasets/wx91726/SpatialGen-Bench) | 14 | Manual 11 + Agentic 3 | <a href="https://huggingface.co/datasets/wx91726/SpatialGen-Bench"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="" width="15" align="absmiddle"> Hugging Face</a> | [View](configs/protocols/spatialgen_bench) |
| [EmbSpatial-Bench](https://github.com/mengfeidu/EmbSpatial-Bench) | 1 | Build 1 | <a href="https://github.com/mengfeidu/EmbSpatial-Bench"><img src="https://cdn.simpleicons.org/github/181717/FFFFFF" alt="" width="15" align="absmiddle"> GitHub</a> | [View](configs/protocols/embspatial) |
| [OmniSpatial](https://huggingface.co/datasets/qizekun/OmniSpatial) | 5 | Fallback 5 | <a href="https://huggingface.co/datasets/qizekun/OmniSpatial"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="" width="15" align="absmiddle"> Hugging Face</a> | [View](configs/protocols/omnispatial) |
| [Q-Spatial+](https://huggingface.co/datasets/andrewliao11/Q-Spatial-Bench) | 3 | Build 3 | <a href="https://huggingface.co/datasets/andrewliao11/Q-Spatial-Bench"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="" width="15" align="absmiddle"> Hugging Face</a> | [View](configs/protocols/q_spatial_plus) |
| [RoboSpatial-Home](https://huggingface.co/datasets/chanhee-luke/RoboSpatial-Home) | 3 | Build 2 + Reuse 1 | <a href="https://huggingface.co/datasets/chanhee-luke/RoboSpatial-Home"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="" width="15" align="absmiddle"> Hugging Face</a> | [View](configs/protocols/robospatial_home) |
| [SAT](https://github.com/arijitray1993/SAT) | 8 | Build 5 + Fallback 3 | <a href="https://github.com/arijitray1993/SAT"><img src="https://cdn.simpleicons.org/github/181717/FFFFFF" alt="" width="15" align="absmiddle"> GitHub</a> | [View](configs/protocols/sat) |
| [RoboAfford](https://github.com/tyb197/RoboAfford) | 3 | Reuse 3 | <a href="https://github.com/tyb197/RoboAfford"><img src="https://cdn.simpleicons.org/github/181717/FFFFFF" alt="" width="15" align="absmiddle"> GitHub</a> | [View](configs/protocols/roboafford) |

<p>🔄 <em>Continuously updated with newly validated spatial benchmarks and protocol artifacts.</em></p>

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/ZJU-OmniAI/ProVisE.git
cd ProVisE

conda create -n provise python=3.10 -y
conda activate provise
pip install -e ".[agentic]"
```

### 2. Configure OpenAI

Set the API key in your shell or place the same variable in a local `.env` file:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

Protocol construction defaults to `gpt-5.4` for the agent and parser, with `gpt-image-2` for visual smoke validation.

### 3. Run SpatialGen-Bench

```bash
hf download wx91726/SpatialGen-Bench \
  --repo-type dataset \
  --local-dir benchmarks/SpatialGen-Bench

provise run \
  --source benchmarks/SpatialGen-Bench \
  --model gpt-image-2
```

The first command downloads the public 470-sample benchmark from
[Hugging Face](https://huggingface.co/datasets/wx91726/SpatialGen-Bench).
ProVisE then normalizes the package, constructs and validates task protocols,
and evaluates a bounded pilot. Add `--full` after the pilot succeeds.
For another benchmark, replace `--source` with its downloaded directory or a
normalized ProVisE package.

## 🗃️ Protocol Pool

For a fair model comparison, build one versioned protocol artifact and reuse it across every evaluated model:

```bash
provise build --source /path/to/benchmark

provise evaluate \
  --protocol outputs/agentic_runs/<benchmark>/<run> \
  --model gpt-image-2 \
  --full
```

The build command prints the protocol directory consumed by `provise evaluate`.
To process a directory of downloaded benchmarks, run the suite command:

```bash
provise suite \
  --benchmark-root /path/to/downloaded/benchmarks \
  --model gpt-image-2
```

Each benchmark receives its own protocol directory; missing datasets are reported separately and are never counted as model failures.
Audited artifacts live in [`configs/protocols`](configs/protocols/README.md):

```text
configs/protocols/<benchmark>/
  configs/       benchmark routes, prompts, readout, and metric contract
  generated/     task-specific protocol definitions
```

Benchmark media is not redistributed.
To evaluate a published protocol, place or link its normalized package under `benchmark/`; see the [Protocol Pool documentation](configs/protocols/README.md).

## 🧩 Extending ProVisE

🤝 **Issues and pull requests are welcome.** Use [issues](https://github.com/ZJU-OmniAI/ProVisE/issues) for bug reports, benchmark requests, and proposed public interfaces; submit focused PRs for benchmark adapters, visual protocols, or Parser Ops.

| Contribution | Location | Required contract |
|---|---|---|
| Benchmark&nbsp;adapter | `benchmark.yaml`,&nbsp;`data.jsonl`,&nbsp;and&nbsp;`assets/` | Task/media mapping, answer schema, metric contract, and a smoke fixture |
| Visual&nbsp;protocol | [`configs/protocol_specs`](configs/protocol_specs/README.md) | Benchmark-independent response contract, parser, output kind, compatible metrics, and focused tests |
| Parser&nbsp;Op | [`provise/parser_ops`](provise/parser_ops) | Typed inputs, parameters, deterministic behavior, and unit tests |

Start with the [Unified Sample Schema](docs/unified_sample_schema.md) and [Agentic Benchmark Adapter](docs/agentic_benchmark_adapter.md).
Before [submitting a PR](https://github.com/ZJU-OmniAI/ProVisE/compare), install the contributor dependencies and run the repository checks:

```bash
pip install -e ".[agentic,dev]"
ruff check .
pytest -q
```

## 🏗️ Repository Layout

```text
ProVisE/
├── assets/                         README figures and branding
├── configs/
│   ├── benchmark_suites/           Validated multi-benchmark run definition
│   ├── protocol_specs/             Global protocol catalog and runtime adapters
│   └── protocols/                  Published benchmark protocol artifacts
├── docs/                           Schemas and benchmark-adapter guides
├── provise/
│   ├── benchmark/                  Ingestion, validation, and sample contracts
│   ├── commands/                   CLI workflow implementations
│   ├── evaluation/                 Evaluation runtime, metrics, and summaries
│   ├── models/                     Image-generation and VLM adapters
│   ├── parser_ops/                 Typed visual readout operators
│   ├── protocol_agent/             Protocol planning and contract compilation
│   ├── protocols/                  Executable protocols and registry
│   ├── cli.py                      `provise` command-line entry point
│   └── reporting.py                Terminal progress and status reporting
├── scripts/                        Conversion and low-level utility scripts
└── tests/                          Unit, integration, and smoke tests
    └── fixtures/                   Minimal benchmark packages for testing
```

See the [configuration layout](configs/README.md) for the ownership and output rules of each directory.
The installed `provise` command is the supported user interface.
Files under `scripts/` are retained only for benchmark conversion and low-level smoke or evaluation workflows.
Downloaded benchmarks and runtime outputs are intentionally not tracked.

## 🙏 Acknowledgements

We sincerely appreciate [CountBench](https://teaching-clip-to-count.github.io/), [BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK), [EgoOrientBench](https://huggingface.co/datasets/jhCOR/EgoOrientBench), [VSR](https://github.com/cambridgeltl/visual-spatial-reasoning), [ViewSpatial-Bench](https://huggingface.co/datasets/lidingm/ViewSpatial-Bench), [MindCube](https://huggingface.co/datasets/MLL-Lab/MindCube), [VisWorld-Eval](https://github.com/thuml/Reasoning-Visual-World), [RoboAfford-Eval](https://huggingface.co/datasets/tyb197/RoboAfford-Eval), [ShareRobot-Bench](https://huggingface.co/datasets/BAAI/ShareRobot-Bench), [PhysBench](https://huggingface.co/datasets/USC-PSI-Lab/PhysBench), [SPHERE-VLM](https://sphere-vlm.github.io/), and [RefCOCOg](https://github.com/lichengunc/refer) for their public datasets, task designs, and evaluation resources.

<a id="citation"></a>

## 📝 Citation

```bibtex
@article{wang2026showdonttell,
  title  = {Show, Don't Tell: Evaluating Spatial Cognition in Generative Pixels Rather Than LLM Text},
  author = {Wang, Xu and Yao, Kaixiang and Pan, Miao and Zhou, Xiaohe and Liu, Xuanyu and Zhang, Wenqi and Zhang, Xuhong},
  journal = {arXiv preprint arXiv:2607.21072},
  year   = {2026}
}
```

## ⚖️ License

ProVisE source code is released under the [Apache License 2.0](LICENSE).
Third-party benchmarks, models, and assets remain subject to their original licenses and terms.
