# Asset Provenance and Usage Notes

The MIT License in this repository applies to the website source code and
original utility scripts. It does **not** automatically apply to files under
`assets/`.

## Project Figures and Branding

The ProVisE logo and original paper figures are copyright the Show, Don't Tell
authors. They are included to render the project website and are not licensed
under MIT unless a file or release explicitly states otherwise.

## Benchmark-Derived Examples

Representative inputs under `assets/examples/` originate from, or are derived
from, the research datasets listed below. Model-generated visual answers and
parsed overlays may remain derivative of their corresponding inputs. Users
must follow the terms of the upstream dataset and, where applicable, the
license of the original media asset.

| Task | Source | Reported upstream license/terms | Reference |
| --- | --- | --- | --- |
| Counting | CountBench | CC BY 4.0 | <https://teaching-clip-to-count.github.io/> |
| Relative depth | BLINK | Apache-2.0 | <https://huggingface.co/datasets/BLINK-Benchmark/BLINK> |
| Orientation | EgoOrientBench | MIT | <https://huggingface.co/datasets/jhCOR/EgoOrientBench> |
| Relationship | VSR | Apache-2.0 | <https://github.com/cambridgeltl/visual-spatial-reasoning> |
| Perspective | ViewSpatial-Bench | Apache-2.0 | <https://huggingface.co/datasets/lidingm/ViewSpatial-Bench> |
| Mental modeling | MindCube | MIT | <https://huggingface.co/datasets/MLL-Lab/MindCube> |
| Multi-hop and prediction | VisWorld-Eval | No explicit license identified in the checked release | <https://huggingface.co/datasets/thuml/VisWorld-Eval> |
| Affordance | RoboAfford-Eval | CC BY 4.0 | <https://huggingface.co/datasets/tyb197/RoboAfford-Eval> |
| Navigation | PhysBench | Apache-2.0 | <https://huggingface.co/datasets/USC-PSI-Lab/PhysBench> |
| Trajectory | ShareRobot-Bench | Apache-2.0 | <https://huggingface.co/datasets/BAAI/ShareRobot-Bench> |
| Object size and geometric feasibility | SPHERE / SPHERE-VLM | Consult the corresponding upstream release | Upstream project terms |
| Spatial grounding | RefCOCOg | Consult the dataset and underlying image terms | Upstream dataset terms |

The table records terms observed from the corresponding project, repository,
or dataset card; it is provenance documentation, not legal advice or a
reinterpretation of upstream rights. An upstream repository license may not
cover every image bundled by that dataset.

Before redistributing an asset independently of this research website, verify
the current upstream terms and preserve all required attribution notices.
