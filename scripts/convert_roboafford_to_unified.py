#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


TASK_BY_CATEGORY = {
    "object affordance": "object_affordance",
    "object reference": "object_reference",
    "spatial affordance": "spatial_affordance",
}


def default_roboafford_output() -> str:
    return str(PROJECT_ROOT / "benchmarks" / "roboafford" / "roboafford.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RoboAfford annotations to GenBench UnifiedSample v1")
    parser.add_argument("--input", required=True, help="Path to annotations_normxy.json")
    parser.add_argument("--output", default=default_roboafford_output(), help="Output UnifiedSample JSONL")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"RoboAfford annotations not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    task_counts: Dict[str, int] = {}
    category_counts: Dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(rows):
            converted = convert_row(row, idx, args.split)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            task = converted["task"]
            category = converted["metadata"]["category"]
            task_counts[task] = task_counts.get(task, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1

    stats_path = output_path.with_name(f"{output_path.stem}_stats.json")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "benchmark": "roboafford",
                "total": len(rows),
                "tasks": task_counts,
                "categories": category_counts,
                "source": portable_path(input_path),
                "assets_source": "directory containing annotations, images/, and masks/",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"converted: {len(rows)}")
    print(f"output:    {output_path}")
    print(f"stats:     {stats_path}")
    return 0


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def convert_row(row: Dict[str, Any], idx: int, split: str) -> Dict[str, Any]:
    category = str(row.get("category") or "unknown")
    task = TASK_BY_CATEGORY.get(category, slugify(category))
    image_path = image_rel_path(row.get("img", ""))
    mask_path = mask_rel_path(row.get("mask", ""))
    question = clean_question(str(row.get("question", "")))

    return {
        "schema_version": "genbench.v1",
        "id": f"roboafford_{idx:05d}",
        "benchmark": "roboafford",
        "task": task,
        "split": split,
        "input": {
            "type": "image",
            "media": [
                {
                    "type": "image",
                    "path": image_path,
                    "role": "primary",
                }
            ],
        },
        "question": question,
        "answer": mask_path,
        "answer_type": "mask",
        "choices": [],
        "evaluation": {
            "metric": "mask_precision",
            "mask_path": mask_path,
        },
        "metadata": {
            "source": "RoboAfford",
            "category": category,
            "raw_question": row.get("question", ""),
            "points": row.get("answer", []),
            "img": row.get("img", ""),
            "mask": row.get("mask", ""),
        },
    }


def clean_question(question: str) -> str:
    question = " ".join(question.strip().split())
    question = re.split(r"\s+Your answer should be formatted as\b", question, maxsplit=1)[0]
    return question.strip()


def image_rel_path(value: Any) -> str:
    text = str(value).strip().lstrip("./")
    if text.startswith("images/"):
        return text
    return f"images/{text}"


def mask_rel_path(value: Any) -> str:
    text = str(value).strip().lstrip("./")
    if text.startswith("masks/"):
        return text
    if text.startswith("mask/"):
        return "masks/" + text[len("mask/") :]
    return f"masks/{text}"


def slugify(text: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return value or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
