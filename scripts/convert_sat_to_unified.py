#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_sat_output() -> str:
    return str(PROJECT_ROOT / "benchmarks" / "sat" / "sat.jsonl")


def default_assets_dir() -> str:
    return str(PROJECT_ROOT / "benchmarks" / "sat" / "assets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SAT parquet to GenBench UnifiedSample v1")
    parser.add_argument("--input", required=True, help="Path to SAT parquet file")
    parser.add_argument("--output", default=default_sat_output(), help="Output UnifiedSample JSONL")
    parser.add_argument("--assets-dir", default=default_assets_dir(), help="Directory for extracted SAT images")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="Optional conversion limit")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic choice shuffling")
    parser.add_argument("--no-shuffle-choices", action="store_true", help="Keep the raw SAT choice order")
    parser.add_argument("--overwrite-assets", action="store_true", help="Rewrite extracted image files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    assets_dir = Path(args.assets_dir)

    rows = load_rows(input_path)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    task_counts: Dict[str, int] = {}
    image_count = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, row in enumerate(rows):
            converted = convert_row(
                row,
                idx,
                args.split,
                assets_dir,
                args.overwrite_assets,
                shuffle_choices=not args.no_shuffle_choices,
                seed=args.seed,
            )
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            task = converted["task"]
            task_counts[task] = task_counts.get(task, 0) + 1
            image_count += len(converted["input"]["media"])

    stats_path = output_path.with_name(f"{output_path.stem}_stats.json")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "benchmark": "sat",
                "total": len(rows),
                "tasks": task_counts,
                "images": image_count,
                "source": portable_path(input_path),
                "assets_dir": portable_path(assets_dir),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"converted: {len(rows)}")
    print(f"images:    {image_count}")
    print(f"output:    {output_path}")
    print(f"assets:    {assets_dir}")
    print(f"stats:     {stats_path}")
    return 0


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"SAT parquet not found: {path}")

    try:
        import pyarrow.parquet as pq
    except ImportError:
        pq = None

    if pq is not None:
        return pq.read_table(path).to_pylist()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Reading SAT parquet requires either pyarrow or datasets. "
            "Install one of them, or run this converter in an environment that already has it."
        ) from exc

    dataset = load_dataset("parquet", data_files=str(path))["train"]
    return [dict(row) for row in dataset]


def convert_row(
    row: Dict[str, Any],
    idx: int,
    split: str,
    assets_dir: Path,
    overwrite_assets: bool,
    shuffle_choices: bool,
    seed: int,
) -> Dict[str, Any]:
    sample_id = f"sat_{idx:05d}"
    answers = [str(answer) for answer in row.get("answers") or []]
    ordered_answers = shuffle_answers(answers, sample_id, seed) if shuffle_choices else list(answers)
    choices = build_choices(ordered_answers)
    answer_label = find_answer_label(str(row.get("correct_answer", "")), choices)
    media = extract_media(row.get("image_bytes") or [], sample_id, assets_dir, overwrite_assets)
    question_type = str(row.get("question_type") or "choice_qa")

    return {
        "schema_version": "genbench.v1",
        "id": sample_id,
        "benchmark": "sat",
        "task": question_type,
        "split": split,
        "input": {
            "type": "multi_image" if len(media) > 1 else "image",
            "media": media,
        },
        "question": str(row.get("question", "")),
        "answer": answer_label or str(row.get("correct_answer", "")),
        "answer_type": "choice",
        "choices": choices,
        "evaluation": {"metric": "accuracy"},
        "metadata": {
            "source": "SAT",
            "question_type": question_type,
            "correct_answer": str(row.get("correct_answer", "")),
            "answers_original_order": answers,
            "choices_shuffled": shuffle_choices,
            "image_count": len(media),
        },
    }


def build_choices(answers: Iterable[str]) -> List[Dict[str, str]]:
    choices = []
    for idx, answer in enumerate(answers):
        choices.append({"label": default_label(idx), "text": str(answer)})
    return choices


def shuffle_answers(answers: List[str], sample_id: str, seed: int) -> List[str]:
    shuffled = list(answers)
    if len(shuffled) <= 1:
        return shuffled
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    rng.shuffle(shuffled)
    return shuffled


def default_label(idx: int) -> str:
    if 0 <= idx < 26:
        return chr(ord("A") + idx)
    return str(idx + 1)


def find_answer_label(correct_answer: str, choices: List[Dict[str, str]]) -> str:
    target = normalize_answer_text(correct_answer)
    for choice in choices:
        if normalize_answer_text(choice.get("text", "")) == target:
            return choice["label"]
    return ""


def normalize_answer_text(text: Any) -> str:
    return " ".join(str(text).strip().lower().split())


def extract_media(
    image_entries: Iterable[Any],
    sample_id: str,
    assets_dir: Path,
    overwrite_assets: bool,
) -> List[Dict[str, str]]:
    media = []
    for idx, entry in enumerate(image_entries, start=1):
        image_bytes = normalize_image_bytes(entry)
        ext = infer_image_ext(image_bytes)
        rel_path = Path("images") / f"{sample_id}_{idx}{ext}"
        output_path = assets_dir / rel_path
        if overwrite_assets or not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(image_bytes)
        media.append(
            {
                "type": "image",
                "path": rel_path.as_posix(),
                "role": "primary" if idx == 1 else "view",
                "label": f"Image {idx}",
            }
        )
    return media


def normalize_image_bytes(entry: Any) -> bytes:
    if isinstance(entry, bytes):
        return entry
    if isinstance(entry, bytearray):
        return bytes(entry)
    if isinstance(entry, dict):
        for key in ("bytes", "data"):
            value = entry.get(key)
            if isinstance(value, bytes):
                return value
            if isinstance(value, bytearray):
                return bytes(value)
    raise TypeError(f"Unsupported SAT image entry type: {type(entry)!r}")


def infer_image_ext(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


if __name__ == "__main__":
    raise SystemExit(main())
