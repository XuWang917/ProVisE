from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..benchmark.media import resolve_input_media_paths
from ..evaluation.metrics import score_prediction
from ..models.direct_vlm import LocalInternVLDirectVLM, LocalQwenDirectVLM
from ..models.vlm import OpenAICompatibleVisionLanguageModel
from ..reporting import style_terminal


SCHEMA_VERSION = "provise.text_baseline.v1"

POINT_ANSWER_FORMAT_MARKERS = (
    "your answer should be formatted as",
    "format your answer as a list of",
    "return your answer as a list of",
    "provide your answer as a list of",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="provise baseline",
        description="Evaluate a direct text-output VLM on a protocol-suite pilot.",
    )
    parser.add_argument(
        "--suite-output",
        required=True,
        help="Existing ProVisE suite output containing benchmark pilot results.",
    )
    parser.add_argument("--model", required=True, help="Published model name or API model id.")
    parser.add_argument(
        "--backend",
        choices=("api", "qwen", "internvl"),
        default="api",
        help="Direct VLM inference backend.",
    )
    parser.add_argument("--model-path", default="", help="Local checkpoint for qwen/internvl.")
    parser.add_argument(
        "--benchmarks",
        default="",
        help="Optional comma-separated benchmark ids; defaults to every completed pilot.",
    )
    parser.add_argument("--pilot-name", default="pilot_24_total")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate every normalized sample instead of reusing pilot sample IDs.",
    )
    parser.add_argument("--output", default="", help="Output directory for this model.")
    parser.add_argument("--limit-per-benchmark", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent API requests (local backends require 1).",
    )
    parser.add_argument("--internvl-max-tiles", type=int, default=4)
    parser.add_argument("--qwen-max-pixels", type=int, default=1_003_520)
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1536,
        help="Resize larger inputs proportionally before every text VLM (default: 1536).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_text_baseline(args)
    except (OSError, ValueError) as exc:
        print(style_terminal(f"Baseline error: {exc}", tone="error"), file=sys.stderr)
        return 2


def run_text_baseline(args: argparse.Namespace, *, model: Any | None = None) -> int:
    suite_root = Path(args.suite_output).expanduser().resolve()
    if not suite_root.is_dir():
        raise FileNotFoundError(f"suite output does not exist: {suite_root}")
    selected = {value.strip() for value in args.benchmarks.split(",") if value.strip()}
    benchmarks = discover_pilot_benchmarks(suite_root, args.pilot_name, selected)
    if not benchmarks:
        raise ValueError("no completed pilot benchmark matched the request")
    workers = max(1, int(args.workers))
    if args.backend != "api" and workers != 1:
        raise ValueError("local qwen/internvl backends require --workers 1")

    output_root = (
        Path(args.output).expanduser().resolve()
        if args.output
        else suite_root
        / ("text_baselines_full" if args.full else "text_baselines")
        / slugify(args.model)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    runtime = model or create_direct_model(args)
    if workers > 1 and hasattr(runtime, "load_model"):
        runtime.load_model()

    print(style_terminal(f"Direct text baseline: {args.model}", bold=True))
    selection = "full" if args.full else args.pilot_name
    print(
        f"Backend: {args.backend} | benchmarks: {len(benchmarks)} | "
        f"selection: {selection} | workers: {workers}"
    )
    print(f"Results: {output_root}")
    started = time.monotonic()
    summaries = []
    for index, workspace in enumerate(benchmarks, 1):
        benchmark = workspace.name
        print(f"\n[Benchmark {index}/{len(benchmarks)}: {benchmark}]")
        summary = evaluate_benchmark(
            workspace,
            runtime,
            model_name=args.model,
            backend=args.backend,
            pilot_name=args.pilot_name,
            full=args.full,
            output_root=output_root / benchmark,
            media_cache_root=output_root.parent / "_media_cache" / benchmark,
            limit=args.limit_per_benchmark,
            resume=args.resume,
            max_image_side=args.max_image_side,
            workers=workers,
        )
        summaries.append(summary)
        print(
            style_terminal(
                f"{summary['correct_count']}/{summary['total_samples']} correct "
                f"({summary['accuracy']:.1f}%), valid {summary['valid_prediction_rate']:.1f}%",
                tone="success" if summary["failed_count"] == 0 else "warning",
            )
        )

    overall = aggregate_summaries(summaries)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "backend": args.backend,
        "model_path": str(args.model_path or ""),
        "suite_output": str(suite_root),
        "pilot_name": args.pilot_name,
        "selection": selection,
        "max_image_side": args.max_image_side,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "overall": overall,
        "benchmarks": summaries,
    }
    write_json(output_root / "summary.json", payload)
    write_model_summary(output_root / "summary.md", payload)
    write_comparison_summary(output_root.parent, suite_root)
    print(
        f"\nOverall: {overall['correct_count']}/{overall['total_samples']} "
        f"({overall['accuracy']:.1f}%)"
    )
    return 0 if overall["failed_count"] == 0 else 1


def discover_pilot_benchmarks(
    suite_root: Path,
    pilot_name: str,
    selected: set[str] | None = None,
) -> list[Path]:
    benchmark_root = suite_root / "benchmarks"
    workspaces = []
    for workspace in sorted(benchmark_root.iterdir() if benchmark_root.is_dir() else []):
        if not workspace.is_dir() or (selected and workspace.name not in selected):
            continue
        if (workspace / pilot_name / "summary.json").is_file():
            workspaces.append(workspace)
    missing = sorted((selected or set()) - {path.name for path in workspaces})
    if missing:
        raise ValueError(f"benchmarks do not have completed {pilot_name} pilots: {missing}")
    return workspaces


def evaluate_benchmark(
    workspace: Path,
    model: Any,
    *,
    model_name: str,
    backend: str,
    pilot_name: str,
    output_root: Path,
    media_cache_root: Path | None = None,
    full: bool = False,
    limit: int = 0,
    resume: bool = True,
    max_image_side: int = 1536,
    workers: int = 1,
) -> dict[str, Any]:
    benchmark = workspace.name
    items = load_unified_items(workspace, benchmark)
    selected_ids = list(items) if full else pilot_sample_ids(workspace / pilot_name)
    if limit > 0:
        selected_ids = selected_ids[:limit]
    selected_items = [items[item_id] for item_id in selected_ids if item_id in items]
    if len(selected_items) != len(selected_ids):
        missing = [item_id for item_id in selected_ids if item_id not in items]
        raise ValueError(f"{benchmark} pilot references missing normalized samples: {missing[:3]}")

    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "results.jsonl"
    loaded = load_jsonl(result_path) if resume else []
    existing = [
        row
        for row in loaded
        if row.get("inference_success") is True
        and row.get("parse_success") is True
        and row.get("score_computed") is True
    ]
    if len(existing) != len(loaded):
        write_jsonl(result_path, existing)
    by_id = {str(row.get("id")): row for row in existing}
    pending = [item for item in selected_items if str(item["id"]) not in by_id]
    completed_before = len(selected_items) - len(pending)
    started = time.monotonic()
    cache_root = media_cache_root or output_root / "_input_cache"

    def run_item(item: Mapping[str, Any]) -> dict[str, Any]:
        return evaluate_item(
            workspace,
            item,
            model,
            model_name=model_name,
            backend=backend,
            media_cache_root=cache_root,
            max_image_side=max_image_side,
        )

    if workers == 1:
        completed_rows = (run_item(item) for item in pending)
        for offset, row in enumerate(completed_rows, 1):
            append_jsonl(result_path, row)
            by_id[str(row["id"])] = row
            print_baseline_progress(completed_before + offset, len(selected_items), started)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_item, item) for item in pending]
            for offset, future in enumerate(concurrent.futures.as_completed(futures), 1):
                row = future.result()
                append_jsonl(result_path, row)
                by_id[str(row["id"])] = row
                print_baseline_progress(completed_before + offset, len(selected_items), started)
    if pending:
        print()
    ordered = [by_id[str(item["id"])] for item in selected_items]
    summary = summarize_rows(benchmark, ordered)
    summary.update(
        {
            "model": model_name,
            "backend": backend,
            "pilot_name": pilot_name,
            "selection": "full" if full else pilot_name,
            "sample_ids": selected_ids,
            "results": str(result_path),
        }
    )
    write_json(output_root / "summary.json", summary)
    return summary


def evaluate_item(
    workspace: Path,
    item: Mapping[str, Any],
    model: Any,
    *,
    model_name: str,
    backend: str,
    media_cache_root: Path,
    max_image_side: int,
) -> dict[str, Any]:
    image_paths = resolve_input_media_paths(dict(item), workspace)
    model_input_paths = prepare_model_inputs(
        image_paths,
        media_cache_root,
        max_image_side=max_image_side,
    )
    prompt = build_direct_prompt(item)
    row: dict[str, Any] = {
        "id": item.get("id"),
        "benchmark": item.get("benchmark"),
        "task": item.get("task"),
        "model": model_name,
        "backend": backend,
        "input_paths": image_paths,
        "model_input_paths": model_input_paths,
        "prompt": prompt,
        "ground_truth": item.get("answer"),
        "metric": (item.get("evaluation") or {}).get("metric"),
        "raw_response": "",
        "prediction": None,
        "inference_success": False,
        "parse_success": False,
        "score_computed": False,
        "score": 0.0,
        "is_correct": False,
        "error_type": "",
        "error_message": "",
    }
    try:
        missing = [path for path in model_input_paths if not Path(path).is_file()]
        if not model_input_paths or missing:
            raise FileNotFoundError(f"missing input image(s): {missing or model_input_paths}")
        raw = str(model.predict_multi(model_input_paths, prompt) or "").strip()
        row["raw_response"] = raw
        row["inference_success"] = True
        prediction = parse_direct_response(raw, item)
        if prediction is None or prediction == "":
            raise ValueError("response did not match the required answer schema")
        row["prediction"] = prediction
        row["parse_success"] = True
        evaluation = item.get("evaluation") or {}
        scored = score_prediction(
            str(evaluation.get("metric") or "accuracy"),
            prediction,
            item,
            workspace,
            evaluation.get("metric_config") or {},
        )
        row["score_computed"] = True
        row["score"] = scored.score
        row["is_correct"] = scored.is_correct
        row["score_details"] = scored.extra
    except Exception as exc:
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)
    return row


def build_direct_prompt(item: Mapping[str, Any]) -> str:
    question = str(item.get("question") or "").strip()
    choices = list(item.get("choices") or [])
    answer_type = str(item.get("answer_type") or "text").lower()
    if answer_type in {"points", "point", "coordinates"}:
        question = sanitize_point_question(question)
    lines = [
        "Answer the spatial question using only the supplied image or ordered images.",
        f"Question: {question}",
    ]
    if choices:
        lines.append("Choices:")
        for choice in choices:
            if isinstance(choice, Mapping):
                lines.append(f"{choice.get('label')}. {choice.get('text')}")
            else:
                lines.append(str(choice))
    if answer_type in {"number", "scalar", "measurement"}:
        rule = 'Return only JSON: {"value": NUMBER, "unit": "UNIT"}.'
    elif answer_type in {"points", "point", "coordinates"}:
        rule = (
            'Return exactly one most-confident point as JSON: {"points": [[x, y]]}. '
            "Use decimal coordinates strictly between 0 and 1, for example "
            '[0.42, 0.67], not [420, 670]. Do not use pixel coordinates or a '
            "0-to-1000 coordinate system."
        )
    elif answer_type in {"bbox", "box", "bounding_box"}:
        rule = 'Return only JSON: {"bbox": [x1, y1, x2, y2]} using normalized coordinates.'
    elif choices:
        labels = ", ".join(str(choice.get("label")) for choice in choices if isinstance(choice, Mapping))
        rule = f"Return only one choice label exactly as listed ({labels})."
    elif answer_type == "choice":
        rule = "Return only the option label printed in the image, such as A, B, C, or D."
    elif answer_type == "boolean":
        rule = "Return only yes or no."
    else:
        rule = "Return only the final short answer, without explanation."
    lines.extend([rule, "Do not include reasoning or Markdown."])
    return "\n".join(lines)


def sanitize_point_question(question: str) -> str:
    """Remove benchmark-specific point-output suffixes before adding our schema."""
    lowered = question.lower()
    offsets = [lowered.find(marker) for marker in POINT_ANSWER_FORMAT_MARKERS]
    offsets = [offset for offset in offsets if offset >= 0]
    if not offsets:
        return question.strip()
    return question[: min(offsets)].strip()


def prepare_model_inputs(
    image_paths: list[str],
    cache_root: Path,
    *,
    max_image_side: int,
) -> list[str]:
    if max_image_side <= 0:
        return image_paths
    from PIL import Image

    prepared = []
    for index, raw_path in enumerate(image_paths):
        source = Path(raw_path)
        if not source.is_file():
            prepared.append(str(source))
            continue
        with Image.open(source) as image:
            width, height = image.size
            if max(width, height) <= max_image_side:
                prepared.append(str(source))
                continue
            cache_root.mkdir(parents=True, exist_ok=True)
            target = cache_root / f"{source.stem}_{index}_{max_image_side}.jpg"
            if not target.is_file():
                resized = image.convert("RGB")
                resized.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
                resized.save(target, format="JPEG", quality=95, subsampling=0)
            prepared.append(str(target))
    return prepared


def parse_direct_response(raw: str, item: Mapping[str, Any]) -> Any:
    text = strip_code_fence(str(raw or "")).strip()
    if not text:
        return None
    answer_type = str(item.get("answer_type") or "text").lower()
    choices = list(item.get("choices") or [])

    if answer_type in {"points", "point", "coordinates", "bbox", "box", "bounding_box"}:
        structured = parse_structured(text)
        if isinstance(structured, Mapping):
            keys = ("points", "point", "bbox", "box", "value")
            structured = next((structured[key] for key in keys if key in structured), None)
        if answer_type in {"points", "point", "coordinates"}:
            return normalize_thousand_scale_points(structured)
        return structured
    if answer_type in {"number", "scalar", "measurement"}:
        structured = parse_structured(text)
        return structured if isinstance(structured, Mapping) else text
    if answer_type == "boolean":
        match = re.search(r"(?i)\b(yes|no|true|false)\b", text)
        if match:
            return {"true": "yes", "false": "no"}.get(match.group(1).lower(), match.group(1).lower())
    if choices:
        parsed = parse_choice(text, choices)
        if parsed is not None:
            return parsed
    if answer_type == "choice":
        ground_truth = str(item.get("answer") or "").strip()
        letter_match = re.search(r"(?i)\b([A-Z])\b", text)
        if ground_truth.isdigit() and letter_match:
            return str(ord(letter_match.group(1).upper()) - ord("A"))
        answer_match = re.search(r"(?i)(?:answer|option)\s*(?:is|:)?\s*([A-Za-z0-9_-]+)", text)
        if answer_match:
            return answer_match.group(1)
        token_match = re.search(r"[A-Za-z0-9_-]+", text)
        return token_match.group(0) if token_match else None
    return text.splitlines()[0].strip().rstrip(".")


def parse_choice(text: str, choices: list[Any]) -> str | None:
    normalized = text.strip().lower().rstrip(".")
    rows = []
    for choice in choices:
        if isinstance(choice, Mapping):
            rows.append((str(choice.get("label") or ""), str(choice.get("text") or "")))
        else:
            rows.append((str(choice), str(choice)))
    for label, _ in rows:
        if normalized == label.lower():
            return label
    for label, _ in rows:
        if label and re.search(rf"(?i)\b{re.escape(label)}\b", text):
            return label
    matches = [label for label, value in rows if value and value.lower() in normalized]
    return matches[0] if len(set(matches)) == 1 else None


def parse_structured(text: str) -> Any:
    candidates = [text]
    for opening, closing in (("{", "}"), ("[", "]"), ("(", ")")):
        start, end = text.find(opening), text.rfind(closing)
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            try:
                return ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                continue
    point_pairs = re.findall(
        r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]",
        text,
    )
    if not point_pairs:
        point_pairs = re.findall(
            r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)",
            text,
        )
    return [[float(x), float(y)] for x, y in point_pairs] or None


def normalize_thousand_scale_points(value: Any) -> Any:
    """Bridge common 0-1000 grounding coordinates to the requested 0-1 scale."""
    if not isinstance(value, (list, tuple)):
        return value

    def coordinate(raw: Any) -> Any:
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return raw
        numeric = float(raw)
        return numeric / 1000.0 if 1.0 < numeric <= 1000.0 else numeric

    if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
        return [coordinate(item) for item in value]
    return [
        [coordinate(item) for item in row] if isinstance(row, (list, tuple)) else row
        for row in value
    ]


def strip_code_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def load_unified_items(workspace: Path, benchmark: str) -> dict[str, dict[str, Any]]:
    candidates = [workspace / f"{benchmark}.unified.jsonl", workspace / "data.jsonl"]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"normalized data not found for {benchmark}")
    return {str(row["id"]): row for row in load_jsonl(source)}


def pilot_sample_ids(pilot_root: Path) -> list[str]:
    summary_path = pilot_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"pilot summary does not exist: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sample_ids = []
    for task in summary.get("tasks") or []:
        result_path = pilot_root / str(task) / "results.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"pilot task result does not exist: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        sample_ids.extend(str(row["id"]) for row in result.get("detailed_results") or [])
    if not sample_ids:
        raise ValueError(f"pilot contains no sample ids: {pilot_root}")
    return sample_ids


def summarize_rows(benchmark: str, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task") or "unknown")].append(row)
    tasks = [summarize_group(task, values) for task, values in by_task.items()]
    return {"benchmark": benchmark, **summarize_group("overall", rows), "tasks": tasks}


def summarize_group(name: str, rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    valid = sum(bool(row.get("parse_success")) for row in rows)
    correct = sum(bool(row.get("is_correct")) for row in rows)
    failed = sum(not bool(row.get("inference_success")) for row in rows)
    return {
        "task": name,
        "total_samples": total,
        "valid_prediction_count": valid,
        "valid_prediction_rate": 100.0 * valid / total if total else 0.0,
        "correct_count": correct,
        "accuracy": 100.0 * correct / total if total else 0.0,
        "mean_score": sum(float(row.get("score") or 0.0) for row in rows) / total if total else 0.0,
        "failed_count": failed,
    }


def aggregate_summaries(summaries: list[Mapping[str, Any]]) -> dict[str, Any]:
    total = sum(int(row.get("total_samples") or 0) for row in summaries)
    correct = sum(int(row.get("correct_count") or 0) for row in summaries)
    valid = sum(int(row.get("valid_prediction_count") or 0) for row in summaries)
    failed = sum(int(row.get("failed_count") or 0) for row in summaries)
    weighted_score = sum(
        float(row.get("mean_score") or 0.0) * int(row.get("total_samples") or 0)
        for row in summaries
    )
    return {
        "total_samples": total,
        "correct_count": correct,
        "accuracy": 100.0 * correct / total if total else 0.0,
        "valid_prediction_count": valid,
        "valid_prediction_rate": 100.0 * valid / total if total else 0.0,
        "mean_score": weighted_score / total if total else 0.0,
        "failed_count": failed,
    }


def create_direct_model(args: argparse.Namespace) -> Any:
    if args.backend == "api":
        return OpenAICompatibleVisionLanguageModel(
            args.model,
            timeout=args.timeout,
            max_tokens=args.max_new_tokens,
        )
    if not args.model_path:
        raise ValueError(f"--model-path is required for backend={args.backend}")
    if args.backend == "qwen":
        return LocalQwenDirectVLM(
            args.model_path,
            max_new_tokens=args.max_new_tokens,
            max_pixels=args.qwen_max_pixels,
        )
    return LocalInternVLDirectVLM(
        args.model_path,
        max_new_tokens=args.max_new_tokens,
        max_tiles_per_image=args.internvl_max_tiles,
    )


def write_model_summary(path: Path, payload: Mapping[str, Any]) -> None:
    overall = payload["overall"]
    lines = [
        f"# {payload['model']}",
        "",
        f"- Backend: `{payload['backend']}`",
        f"- Samples: {overall['total_samples']}",
        f"- Accuracy: {overall['accuracy']:.1f}%",
        f"- Valid predictions: {overall['valid_prediction_rate']:.1f}%",
        "",
        "| Benchmark | Samples | Accuracy | Valid | Mean score |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["benchmarks"]:
        lines.append(
            f"| {row['benchmark']} | {row['total_samples']} | {row['accuracy']:.1f}% | "
            f"{row['valid_prediction_rate']:.1f}% | {row['mean_score']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_summary(root: Path, suite_root: Path) -> None:
    models = []
    for path in sorted(root.glob("*/summary.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if row.get("schema_version") == SCHEMA_VERSION:
            models.append(row)
    if not models:
        return
    benchmarks = sorted(
        {row["benchmark"] for model in models for row in model.get("benchmarks") or []}
    )
    selections = {str(model.get("selection") or "pilot") for model in models}
    is_full = selections == {"full"}
    visual_reference = None if is_full else load_visual_reference(suite_root, benchmarks)
    lines = [
        "# Full Text Baselines" if is_full else "# Visual and Text Baselines",
        "",
        (
            "All models use every normalized benchmark sample."
            if is_full
            else "All models use the exact sample IDs selected by the Image2 pilot."
        ),
        "",
        "| Model | Output | " + " | ".join(benchmarks) + " | Overall |",
        "| --- | --- | " + " | ".join("---:" for _ in benchmarks) + " | ---: |",
    ]
    if visual_reference:
        cells = [f"{visual_reference['scores'][name]:.1f}%" for name in benchmarks]
        lines.append(
            f"| {visual_reference['model']} | Visual | "
            + " | ".join(cells)
            + f" | {visual_reference['overall_accuracy']:.1f}% |"
        )
    for model in models:
        values = {row["benchmark"]: row for row in model.get("benchmarks") or []}
        cells = [
            f"{values[name]['accuracy']:.1f}%" if name in values else "-" for name in benchmarks
        ]
        lines.append(
            f"| {model_display_name(model['model'])} | Text | "
            + " | ".join(cells)
            + f" | {model['overall']['accuracy']:.1f}% |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        root / "summary.json",
        {
            "schema_version": "provise.text_baseline_comparison.v1",
            "suite_output": str(suite_root),
            "visual_reference": visual_reference,
            "models": models,
        },
    )


def load_visual_reference(suite_root: Path, benchmarks: list[str]) -> dict[str, Any] | None:
    scores = {}
    weighted_correct = 0.0
    total_samples = 0
    for benchmark in benchmarks:
        summary_path = suite_root / "benchmarks" / benchmark / "pilot_24_total" / "summary.json"
        if not summary_path.is_file():
            return None
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        overall = summary.get("overall") or {}
        sample_count = int(overall.get("total_samples") or 0)
        accuracy = float(overall.get("accuracy") or 0.0)
        scores[benchmark] = accuracy
        weighted_correct += accuracy * sample_count / 100.0
        total_samples += sample_count
    suite_manifest = suite_root / "suite_manifest.json"
    model_name = "gpt-image-2"
    if suite_manifest.is_file():
        payload = json.loads(suite_manifest.read_text(encoding="utf-8"))
        model_name = str(payload.get("evaluation_model") or model_name)
    return {
        "model": model_name,
        "output": "visual",
        "total_samples": total_samples,
        "overall_accuracy": 100.0 * weighted_correct / total_samples if total_samples else 0.0,
        "scores": scores,
    }


def model_display_name(value: str) -> str:
    aliases = {"qwen/qwen3-vl-8b-instruct": "Qwen3-VL-8B"}
    return aliases.get(str(value), str(value))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "model"


def format_elapsed(seconds: float) -> str:
    elapsed = max(0, int(seconds))
    minutes, remainder = divmod(elapsed, 60)
    return f"{minutes}m {remainder:02d}s" if minutes else f"{remainder}s"


def print_baseline_progress(completed: int, total: int, started: float) -> None:
    elapsed = format_elapsed(time.monotonic() - started)
    print(
        f"\r  {completed:>5}/{total} samples | {elapsed}",
        end="",
        flush=True,
    )
