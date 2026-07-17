import json
from pathlib import Path

from PIL import Image

from provise.commands.baseline import (
    build_direct_prompt,
    evaluate_benchmark,
    parse_direct_response,
    pilot_sample_ids,
    prepare_model_inputs,
)


class _FixedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def predict_multi(self, image_paths, prompt):
        self.requests.append((image_paths, prompt))
        return next(self.responses)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_direct_response_parser_handles_choices_measurements_and_points():
    choice = {
        "answer_type": "choice",
        "choices": [{"label": "A", "text": "left"}, {"label": "B", "text": "right"}],
    }
    assert parse_direct_response("B", choice) == "B"
    assert parse_direct_response("right", choice) == "B"
    assert parse_direct_response('{"value": 12, "unit": "cm"}', {"answer_type": "number"}) == {
        "value": 12,
        "unit": "cm",
    }
    assert parse_direct_response(
        '{"points": [[0.2, 0.4], [0.3, 0.5]]}',
        {"answer_type": "points"},
    ) == [[0.2, 0.4], [0.3, 0.5]]
    assert parse_direct_response(
        '{"points": [[420, 0.67]]}',
        {"answer_type": "points"},
    ) == [[0.42, 0.67]]
    assert parse_direct_response(
        '{"points": [[0.51, 0.59], [0.49',
        {"answer_type": "points"},
    ) == [[0.51, 0.59]]
    assert parse_direct_response(
        "D",
        {"answer_type": "choice", "choices": [], "answer": "3"},
    ) == "3"


def test_direct_prompt_does_not_include_ground_truth():
    item = {
        "question": "Where is the chair?",
        "answer": "SECRET_GOLD",
        "answer_type": "choice",
        "choices": [{"label": "A", "text": "left"}, {"label": "B", "text": "right"}],
    }
    prompt = build_direct_prompt(item)
    assert "Where is the chair?" in prompt
    assert "A. left" in prompt
    assert "SECRET_GOLD" not in prompt

    point_prompt = build_direct_prompt(
        {"question": "Point to it", "answer_type": "points", "choices": []}
    )
    assert "exactly one most-confident point" in point_prompt


def test_direct_point_prompt_replaces_embedded_multi_point_format():
    prompt = build_direct_prompt(
        {
            "question": (
                "What part of a mug should be gripped to lift it? "
                "Your answer should be formatted as a list of tuples, i.e. "
                "[(x1, y1), (x2, y2), ...], with normalized coordinates."
            ),
            "answer_type": "points",
            "choices": [],
        }
    )

    assert "What part of a mug should be gripped to lift it?" in prompt
    assert "list of tuples" not in prompt
    assert prompt.count("exactly one most-confident point") == 1
    assert "not [420, 670]" in prompt
    assert "0-to-1000 coordinate system" in prompt


def test_model_input_resize_preserves_aspect_ratio(tmp_path: Path):
    source = tmp_path / "large.png"
    Image.new("RGB", (3000, 2000), "white").save(source)

    prepared = prepare_model_inputs([str(source)], tmp_path / "cache", max_image_side=1500)

    assert prepared[0] != str(source)
    with Image.open(prepared[0]) as image:
        assert image.size == (1500, 1000)


def test_text_baseline_reuses_exact_pilot_ids_and_scores_without_a_judge(tmp_path: Path):
    workspace = tmp_path / "toy"
    image_path = workspace / "assets" / "image.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (24, 16), "white").save(image_path)
    items = [
        {
            "schema_version": "genbench.v1",
            "id": "sample-a",
            "benchmark": "toy",
            "task": "relation",
            "input": {"media": [{"path": "assets/image.png", "role": "primary"}]},
            "question": "Which side?",
            "answer": "B",
            "answer_type": "choice",
            "choices": [{"label": "A", "text": "left"}, {"label": "B", "text": "right"}],
            "evaluation": {"metric": "accuracy", "metric_config": {}},
        },
        {
            "schema_version": "genbench.v1",
            "id": "sample-b",
            "benchmark": "toy",
            "task": "distance",
            "input": {"media": [{"path": "assets/image.png", "role": "primary"}]},
            "question": "How far?",
            "answer": 10,
            "answer_type": "number",
            "choices": [],
            "evaluation": {
                "metric": "qspatial_ratio",
                "metric_config": {"delta": 2},
            },
            "metadata": {"answer_unit": "cm"},
        },
    ]
    (workspace / "toy.unified.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in items),
        encoding="utf-8",
    )
    _write_json(workspace / "pilot" / "summary.json", {"tasks": ["relation", "distance"]})
    _write_json(
        workspace / "pilot" / "relation" / "results.json",
        {"detailed_results": [{"id": "sample-a"}]},
    )
    _write_json(
        workspace / "pilot" / "distance" / "results.json",
        {"detailed_results": [{"id": "sample-b"}]},
    )

    assert pilot_sample_ids(workspace / "pilot") == ["sample-a", "sample-b"]
    model = _FixedModel(["B", '{"value": 12, "unit": "cm"}'])
    summary = evaluate_benchmark(
        workspace,
        model,
        model_name="fixed",
        backend="mock",
        pilot_name="pilot",
        output_root=tmp_path / "results",
        resume=False,
    )

    assert summary["total_samples"] == 2
    assert summary["correct_count"] == 2
    assert summary["accuracy"] == 100.0
    assert len(model.requests) == 2

    full_model = _FixedModel(["B", '{"value": 12, "unit": "cm"}'])
    full_summary = evaluate_benchmark(
        workspace,
        full_model,
        model_name="fixed",
        backend="mock",
        pilot_name="pilot",
        output_root=tmp_path / "full_results",
        full=True,
        resume=False,
    )
    assert full_summary["selection"] == "full"
    assert full_summary["total_samples"] == 2


def test_resume_retries_inference_failures_without_duplicate_rows(tmp_path: Path):
    workspace = tmp_path / "toy"
    image_path = workspace / "assets" / "image.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (24, 16), "white").save(image_path)
    item = {
        "id": "sample-a",
        "benchmark": "toy",
        "task": "relation",
        "input": {"media": [{"path": "assets/image.png", "role": "primary"}]},
        "question": "Which side?",
        "answer": "A",
        "answer_type": "choice",
        "choices": [{"label": "A", "text": "left"}, {"label": "B", "text": "right"}],
        "evaluation": {"metric": "accuracy", "metric_config": {}},
    }
    (workspace / "toy.unified.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
    _write_json(workspace / "pilot" / "summary.json", {"tasks": ["relation"]})
    _write_json(
        workspace / "pilot" / "relation" / "results.json",
        {"detailed_results": [{"id": "sample-a"}]},
    )
    output = tmp_path / "results"
    output.mkdir()
    (output / "results.jsonl").write_text(
        json.dumps({"id": "sample-a", "inference_success": False}) + "\n",
        encoding="utf-8",
    )

    summary = evaluate_benchmark(
        workspace,
        _FixedModel(["A"]),
        model_name="fixed",
        backend="mock",
        pilot_name="pilot",
        output_root=output,
        resume=True,
    )

    assert summary["correct_count"] == 1
    assert len((output / "results.jsonl").read_text().splitlines()) == 1
