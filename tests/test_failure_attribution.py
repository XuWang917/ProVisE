import argparse
import json
from pathlib import Path

from PIL import Image

from provise.evaluation.results import summarize_details
from provise.evaluation.runner import MockProtocolModel, load_protocol_pool, run_task


class FailingGenerateModel:
    def __init__(self):
        self.last_error_type = ""
        self.last_error_message = ""

    def clear_last_error(self):
        self.last_error_type = ""
        self.last_error_message = ""

    def generate(self, image_path: str, prompt: str, save_path: str) -> bool:
        self.last_error_type = "timeout"
        self.last_error_message = "synthetic timeout"
        return False


def _write_sample(tmp_path: Path, *, image_name: str = "sample.png") -> tuple[Path, Path]:
    image_path = tmp_path / image_name
    Image.new("RGB", (64, 64), "white").save(image_path)

    data_path = tmp_path / "data.jsonl"
    item = {
        "id": "sample_001",
        "task": "choice_qa",
        "capability": "reasoning",
        "image_path": image_name,
        "question": "Select option A.",
        "answer": "A",
        "choices": [
            {"label": "A", "text": "Option A"},
            {"label": "B", "text": "Option B"},
            {"label": "C", "text": "Option C"},
            {"label": "D", "text": "Option D"},
        ],
    }
    data_path.write_text(json.dumps(item, ensure_ascii=False) + "\n", encoding="utf-8")
    return image_path, data_path


def _args(tmp_path: Path, data_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        data_file=str(data_path),
        limit=None,
        benchmark_root=str(tmp_path),
        no_reuse=True,
        protocol="",
        print_prompt=False,
        model="test-model",
    )


def _task_cfg() -> dict:
    return {
        "protocol": "label_code",
        "prompt_variant": "choice_4_corners",
        "input": {"mode": "single"},
    }


def test_summarize_details_reports_failure_breakdown():
    details = [
        {"sample_status": "correct", "generation_success": True, "parse_success": True, "score_computed": True, "is_correct": True, "score": 1.0},
        {"sample_status": "model_incorrect", "generation_success": True, "parse_success": True, "score_computed": True, "is_correct": False, "score": 0.0},
        {"sample_status": "partial_credit", "generation_success": True, "parse_success": True, "score_computed": True, "is_correct": False, "score": 0.5},
        {"sample_status": "parse_failed", "generation_success": True, "parse_success": False, "score_computed": True, "is_correct": False, "score": 0.0},
        {"sample_status": "generation_failed", "generation_success": False, "parse_success": False, "score_computed": False, "is_correct": False, "score": 0.0},
    ]

    summary = summarize_details(details)

    assert summary["total_samples"] == 5
    assert summary["generated_count"] == 4
    assert summary["valid_parse_count"] == 3
    assert summary["invalid_output_count"] == 1
    assert summary["generation_failed_count"] == 1
    assert summary["model_error_count"] == 2
    assert summary["correct_count"] == 1
    assert summary["correct_among_valid"] == 100 / 3
    assert summary["failure_category_counts"]["generation_failure"] == 1
    assert summary["failure_category_counts"]["parser_failure"] == 1
    assert summary["failure_category_counts"]["incorrect_prediction"] == 2


def test_unverified_metric_is_unscored_not_a_model_error():
    summary = summarize_details(
        [
            {
                "generation_success": True,
                "parse_success": True,
                "score_computed": False,
                "metric_unverified": True,
                "is_correct": False,
                "score": 0.0,
            }
        ]
    )

    assert summary["status_counts"] == {"unscored": 1}
    assert summary["unscored_count"] == 1
    assert summary["model_error_count"] == 0
    assert summary["failure_category_counts"] == {}


def test_structured_protocol_rejection_is_a_model_invalid_output():
    summary = summarize_details(
        [
            {
                "generation_success": True,
                "parse_success": False,
                "score_computed": True,
                "is_correct": False,
                "score": 0.0,
                "agentic_status": "ambiguous",
                "agentic_evidence": "The requested spatial edit is not visible.",
                "model_protocol_noncompliance": True,
            }
        ]
    )

    assert summary["status_counts"] == {"model_invalid_output": 1}
    assert summary["model_invalid_output_count"] == 1
    assert summary["model_error_count"] == 1
    assert summary["parser_failure_count"] == 0
    assert summary["failure_category_counts"] == {"model_protocol_noncompliance": 1}


def test_run_task_records_generation_failure(tmp_path: Path):
    _, data_path = _write_sample(tmp_path)
    protocol_pool = load_protocol_pool("configs/protocol_specs")
    args = _args(tmp_path, data_path)
    output_root = tmp_path / "out"

    results = run_task("choice_qa", _task_cfg(), protocol_pool, args, FailingGenerateModel(), output_root)

    assert results["generation_failed_count"] == 1
    assert results["generated_count"] == 0
    assert results["status_counts"]["generation_failed"] == 1
    assert results["detailed_results"][0]["sample_status"] == "generation_failed"
    assert results["detailed_results"][0]["generation_error_type"] == "timeout"


def test_run_task_records_parse_failed(tmp_path: Path):
    _, data_path = _write_sample(tmp_path)
    protocol_pool = load_protocol_pool("configs/protocol_specs")
    args = _args(tmp_path, data_path)
    output_root = tmp_path / "out"

    results = run_task("choice_qa", _task_cfg(), protocol_pool, args, MockProtocolModel("mock-copy"), output_root)

    assert results["generated_count"] == 1
    assert results["parser_failure_count"] == 1
    assert results["valid_parse_count"] == 0
    assert results["status_counts"]["parse_failed"] == 1
    assert results["detailed_results"][0]["sample_status"] == "parse_failed"
