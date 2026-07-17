from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pct(numerator: int, denominator: int) -> float:
    return numerator * 100.0 / denominator if denominator else 0.0


def infer_sample_status(detail: Dict[str, Any]) -> str:
    status = str(detail.get("sample_status") or "").strip()
    if status:
        return status

    if detail.get("missing_input"):
        return "missing_input"
    if detail.get("score_error_type"):
        return "score_exception"
    if detail.get("parse_error_type") and not detail.get("score_computed"):
        return "parse_exception"
    if detail.get("generation_success") is False:
        return "generation_failed"
    if detail.get("model_protocol_noncompliance"):
        return "model_invalid_output"

    parse_success = detail.get("parse_success")
    score = to_float(detail.get("score"))
    is_correct = bool(detail.get("is_correct"))
    if parse_success is False:
        return "parse_failed"
    if detail.get("metric_unverified"):
        return "unscored"
    if is_correct:
        return "correct"
    if score > 0.0:
        return "partial_credit"
    return "model_incorrect"


def infer_failure_category(detail: Dict[str, Any], sample_status: str | None = None) -> str:
    status = sample_status or infer_sample_status(detail)
    mapping = {
        "correct": "",
        "partial_credit": "incorrect_prediction",
        "unscored": "",
        "model_incorrect": "incorrect_prediction",
        "model_invalid_output": "model_protocol_noncompliance",
        "parse_failed": "parser_failure",
        "parse_exception": "parser_failure",
        "generation_failed": "generation_failure",
        "missing_input": "input_missing",
        "score_exception": "scoring_failure",
    }
    return mapping.get(status, "unknown_failure")


def classify_sample_detail(detail: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(detail)
    status = infer_sample_status(normalized)
    normalized["sample_status"] = status
    normalized["failure_category"] = infer_failure_category(normalized, status)
    return normalized


def _counter_from_values(values: Iterable[str]) -> Dict[str, int]:
    return dict(sorted(Counter(v for v in values if v).items()))


def summarize_details(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = [classify_sample_detail(detail) for detail in details]
    total = len(normalized)

    scores = [to_float(detail.get("score")) for detail in normalized]
    valid_details = [detail for detail in normalized if detail.get("parse_success") is True]
    generated_details = [detail for detail in normalized if detail.get("generation_success") is True]

    status_counts = _counter_from_values(detail["sample_status"] for detail in normalized)
    failure_category_counts = _counter_from_values(detail["failure_category"] for detail in normalized)
    generation_error_type_counts = _counter_from_values(
        str(detail.get("generation_error_type") or "").strip() for detail in normalized
    )
    parse_error_type_counts = _counter_from_values(
        str(detail.get("parse_error_type") or "").strip() for detail in normalized
    )
    score_error_type_counts = _counter_from_values(
        str(detail.get("score_error_type") or "").strip() for detail in normalized
    )

    correct_count = sum(1 for detail in normalized if detail.get("is_correct") is True)
    partial_credit_count = sum(1 for detail in normalized if detail["sample_status"] == "partial_credit")
    model_incorrect_count = sum(1 for detail in normalized if detail["sample_status"] == "model_incorrect")
    model_invalid_output_count = sum(
        1 for detail in normalized if detail["sample_status"] == "model_invalid_output"
    )
    parse_failed_count = sum(1 for detail in normalized if detail["sample_status"] == "parse_failed")
    parse_exception_count = sum(1 for detail in normalized if detail["sample_status"] == "parse_exception")
    generation_failed_count = sum(1 for detail in normalized if detail["sample_status"] == "generation_failed")
    missing_input_count = sum(1 for detail in normalized if detail["sample_status"] == "missing_input")
    score_exception_count = sum(1 for detail in normalized if detail["sample_status"] == "score_exception")
    unscored_count = sum(1 for detail in normalized if detail["sample_status"] == "unscored")

    generated_count = len(generated_details)
    valid_parse_count = len(valid_details)
    invalid_output_count = parse_failed_count + parse_exception_count + model_invalid_output_count
    model_error_count = partial_credit_count + model_incorrect_count + model_invalid_output_count
    scored_count = sum(1 for detail in normalized if detail.get("score_computed"))

    mean_score = sum(scores) / total if total else 0.0
    mean_score_valid = (
        sum(to_float(detail.get("score")) for detail in valid_details) / valid_parse_count if valid_parse_count else 0.0
    )

    return {
        "total_samples": total,
        "scored_count": scored_count,
        "generated_count": generated_count,
        "generated_rate": pct(generated_count, total),
        "valid_parse_count": valid_parse_count,
        "valid_parse_rate": pct(valid_parse_count, total),
        "valid_parse_among_generated_rate": pct(valid_parse_count, generated_count),
        "invalid_output_count": invalid_output_count,
        "invalid_output_rate": pct(invalid_output_count, total),
        "invalid_output_among_generated_rate": pct(invalid_output_count, generated_count),
        "missing_input_count": missing_input_count,
        "missing_input_rate": pct(missing_input_count, total),
        "generation_failed_count": generation_failed_count,
        "generation_failed_rate": pct(generation_failed_count, total),
        "parser_failure_count": parse_failed_count + parse_exception_count,
        "parser_failure_rate": pct(parse_failed_count + parse_exception_count, total),
        "parse_failed_count": parse_failed_count,
        "parse_exception_count": parse_exception_count,
        "score_exception_count": score_exception_count,
        "unscored_count": unscored_count,
        "unscored_rate": pct(unscored_count, total),
        "correct_count": correct_count,
        "accuracy": pct(correct_count, total),
        "correct_among_valid": pct(correct_count, valid_parse_count),
        "partial_credit_count": partial_credit_count,
        "model_incorrect_count": model_incorrect_count,
        "model_invalid_output_count": model_invalid_output_count,
        "model_error_count": model_error_count,
        "model_error_rate": pct(model_error_count, total),
        "mean_score": mean_score,
        "mean_score_valid": mean_score_valid,
        "status_counts": status_counts,
        "failure_category_counts": failure_category_counts,
        "generation_error_type_counts": generation_error_type_counts,
        "parse_error_type_counts": parse_error_type_counts,
        "score_error_type_counts": score_error_type_counts,
    }
