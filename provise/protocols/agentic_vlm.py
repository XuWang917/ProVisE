from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

from ..evaluation.metrics import score_prediction
from .base import ParseResult, ScoreResult
from .fallback import (
    GenericVLMFallbackProtocol,
    _has_prediction,
    expected_answer_format,
    fill_fallback_prompt_template,
    format_choices,
    parse_fallback_json_response,
)


DEFAULT_AGENTIC_PARSE_PROMPT = """You are a parser for an agent-constructed visual-answer protocol.

Your task is NOT to solve the original problem yourself. Your task is to infer
what answer the image-generation model expressed in its generated visual
response.

Protocol-specific parsing instruction:
{parse_prompt}

Expected visual strategy:
{visual_strategy}

Expected task-relevant spatial evidence:
{visual_evidence}

Concrete parser observation:
{parser_observation}

Answer recovery contract by schema:
{answer_recovery}

Role of any answer label:
{label_role}

Expected answer format:
{answer_format}

Candidate options, if any:
{choices}

Invalid output conditions:
{invalid_conditions}

Rules:
- Only use the generated visual response and fixed protocol contract to infer
  the model's intended answer. The original task question and source image are
  intentionally unavailable.
- Do not answer the task from your own spatial reasoning.
- First identify the spatial evidence expressed by the edit, such as selected
  object outline, size cue, distance line, hidden-region overlay, target mask,
  point, path arrow, trajectory cue, or before/after state cue.
- If the only evidence is an isolated answer label/code with no task-relevant
  spatial mark, return status="ambiguous" unless the protocol explicitly allows
  a label-only response.
- If an optional label is present, use it only after checking that it is attached
  to or consistent with the spatial evidence.
- Convert a clear visual response into the expected answer format.
- If the generated response is ambiguous, incomplete, contradictory, or does
  not express an answer, return status="invalid" or status="ambiguous".
- The evidence field must name the concrete spatial mark or generated state
  that supports the recovered prediction. A bare option label is not evidence.
- Do not evaluate whether the expressed answer is correct.

Return ONLY JSON:
{
  "status": "valid" | "invalid" | "ambiguous",
  "prediction": "...",
  "evidence": "briefly describe the generated visual evidence used",
  "confidence": "high" | "medium" | "low"
}
"""


class AgenticVLMProtocol(GenericVLMFallbackProtocol):
    """Task-specific agent-constructed protocol with a VLM-assisted parser.

    The generation prompt is supplied by the generated protocol spec. The parser
    prompt is task-specific rather than the generic fallback prompt, while score
    normalization reuses the benchmark-compatible BaseProtocol behavior.
    """

    name = "agentic_vlm_protocol"
    _eval_vlm = None

    def __init__(self, config: Dict[str, Any] | None = None):
        normalized = dict(config or {})
        normalized.setdefault("include_source_images", False)
        super().__init__(normalized)

    def variables(self, item: Dict[str, Any], benchmark_root: str) -> Dict[str, Any]:
        values = super().variables(item, benchmark_root)
        values.update(
            {
                "agentic_protocol_id": self.config.get("generated_protocol_id", ""),
                "agentic_rationale": self.config.get("rationale", ""),
                "answer_format": self.config.get("answer_format") or expected_answer_format(item),
                "invalid_conditions": _format_invalid_conditions(self.config.get("invalid_conditions")),
                "visual_strategy": self.config.get("visual_strategy", ""),
                "visual_evidence": self.config.get("visual_evidence", ""),
                "parser_observation": self.config.get("parser_observation", ""),
                "answer_recovery": _format_answer_recovery(self.config.get("answer_recovery")),
                "label_role": self.config.get("label_role", "none"),
            }
        )
        return values

    def parse(self, generated_path: str, item: Dict[str, Any], benchmark_root: str) -> ParseResult:
        mock_response = str(self.config.get("mock_parse_response") or "").strip()
        if mock_response:
            return self._parse_response_payload(mock_response)
        try:
            input_paths = self.input_paths(item, benchmark_root)
            source_path = input_paths[0] if input_paths else ""
            visual_change = measure_visual_change(
                source_path,
                generated_path,
                pixel_threshold=int(self.config.get("visual_change_pixel_threshold", 12)),
                min_changed_fraction=float(self.config.get("min_visual_change_fraction", 0.002)),
            )
        except Exception as exc:
            return ParseResult(
                "",
                False,
                {
                    "parser": "agentic_vlm_protocol",
                    "parser_backend": "vlm_fallback",
                    "error_type": "visual_change_check_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        if not visual_change["sufficient"]:
            return ParseResult(
                "",
                False,
                {
                    **visual_change,
                    "agentic_status": "invalid",
                    "agentic_evidence": (
                        "The generated image changed too little to contain the required visual answer."
                    ),
                    "model_protocol_noncompliance": True,
                    "parser": "agentic_vlm_protocol",
                    "parser_backend": "vlm_fallback",
                    "error_type": "insufficient_visual_change",
                    "error": "generated response does not contain enough visible change to encode an answer",
                },
            )
        parsed = super().parse(generated_path, item, benchmark_root)
        parsed.extra.update(visual_change)
        return parsed

    def _parse_prompt(self, item: Dict[str, Any]) -> str:
        task_parse_prompt = str(self.config.get("parse_prompt") or "").strip()
        template = str(self.config.get("parse_prompt_template") or DEFAULT_AGENTIC_PARSE_PROMPT)
        return fill_fallback_prompt_template(
            template,
            question=str(item.get("question", "")),
            answer_format=str(self.config.get("answer_format") or expected_answer_format(item)),
            choices=format_choices(item.get("choices") or item.get("options") or []),
            parse_prompt=task_parse_prompt,
            invalid_conditions=_format_invalid_conditions(self.config.get("invalid_conditions")),
            visual_strategy=str(self.config.get("visual_strategy") or ""),
            visual_evidence=str(self.config.get("visual_evidence") or ""),
            parser_observation=str(self.config.get("parser_observation") or ""),
            answer_recovery=_format_answer_recovery(self.config.get("answer_recovery")),
            label_role=str(self.config.get("label_role") or "none"),
        )

    def _parse_response_payload(self, response: str) -> ParseResult:
        raw_response = str(response or "").strip()
        parsed_response = parse_fallback_json_response(raw_response)
        status = str(parsed_response.get("status") or "").strip().lower()
        prediction = parsed_response.get("prediction", "")
        parse_success = status == "valid" and _has_prediction(prediction)
        model_noncompliance = status in {"invalid", "ambiguous"}
        return ParseResult(
            prediction,
            parse_success,
            {
                "prediction": prediction,
                "agentic_status": status or "invalid",
                "agentic_evidence": str(parsed_response.get("evidence") or "")[:500],
                "agentic_confidence": str(parsed_response.get("confidence") or ""),
                "model_protocol_noncompliance": model_noncompliance,
                "agentic_visual_strategy": str(self.config.get("visual_strategy") or ""),
                "vlm_response": raw_response[:1000],
                "parser": "agentic_vlm_protocol",
                "parser_backend": "vlm_fallback",
                "error_type": "" if parse_success else "agentic_invalid_or_ambiguous",
                "error": "" if parse_success else "agentic parser did not return a valid prediction",
            },
        )

    def score(self, parsed: ParseResult, item: Dict[str, Any], benchmark_root: str) -> ScoreResult:
        if parsed.extra.get("model_protocol_noncompliance"):
            return ScoreResult(
                0.0,
                False,
                {"model_protocol_noncompliance": True},
            )
        metric = str(
            self.config.get("metric")
            or (item.get("evaluation") or {}).get("metric")
            or "unverified"
        )
        metric_config = dict((item.get("evaluation") or {}).get("metric_config") or {})
        metric_config.update(dict(self.config.get("metric_config") or {}))
        result = score_prediction(metric, parsed.prediction, item, benchmark_root, metric_config)
        return ScoreResult(result.score, result.is_correct, dict(result.extra))


def _format_invalid_conditions(value: Any) -> str:
    if value is None or value == "":
        return "ambiguous, incomplete, contradictory, text-only, or unrelated generated response"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {item}" for item in value)
    return str(value)


def _format_answer_recovery(value: Any) -> str:
    if not value:
        return "Use the protocol-specific parsing instruction."
    if isinstance(value, dict):
        return "\n".join(f"- {schema}: {readout}" for schema, readout in value.items())
    if isinstance(value, (list, tuple)):
        rows = []
        for item in value:
            if isinstance(item, dict):
                schema = item.get("answer_schema") or item.get("schema") or "schema"
                readout = item.get("visual_readout") or item.get("readout") or ""
                rows.append(f"- {schema}: {readout}")
            else:
                rows.append(f"- {item}")
        return "\n".join(rows)
    return str(value)


def measure_visual_change(
    source_path: str,
    generated_path: str,
    *,
    pixel_threshold: int = 12,
    min_changed_fraction: float = 0.002,
) -> Dict[str, Any]:
    if not source_path or not Path(source_path).exists():
        raise FileNotFoundError(f"source image is unavailable: {source_path}")
    if not generated_path or not Path(generated_path).exists():
        raise FileNotFoundError(f"generated image is unavailable: {generated_path}")
    with Image.open(source_path) as source_image, Image.open(generated_path) as generated_image:
        source = source_image.convert("RGB")
        generated = generated_image.convert("RGB")
        if source.size != generated.size:
            source = source.resize(generated.size, Image.Resampling.BILINEAR)
        source_array = np.asarray(source, dtype=np.int16)
        generated_array = np.asarray(generated, dtype=np.int16)
    delta = np.abs(generated_array - source_array)
    changed = np.max(delta, axis=2) >= max(1, int(pixel_threshold))
    changed_fraction = float(np.mean(changed))
    mean_absolute_delta = float(np.mean(delta) / 255.0)
    threshold = max(0.0, float(min_changed_fraction))
    return {
        "visual_change_fraction": changed_fraction,
        "visual_change_mean_absolute_delta": mean_absolute_delta,
        "visual_change_pixel_threshold": int(pixel_threshold),
        "visual_change_min_fraction": threshold,
        "sufficient": changed_fraction >= threshold,
    }
