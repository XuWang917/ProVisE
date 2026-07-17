from __future__ import annotations

import json
import re
from typing import Any, Dict

from .base import BaseProtocol, ParseResult, format_choices


FALLBACK_PARSE_PROMPT = """You are a parser for protocolized visual evaluation.

Your task is NOT to solve the original problem yourself.
Your task is to infer what answer the image-generation model expressed in its generated visual response.

You are given:
1. The original task instruction.
2. The expected answer format.
3. The candidate options, if any.
4. The original input image, if needed for context.
5. The generated visual response from the image-generation model.

Original task instruction:
{question}

Expected answer format:
{answer_format}

Candidate options, if any:
{choices}

Rules:
- Only use the generated visual response to infer the model's intended answer.
- Do not answer the task based on your own reasoning.
- If the generated response clearly indicates one answer, convert it into the required answer format.
- If the response is ambiguous, incomplete, contradictory, or does not express an answer, return INVALID.
- Do not evaluate whether the answer is correct.

Return your result in JSON:
{
  "status": "valid" | "invalid" | "ambiguous",
  "prediction": "...",
  "evidence": "briefly describe the visual evidence used",
  "confidence": "high" | "medium" | "low"
}"""


class GenericVLMFallbackProtocol(BaseProtocol):
    """Generic fallback protocol for unsupported tasks.

    This path deliberately uses a universal VLM-assisted parser instead of a
    task-specific deterministic parser, so it is expected to be less stable than
    manually designed or verified agent-constructed protocols.
    """

    name = "generic_vlm_fallback"
    _eval_vlm = None

    def variables(self, item: Dict[str, Any], benchmark_root: str) -> Dict[str, Any]:
        try:
            values = super().variables(item, benchmark_root)
        except Exception:
            choices = item.get("choices") or item.get("options") or []
            values = {
                "id": item.get("id", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "choices": format_choices(choices),
                "choices_text": format_choices(choices),
                "n_images": 0,
            }
        values["answer_format"] = expected_answer_format(item)
        return values

    def parse(self, generated_path: str, item: Dict[str, Any], benchmark_root: str) -> ParseResult:
        eval_mode = str(self.config.get("eval_mode", "vlm")).strip().lower()
        if eval_mode in {"none", "disabled", "off"}:
            return ParseResult(
                "",
                False,
                {
                    "error": "generic fallback parser requires eval_mode=vlm",
                    "parser": "generic_vlm_fallback",
                },
            )

        try:
            vlm = self._get_eval_vlm()
            prompt = self._parse_prompt(item)
            input_paths = []
            try:
                input_paths = self.input_paths(item, benchmark_root)
            except Exception:
                input_paths = []
            include_source_images = bool(self.config.get("include_source_images", True))
            if hasattr(vlm, "predict_multi") and input_paths and include_source_images:
                response = vlm.predict_multi([*input_paths, generated_path], prompt)
            elif hasattr(vlm, "predict_multi"):
                response = vlm.predict_multi([generated_path], prompt)
            else:
                response = vlm.predict(generated_path, prompt)
        except Exception as exc:
            return ParseResult(
                "",
                False,
                {
                    "error": str(exc),
                    "parser": "generic_vlm_fallback",
                },
            )

        return self._parse_response_payload(str(response or "").strip())

    def _parse_response_payload(self, raw_response: str) -> ParseResult:
        parsed_response = parse_fallback_json_response(raw_response)
        status = str(parsed_response.get("status") or "").strip().lower()
        prediction = parsed_response.get("prediction", "")
        parse_success = status == "valid" and _has_prediction(prediction)
        return ParseResult(
            prediction,
            parse_success,
            {
                "prediction": prediction,
                "fallback_status": status or "invalid",
                "fallback_evidence": str(parsed_response.get("evidence") or "")[:500],
                "fallback_confidence": str(parsed_response.get("confidence") or ""),
                "vlm_response": raw_response[:1000],
                "parser": "generic_vlm_fallback",
                "error_type": "" if parse_success else "fallback_invalid_or_ambiguous",
                "error": "" if parse_success else "fallback parser did not return a valid prediction",
            },
        )

    def _parse_prompt(self, item: Dict[str, Any]) -> str:
        template = str(self.config.get("parse_prompt") or FALLBACK_PARSE_PROMPT)
        return fill_fallback_prompt_template(
            template,
            question=str(item.get("question", "")),
            answer_format=expected_answer_format(item),
            choices=format_choices(item.get("choices") or item.get("options") or []),
        )

    def _get_eval_vlm(self):
        if self.__class__._eval_vlm is None:
            from ..models.vlm import create_eval_vlm

            timeout = int(self.config.get("vlm_timeout", 60))
            self.__class__._eval_vlm = create_eval_vlm(timeout=timeout)
            self.__class__._eval_vlm.load_model()
        return self.__class__._eval_vlm


def expected_answer_format(item: Dict[str, Any]) -> str:
    answer_type = str(item.get("answer_type") or "").strip()
    metric = str((item.get("evaluation") or {}).get("metric") or item.get("metric") or "").strip()
    choices = item.get("choices") or item.get("options") or []
    if choices:
        labels = []
        for idx, choice in enumerate(choices):
            if isinstance(choice, dict) and choice.get("label") not in (None, ""):
                labels.append(str(choice["label"]))
            elif idx < 26:
                labels.append(chr(ord("A") + idx))
            else:
                labels.append(str(idx + 1))
        return f"one candidate option label from: {', '.join(labels)}"
    if answer_type:
        return answer_type
    if metric:
        return f"answer compatible with metric: {metric}"
    return "the original benchmark answer format"


def parse_fallback_json_response(response: str) -> Dict[str, Any]:
    text = str(response or "").strip()
    if not text:
        return {"status": "invalid", "prediction": "", "evidence": "", "confidence": "low"}

    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return _normalize_fallback_payload(parsed)

    if re.search(r"\bINVALID\b", text, flags=re.IGNORECASE):
        return {"status": "invalid", "prediction": "", "evidence": text[:500], "confidence": "low"}
    if re.search(r"\bAMBIGUOUS\b", text, flags=re.IGNORECASE):
        return {"status": "ambiguous", "prediction": "", "evidence": text[:500], "confidence": "low"}
    return {"status": "invalid", "prediction": "", "evidence": text[:500], "confidence": "low"}


def fill_fallback_prompt_template(
    template: str,
    *,
    question: str,
    answer_format: str,
    choices: str,
    **extra_values: str,
) -> str:
    values = {
        "question": question,
        "answer_format": answer_format,
        "choices": choices,
        **{k: str(v) for k, v in extra_values.items()},
    }
    rendered = str(template)
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    return candidates


def _normalize_fallback_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    status = str(payload.get("status") or "").strip().lower()
    if status not in {"valid", "invalid", "ambiguous"}:
        status = "valid" if str(payload.get("prediction") or "").strip() else "invalid"
    confidence = str(payload.get("confidence") or "").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = ""
    return {
        "status": status,
        "prediction": payload.get("prediction", ""),
        "evidence": payload.get("evidence", ""),
        "confidence": confidence,
    }


def _has_prediction(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    return True
