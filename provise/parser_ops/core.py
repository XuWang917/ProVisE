from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple


class ParserPlanError(ValueError):
    """Raised when a declarative parser pipeline is structurally invalid."""


class ParserOpError(RuntimeError):
    """A recoverable failure produced by one parser operator."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "operator_failed",
        diagnostics: Mapping[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.diagnostics = dict(diagnostics or {})


@dataclass(frozen=True)
class ParserContext:
    generated_path: str
    item: Mapping[str, Any]
    benchmark_root: str
    source_paths: Tuple[str, ...] = ()
    protocol_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ParserValue:
    kind: str
    value: Any
    diagnostics: Dict[str, Any] = field(default_factory=dict)


ParserOpCallable = Callable[[ParserContext, Sequence[ParserValue], Mapping[str, Any]], ParserValue]
ParserParamValidator = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True)
class ParserOpSpec:
    name: str
    input_kinds: Tuple[str, ...]
    output_kind: str
    function: ParserOpCallable
    description: str = ""
    allowed_params: frozenset[str] = frozenset()
    required_params: frozenset[str] = frozenset()
    validate_params: ParserParamValidator | None = None


@dataclass(frozen=True)
class ParserStep:
    id: str
    op: str
    inputs: Tuple[str, ...]
    params: Mapping[str, Any]


@dataclass(frozen=True)
class ParserPipeline:
    steps: Tuple[ParserStep, ...]
    output: str


@dataclass
class ParserPipelineResult:
    output: ParserValue | None
    success: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    error_type: str = ""

    @property
    def prediction(self) -> Any:
        return self.output.value if self.output is not None else None


class ParserRegistry:
    """Registry and executor for small, typed parser pipelines."""

    def __init__(self) -> None:
        self._ops: Dict[str, ParserOpSpec] = {}

    def register(self, spec: ParserOpSpec) -> None:
        name = str(spec.name or "").strip()
        if not name:
            raise ValueError("Parser operator name cannot be empty")
        if name in self._ops:
            raise ValueError(f"Parser operator already registered: {name}")
        if not spec.output_kind:
            raise ValueError(f"Parser operator {name} must declare an output kind")
        self._ops[name] = spec

    def get(self, name: str) -> ParserOpSpec:
        try:
            return self._ops[name]
        except KeyError as exc:
            raise ParserPlanError(
                f"Unknown parser operator {name!r}. Available: {sorted(self._ops)}"
            ) from exc

    def list_ops(self) -> list[str]:
        return sorted(self._ops)

    def inventory(self) -> list[Dict[str, Any]]:
        return [
            {
                "op": spec.name,
                "description": spec.description,
                "input_kinds": list(spec.input_kinds),
                "output_kind": spec.output_kind,
                "allowed_params": sorted(spec.allowed_params),
                "required_params": sorted(spec.required_params),
            }
            for spec in (self._ops[name] for name in sorted(self._ops))
        ]

    def compile(self, raw_pipeline: Mapping[str, Any]) -> ParserPipeline:
        if not isinstance(raw_pipeline, Mapping):
            raise ParserPlanError("Parser pipeline must be a mapping")
        raw_steps = raw_pipeline.get("steps", raw_pipeline.get("pipeline"))
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ParserPlanError("Parser pipeline must contain a non-empty steps list")
        if len(raw_steps) > 32:
            raise ParserPlanError("Parser pipeline cannot contain more than 32 steps")

        steps = []
        output_kinds: Dict[str, str] = {}
        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, Mapping):
                raise ParserPlanError(f"Parser step {index} must be a mapping")
            step_id = str(raw_step.get("id") or "").strip()
            op_name = str(raw_step.get("op") or "").strip()
            if not step_id:
                raise ParserPlanError(f"Parser step {index} is missing id")
            if step_id in output_kinds:
                raise ParserPlanError(f"Duplicate parser step id: {step_id}")
            spec = self.get(op_name)

            raw_inputs = raw_step.get("inputs", [])
            if isinstance(raw_inputs, str):
                raw_inputs = [raw_inputs]
            if not isinstance(raw_inputs, list) or not all(isinstance(value, str) for value in raw_inputs):
                raise ParserPlanError(f"Parser step {step_id} inputs must be a list of step ids")
            inputs = tuple(raw_inputs)
            if len(inputs) != len(spec.input_kinds):
                raise ParserPlanError(
                    f"Parser step {step_id} ({op_name}) expects {len(spec.input_kinds)} inputs, "
                    f"received {len(inputs)}"
                )
            for input_id, expected_kind in zip(inputs, spec.input_kinds):
                if input_id not in output_kinds:
                    raise ParserPlanError(
                        f"Parser step {step_id} references missing or forward input {input_id!r}"
                    )
                actual_kind = output_kinds[input_id]
                if actual_kind != expected_kind:
                    raise ParserPlanError(
                        f"Parser step {step_id} expects {input_id!r} to have kind "
                        f"{expected_kind!r}, received {actual_kind!r}"
                    )

            params = raw_step.get("params") or {}
            if not isinstance(params, Mapping):
                raise ParserPlanError(f"Parser step {step_id} params must be a mapping")
            unknown_params = set(params) - set(spec.allowed_params)
            if unknown_params:
                raise ParserPlanError(
                    f"Parser step {step_id} has unsupported params for {op_name}: {sorted(unknown_params)}"
                )
            missing_params = set(spec.required_params) - set(params)
            if missing_params:
                raise ParserPlanError(
                    f"Parser step {step_id} is missing params for {op_name}: {sorted(missing_params)}"
                )
            if spec.validate_params is not None and not _contains_config_reference(params):
                try:
                    spec.validate_params(params)
                except (TypeError, ValueError) as exc:
                    raise ParserPlanError(
                        f"Parser step {step_id} has invalid params for {op_name}: {exc}"
                    ) from exc

            steps.append(ParserStep(step_id, op_name, inputs, dict(params)))
            output_kinds[step_id] = spec.output_kind

        output = str(raw_pipeline.get("output") or "").strip()
        if not output:
            raise ParserPlanError("Parser pipeline must declare an output step id")
        if output not in output_kinds:
            raise ParserPlanError(f"Parser pipeline output references unknown step {output!r}")
        return ParserPipeline(tuple(steps), output)

    def execute(
        self,
        raw_pipeline: Mapping[str, Any],
        context: ParserContext,
    ) -> ParserPipelineResult:
        try:
            pipeline = self.compile(raw_pipeline)
        except ParserPlanError as exc:
            return ParserPipelineResult(
                None,
                False,
                {"steps": {}},
                str(exc),
                "invalid_pipeline",
            )

        values: Dict[str, ParserValue] = {}
        step_diagnostics: Dict[str, Any] = {}
        for step in pipeline.steps:
            spec = self._ops[step.op]
            try:
                params = _resolve_config_refs(step.params, context.protocol_config)
                if spec.validate_params is not None:
                    spec.validate_params(params)
                result = spec.function(context, [values[value] for value in step.inputs], params)
                if not isinstance(result, ParserValue):
                    raise ParserOpError(
                        f"Operator {step.op} returned {type(result).__name__}, expected ParserValue",
                        code="operator_contract_error",
                    )
                if result.kind != spec.output_kind:
                    raise ParserOpError(
                        f"Operator {step.op} returned kind {result.kind!r}, expected {spec.output_kind!r}",
                        code="operator_kind_mismatch",
                    )
            except ParserOpError as exc:
                step_diagnostics[step.id] = _json_safe(
                    {
                        "op": step.op,
                        "status": "failed",
                        "error": str(exc),
                        "error_type": exc.code,
                        **exc.diagnostics,
                    }
                )
                return ParserPipelineResult(
                    None,
                    False,
                    {"output": pipeline.output, "steps": step_diagnostics},
                    str(exc),
                    exc.code,
                )
            except (TypeError, ValueError) as exc:
                step_diagnostics[step.id] = {
                    "op": step.op,
                    "status": "failed",
                    "error": str(exc),
                    "error_type": "invalid_operator_params",
                }
                return ParserPipelineResult(
                    None,
                    False,
                    {"output": pipeline.output, "steps": step_diagnostics},
                    str(exc),
                    "invalid_operator_params",
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                step_diagnostics[step.id] = {
                    "op": step.op,
                    "status": "failed",
                    "error": message,
                    "error_type": "operator_exception",
                }
                return ParserPipelineResult(
                    None,
                    False,
                    {"output": pipeline.output, "steps": step_diagnostics},
                    message,
                    "operator_exception",
                )

            values[step.id] = result
            step_diagnostics[step.id] = _json_safe(
                {"op": step.op, "status": "ok", "kind": result.kind, **result.diagnostics}
            )

        return ParserPipelineResult(
            values[pipeline.output],
            True,
            {"output": pipeline.output, "steps": step_diagnostics},
        )

    def output_kind(self, raw_pipeline: Mapping[str, Any]) -> str:
        pipeline = self.compile(raw_pipeline)
        output_step = next(step for step in pipeline.steps if step.id == pipeline.output)
        return self.get(output_step.op).output_kind


def _resolve_config_refs(value: Any, config: Mapping[str, Any]) -> Any:
    if isinstance(value, Mapping):
        if "$config" in value:
            extra_keys = set(value) - {"$config", "default"}
            if extra_keys:
                raise ParserOpError(
                    f"Config reference has unsupported keys: {sorted(extra_keys)}",
                    code="invalid_config_reference",
                )
            path = str(value.get("$config") or "").strip()
            found, resolved = _lookup_config(config, path)
            if found:
                return resolved
            if "default" in value:
                return _resolve_config_refs(value["default"], config)
            raise ParserOpError(
                f"Parser config value not found: {path or '<empty>'}",
                code="missing_config_value",
            )
        return {key: _resolve_config_refs(item, config) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_config_refs(item, config) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_config_refs(item, config) for item in value)
    return value


def _contains_config_reference(value: Any) -> bool:
    if isinstance(value, Mapping):
        return "$config" in value or any(_contains_config_reference(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_config_reference(item) for item in value)
    return False


def _lookup_config(config: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    if not path:
        return False, None
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value
