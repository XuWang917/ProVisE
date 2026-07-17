"""Validate and compile Agent-proposed Visual Contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

from provise.evaluation.metrics import (
    metric_accepts_output,
    normalize_metric_name,
)
from provise.parser_ops import (
    DEFAULT_REGISTRY,
    ParserPlanError,
    dual_anchor_relation_pipeline,
    green_choice_count_board_pipeline,
    grounded_dimension_pipeline,
    marked_object_choice_pipeline,
    relation_zone_boolean_pipeline,
    semantic_choice_clip_pipeline,
)
from provise.benchmark.tasks import infer_answer_schema


RECIPE_SPECS: Dict[str, Dict[str, Any]] = {
    "point_marker": {
        "description": "Mark one target point or object with a source-aware cyan dot.",
        "evidence": ["target point", "object localization"],
        "output_kind": "normalized_point",
        "parser_backend": "deterministic",
        "input_modes": ["single"],
    },
    "instance_marker_count": {
        "description": (
            "Place one source-aware green marker on every qualifying instance and "
            "count markers. Set target_scope=visible_and_occlusion_inferred when the "
            "benchmark asks for instances hidden by an occluder."
        ),
        "evidence": ["marked object set", "instance count"],
        "output_kind": "integer_count",
        "parser_backend": "deterministic",
        "input_modes": ["single"],
        "parameters": {
            "target_scope": {
                "type": "string",
                "enum": ["visible", "visible_and_occlusion_inferred"],
                "default": "visible",
                "description": (
                    "Use visible for ordinary counting; use "
                    "visible_and_occlusion_inferred to place markers at inferred "
                    "hidden positions as the visible pattern continues."
                ),
            }
        },
    },
    "choice_count_board": {
        "description": (
            "For a numeric choice question over one or more views, synthesize a clean evidence "
            "board with one recognizable target tile and one uniform marker per distinct counted "
            "instance or region, then map the marker count to the numeric choice."
        ),
        "evidence": ["counted target set", "one marker per distinct item", "numeric choice"],
        "output_kind": "choice_label",
        "parser_backend": "deterministic",
        "input_modes": ["single", "metadata_images"],
    },
    "region_mask": {
        "description": "Render the answer as a white target region on a black mask.",
        "evidence": ["target region", "affordance region"],
        "output_kind": "mask_path",
        "parser_backend": "deterministic",
        "input_modes": ["single"],
    },
    "trajectory": {
        "description": "Draw one continuous red spatial path from the task start to its target.",
        "evidence": ["path", "trajectory"],
        "output_kind": "normalized_polyline",
        "parser_backend": "deterministic",
        "input_modes": ["single"],
    },
    "state_image_match": {
        "description": "Generate the predicted spatial state and match it against candidate images.",
        "evidence": ["future state", "next view", "edited state"],
        "output_kind": "choice_label",
        "parser_backend": "fixed_model",
        "input_modes": ["single", "metadata_images"],
    },
    "grounded_dimension": {
        "description": "Outline measured object anchors, draw a dimension line, and attach a numeric unit label.",
        "evidence": ["object anchors", "dimension line", "numeric measurement"],
        "output_kind": "scalar_measurement",
        "parser_backend": "deterministic_ocr",
        "input_modes": ["single"],
        "parameters": {
            "anchor_count": {
                "type": "integer",
                "enum": [1, 2],
                "default": 2,
                "description": "1 for one object's extent; 2 for distance between two objects",
            },
            "unit": {"type": "string", "default": "cm"},
        },
    },
    "dual_anchor_relation": {
        "description": "Outline subject and reference objects with role colors and recover their geometry.",
        "evidence": ["subject anchor", "reference anchor", "2D relation"],
        "output_kind": "choice_label",
        "parser_backend": "deterministic",
        "input_modes": ["single"],
        "answer_schemas": ["choice_selection"],
        "parameters": {
            "relation_mode": {
                "type": "string",
                "enum": ["dominant_axis", "horizontal", "vertical", "distance"],
                "default": "dominant_axis",
            }
        },
    },
    "relation_zone_boolean": {
        "description": (
            "For a binary spatial-relation question, mark the subject and reference "
            "objects and render the full spatial region in which the subject would "
            "satisfy the queried relation. Recover yes/no by deterministic geometric "
            "membership, without a verdict symbol."
        ),
        "evidence": ["subject anchor", "reference anchor", "valid relation region"],
        "output_kind": "boolean",
        "parser_backend": "deterministic",
        "input_modes": ["single"],
        "answer_schemas": ["binary_boolean"],
    },
    "semantic_visual_choice": {
        "description": "Generate a clean visual realization of the selected spatial option and match it with fixed CLIP.",
        "evidence": ["selected spatial state", "visual option semantics"],
        "output_kind": "choice_label",
        "parser_backend": "fixed_model",
        "input_modes": ["single", "metadata_images"],
    },
    "marked_object_choice": {
        "description": (
            "Outline exactly one selected answer object, crop that grounded source object, and "
            "match it to textual object choices with fixed CLIP."
        ),
        "evidence": ["selected object outline", "object identity"],
        "output_kind": "choice_label",
        "parser_backend": "fixed_model",
        "input_modes": ["single"],
    },
}


REGISTERED_RECIPE_ROUTES = {
    "point_marker": ("agentic_point_marker", "cyan_point_marker"),
    "instance_marker_count": ("instance_marker_count", "green_star_count"),
    "region_mask": ("region_mask", "binary_target_mask"),
    "trajectory": ("trajectory", "red_motion_path"),
    "state_image_match": ("state_similarity", "next_view"),
}


@dataclass
class CompiledContract:
    task_config: Dict[str, Any] = field(default_factory=dict)
    artifact: Dict[str, Any] = field(default_factory=dict)
    route: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def recipe_inventory() -> List[Dict[str, Any]]:
    return [{"recipe": name, **spec} for name, spec in sorted(RECIPE_SPECS.items())]


def compile_contract(
    *,
    task: str,
    items: Sequence[Mapping[str, Any]],
    raw: Mapping[str, Any],
    input_mode: str,
    confidence: str,
    reason: str,
) -> CompiledContract:
    decision = str(raw.get("decision") or "").strip().lower()
    build_mode = str(raw.get("build_mode") or "").strip().lower()
    visual = dict(raw.get("visual_contract") or {})
    readout = dict(raw.get("readout") or {})
    if decision != "build":
        return CompiledContract(errors=[f"unsupported contract decision: {decision or '<empty>'}"])
    if build_mode == "recipe":
        return _compile_recipe(
            task=task,
            items=items,
            visual=visual,
            readout=readout,
            input_mode=input_mode,
            confidence=confidence,
            reason=reason,
        )
    if build_mode == "parser_ops":
        return _compile_direct_pipeline(
            task=task,
            items=items,
            raw=raw,
            visual=visual,
            readout=readout,
            input_mode=input_mode,
            confidence=confidence,
            reason=reason,
        )
    return CompiledContract(errors=["build_mode must be either 'recipe' or 'parser_ops'"])


def task_metric_contract(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    evaluations = [dict(item.get("evaluation") or {}) for item in items]
    metrics = {normalize_metric_name(row.get("metric")) for row in evaluations}
    metric = sorted(metrics)[0] if len(metrics) == 1 else "unverified"
    configs = [dict(row.get("metric_config") or {}) for row in evaluations if row.get("metric_config")]
    return {
        "metric": metric,
        "config": configs[0] if configs else {},
        "formal_evaluation": metric != "unverified",
    }


def ensure_choice_context(
    prompt: str,
    items: Sequence[Mapping[str, Any]],
) -> str:
    rendered = str(prompt or "").strip()
    has_choices = any(bool(item.get("choices") or item.get("options")) for item in items)
    if has_choices and "{choices}" not in rendered:
        rendered += "\n\nCandidate answers:\n{choices}"
    return rendered


def _compile_recipe(
    *,
    task: str,
    items: Sequence[Mapping[str, Any]],
    visual: Dict[str, Any],
    readout: Dict[str, Any],
    input_mode: str,
    confidence: str,
    reason: str,
) -> CompiledContract:
    recipe = str(readout.get("recipe") or visual.get("recipe") or "").strip().lower()
    spec = RECIPE_SPECS.get(recipe)
    if spec is None:
        return CompiledContract(errors=[f"unknown visual-contract recipe: {recipe or '<empty>'}"])
    if input_mode not in spec["input_modes"]:
        return CompiledContract(
            errors=[f"recipe {recipe} does not support input_mode={input_mode}"]
        )
    supported_answer_schemas = set(spec.get("answer_schemas") or [])
    task_answer_schemas = {infer_answer_schema(dict(item)) for item in items}
    unsupported_answer_schemas = task_answer_schemas - supported_answer_schemas
    if supported_answer_schemas and unsupported_answer_schemas:
        return CompiledContract(
            errors=[
                f"recipe {recipe} does not support answer schema(s): "
                + ", ".join(sorted(unsupported_answer_schemas))
            ]
        )
    parameters = dict(visual.get("parameters") or {})
    parameters.update(dict(readout.get("parameters") or {}))
    parameters, parameter_errors = _normalize_recipe_parameters(recipe, parameters)
    if parameter_errors:
        return CompiledContract(errors=parameter_errors)
    if recipe == "instance_marker_count" and _requires_occlusion_inference(items):
        if parameters.get("target_scope") != "visible_and_occlusion_inferred":
            return CompiledContract(
                errors=[
                    "instance_marker_count.target_scope must be "
                    "visible_and_occlusion_inferred for an occluded-total task"
                ]
            )
    metric_contract = task_metric_contract(items)
    metric = metric_contract["metric"]
    output_kind = str(spec["output_kind"])
    if metric != "unverified" and not metric_accepts_output(metric, output_kind):
        return CompiledContract(
            errors=[f"metric {metric!r} cannot consume recipe output {output_kind!r}"]
        )

    if recipe in REGISTERED_RECIPE_ROUTES:
        protocol, prompt_variant = REGISTERED_RECIPE_ROUTES[recipe]
        if (
            recipe == "instance_marker_count"
            and parameters.get("target_scope") == "visible_and_occlusion_inferred"
        ):
            prompt_variant = "green_star_amodal_count"
        task_config = {
            "protocol": protocol,
            "prompt_variant": prompt_variant,
            "input": {"mode": input_mode},
            "metric": metric,
            "metric_config": metric_contract["config"],
            "formal_evaluation": metric_contract["formal_evaluation"],
            "protocol_config": parameters,
        }
        artifact = _artifact(
            task,
            recipe,
            visual,
            {"recipe": recipe, "output_kind": output_kind},
            "",
            metric_contract,
            reason,
        )
    else:
        try:
            prompt, pipeline = _recipe_prompt_and_pipeline(recipe, parameters)
        except (TypeError, ValueError) as exc:
            return CompiledContract(errors=[f"invalid parameters for recipe {recipe}: {exc}"])
        prompt = ensure_choice_context(prompt, items)
        try:
            actual_output = DEFAULT_REGISTRY.output_kind(pipeline)
        except ParserPlanError as exc:
            return CompiledContract(errors=[f"compiled recipe pipeline is invalid: {exc}"])
        if actual_output != output_kind:
            return CompiledContract(
                errors=[f"recipe declares output {output_kind!r}, compiled {actual_output!r}"]
            )
        protocol_config = {
            **parameters,
            "parser_pipeline": pipeline,
            "parser_output_kind": output_kind,
            "metric": metric,
            "metric_config": metric_contract["config"],
        }
        task_config = {
            "protocol": "agentic_parser_ops_protocol",
            "prompt_variant": "generated",
            "prompt": prompt,
            "input": {"mode": input_mode},
            "metric": metric,
            "metric_config": metric_contract["config"],
            "formal_evaluation": metric_contract["formal_evaluation"],
            "protocol_config": protocol_config,
        }
        artifact = _artifact(
            task,
            recipe,
            visual,
            {"recipe": recipe, "output_kind": output_kind, "pipeline": pipeline},
            prompt,
            metric_contract,
            reason,
        )
    route = {
        "task": task,
        "decision": "build",
        "build_mode": "recipe",
        "active": True,
        "sample_count": len(items),
        "confidence": confidence,
        "recipe": recipe,
        "protocol": task_config["protocol"],
        "parser_backend": spec["parser_backend"],
        "parser_output_kind": output_kind,
        "metric": metric,
        "formal_evaluation": metric_contract["formal_evaluation"],
        "reason": reason,
    }
    return CompiledContract(task_config, artifact, route, [])


def _compile_direct_pipeline(
    *,
    task: str,
    items: Sequence[Mapping[str, Any]],
    raw: Mapping[str, Any],
    visual: Dict[str, Any],
    readout: Dict[str, Any],
    input_mode: str,
    confidence: str,
    reason: str,
) -> CompiledContract:
    prompt = str(raw.get("generation_prompt") or visual.get("generation_prompt") or "").strip()
    prompt = ensure_choice_context(prompt, items)
    pipeline = readout.get("pipeline") or raw.get("parser_pipeline")
    primitives = visual.get("primitives") or []
    errors = []
    if not prompt or "{question}" not in prompt:
        errors.append("generation_prompt must contain {question}")
    if re.search(r"\{\s*answer\s*\}|ground[- ]truth answer", prompt, flags=re.IGNORECASE):
        errors.append("generation_prompt may not reference the ground-truth answer")
    if not isinstance(primitives, list) or not primitives:
        errors.append("visual_contract.primitives must contain task-grounded visual evidence")
    if not isinstance(pipeline, Mapping):
        errors.append("readout.pipeline must be a Parser Ops mapping")
        output_kind = ""
    else:
        try:
            output_kind = DEFAULT_REGISTRY.output_kind(pipeline)
        except ParserPlanError as exc:
            errors.append(f"invalid Parser Ops pipeline: {exc}")
            output_kind = ""
    declared_output = str(readout.get("output_kind") or raw.get("output_kind") or "").strip()
    if declared_output and output_kind and declared_output != output_kind:
        errors.append(
            f"declared output_kind={declared_output!r} does not match pipeline output={output_kind!r}"
        )
    metric_contract = task_metric_contract(items)
    metric = metric_contract["metric"]
    if metric != "unverified" and output_kind and not metric_accepts_output(metric, output_kind):
        errors.append(f"metric {metric!r} cannot consume pipeline output {output_kind!r}")
    if errors:
        return CompiledContract(errors=errors)

    protocol_id = _protocol_id(task)
    protocol_config = {
        "generated_protocol_id": protocol_id,
        "parser_pipeline": dict(pipeline),
        "parser_output_kind": output_kind,
        "metric": metric,
        "metric_config": metric_contract["config"],
    }
    task_config = {
        "protocol": "agentic_parser_ops_protocol",
        "prompt_variant": "generated",
        "prompt": prompt,
        "input": {"mode": input_mode},
        "metric": metric,
        "metric_config": metric_contract["config"],
        "formal_evaluation": metric_contract["formal_evaluation"],
        "protocol_config": protocol_config,
    }
    artifact = {
        "id": protocol_id,
        "task": task,
        "decision": "build",
        "build_mode": "parser_ops",
        "visual_contract": visual,
        "generation_prompt": prompt,
        "readout": {"output_kind": output_kind, "pipeline": dict(pipeline)},
        "metric_contract": metric_contract,
        "rationale": reason,
    }
    route = {
        "task": task,
        "decision": "build",
        "build_mode": "parser_ops",
        "active": True,
        "sample_count": len(items),
        "confidence": confidence,
        "protocol_id": protocol_id,
        "protocol": "agentic_parser_ops_protocol",
        "parser_backend": "deterministic",
        "parser_output_kind": output_kind,
        "metric": metric,
        "formal_evaluation": metric_contract["formal_evaluation"],
        "reason": reason,
    }
    return CompiledContract(task_config, artifact, route, [])


def _recipe_prompt_and_pipeline(recipe: str, parameters: Mapping[str, Any]) -> tuple[str, Dict[str, Any]]:
    if recipe == "choice_count_board":
        prompt = (
            "Answer the counting question: \"{question}\" using the supplied image or ordered "
            "images jointly. Choices: {choices}. Create one clean white evidence board containing "
            "exactly one recognizable crop, thumbnail, or simple silhouette for every distinct "
            "target instance or functional region that you count. Merge duplicate views of the "
            "same target. Place exactly one large solid vivid-green circle (#00FF00) directly below "
            "each counted tile. Do not include uncounted target tiles, do not use green anywhere "
            "else, and do not write a number, option label, or answer text. Output only the evidence "
            "board."
        )
        return prompt, green_choice_count_board_pipeline()
    if recipe == "grounded_dimension":
        anchor_count = int(parameters.get("anchor_count", 2))
        if anchor_count == 1:
            prompt = (
                "Answer the metric spatial question: \"{question}\". Edit the source image minimally. "
                "Outline the measured object with one thick solid magenta contour (#FF00FF). Draw one "
                "thick solid yellow double-headed dimension line (#FFFF00) exactly across the requested "
                "width or height. Place one high-contrast white measurement tag directly beside the line "
                "containing only the estimated number and its physical unit. Do not place the answer "
                "elsewhere and add no other magenta or yellow marks. Output only the annotated image."
            )
        else:
            prompt = (
                "Answer the metric spatial question: \"{question}\". Edit the source image minimally. "
                "Outline the first measured object with one thick solid magenta contour (#FF00FF) and the "
                "second measured object with one thick solid cyan contour (#00FFFF). Draw one thick solid "
                "yellow double-headed dimension line (#FFFF00) between the requested nearest boundaries. "
                "Place one high-contrast white measurement tag directly beside the line containing only "
                "the estimated number and its physical unit. Do not place the answer elsewhere and add no "
                "other cyan, magenta, or yellow marks. Output only the annotated image."
            )
        return prompt, grounded_dimension_pipeline(anchor_count=anchor_count)
    if recipe == "dual_anchor_relation":
        mode = str(parameters.get("relation_mode", "dominant_axis"))
        prompt = (
            "Solve the spatial relation in: \"{question}\". Keep the source image unchanged except for "
            "two thick outlines. Outline the subject named by the question in solid magenta (#FF00FF), "
            "and outline the reference object in solid cyan (#00FFFF). Use exactly one outline for each "
            "role. Do not write an answer label or explanatory text. Output only the annotated image."
        )
        return prompt, dual_anchor_relation_pipeline(mode=mode, map_to_choices=True)
    if recipe == "relation_zone_boolean":
        prompt = (
            "Answer the binary spatial-relation question: \"{question}\". Keep the source image "
            "unchanged. Place one solid magenta disk (#FF00FF) at the center of the subject object "
            "and one solid cyan disk (#00FFFF) at the center of the reference object. Overlay one "
            "large translucent vivid-green filled region (#00FF00) covering the image locations "
            "where the subject would satisfy exactly the relation asked in the question. Keep the "
            "two disks visible above the region. Do not move either object. Do not add a check mark, "
            "X, yes/no text, option label, or other verdict symbol. Output only the annotated image."
        )
        return prompt, relation_zone_boolean_pipeline()
    if recipe == "semantic_visual_choice":
        prompt = (
            "Use the supplied image or ordered images to answer this spatial task: \"{question}\". "
            "Choices: {choices}. Produce one clean visual realization of the selected option's actual "
            "spatial state, relation, action, or result. Do not write option letters, answer labels, or "
            "explanatory text. Preserve recognizable task-relevant objects. Output only the visual answer."
        )
        return prompt, semantic_choice_clip_pipeline()
    if recipe == "marked_object_choice":
        prompt = (
            "Answer the object-selection spatial task: \"{question}\". Choices: {choices}. Keep "
            "the source image unchanged except for one thick solid magenta outline (#FF00FF) "
            "around exactly the selected answer object. Do not outline any other object. Do not "
            "write an option letter, object name, answer label, or explanation. Output only the "
            "annotated source image."
        )
        return prompt, marked_object_choice_pipeline()
    raise ValueError(f"Recipe {recipe!r} has no compiler")


def _normalize_recipe_parameters(
    recipe: str, parameters: Mapping[str, Any]
) -> tuple[Dict[str, Any], List[str]]:
    normalized = dict(parameters)
    errors: List[str] = []
    if recipe == "grounded_dimension":
        raw_anchor_count = normalized.get("anchor_count", 2)
        if isinstance(raw_anchor_count, bool):
            anchor_count = None
        elif isinstance(raw_anchor_count, int):
            anchor_count = raw_anchor_count
        else:
            match = re.fullmatch(r"\s*([12])(?:\s+[^\d].*)?\s*", str(raw_anchor_count))
            anchor_count = int(match.group(1)) if match else None
        if anchor_count not in {1, 2}:
            errors.append("grounded_dimension.anchor_count must be integer 1 or 2")
        else:
            normalized["anchor_count"] = anchor_count
        unit = str(normalized.get("unit", "cm") or "cm").strip().lower()
        if not re.fullmatch(r"[a-z]{1,12}", unit):
            errors.append("grounded_dimension.unit must be a short alphabetic physical unit")
        else:
            normalized["unit"] = unit
    elif recipe == "dual_anchor_relation":
        mode = str(normalized.get("relation_mode", "dominant_axis") or "").strip().lower()
        allowed = {"dominant_axis", "horizontal", "vertical", "distance"}
        if mode not in allowed:
            errors.append(
                "dual_anchor_relation.relation_mode must be one of " + ", ".join(sorted(allowed))
            )
        else:
            normalized["relation_mode"] = mode
    elif recipe == "instance_marker_count":
        target_scope = str(normalized.get("target_scope", "visible") or "").strip().lower()
        allowed = {"visible", "visible_and_occlusion_inferred"}
        if target_scope not in allowed:
            errors.append(
                "instance_marker_count.target_scope must be one of "
                + ", ".join(sorted(allowed))
            )
        else:
            normalized["target_scope"] = target_scope
    return normalized, errors


def _requires_occlusion_inference(items: Sequence[Mapping[str, Any]]) -> bool:
    patterns = (
        "continues behind",
        "continue behind",
        "as if the black box were not there",
        "including those hidden",
        "including hidden",
        "occluded total",
    )
    return any(
        any(pattern in str(item.get("question") or "").lower() for pattern in patterns)
        for item in items
    )


def _artifact(
    task: str,
    recipe: str,
    visual: Mapping[str, Any],
    readout: Mapping[str, Any],
    prompt: str,
    metric_contract: Mapping[str, Any],
    reason: str,
) -> Dict[str, Any]:
    return {
        "id": _protocol_id(task),
        "task": task,
        "decision": "build",
        "build_mode": "recipe",
        "recipe": recipe,
        "visual_contract": dict(visual),
        "generation_prompt": prompt,
        "readout": dict(readout),
        "metric_contract": dict(metric_contract),
        "rationale": reason,
    }


def _protocol_id(task: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(task).strip().lower()).strip("_")
    return f"generated_{slug or 'task'}_v2"
