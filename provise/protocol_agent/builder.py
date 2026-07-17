from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from provise.benchmark.media import input_media_entries, primary_media_path, resolve_path
from provise.evaluation.metrics import metric_accepts_output
from provise.reporting import NullProgressReporter, ProgressReporter
from provise.parser_ops import DEFAULT_REGISTRY
from provise.protocol_agent.visual_contract import (
    compile_contract,
    ensure_choice_context,
    recipe_inventory,
    task_metric_contract,
)
from provise.protocols.fallback import expected_answer_format
from provise.benchmark.tasks import infer_answer_schema
from provise.benchmark.schema import canonicalize_unified_item, load_unified_items


DECISIONS = {"reuse", "build", "fallback", "unsupported"}
NON_SPATIAL_REUSE_PROTOCOLS = {
    "generic_vlm_fallback",
    "label_code",
}
PROTOCOL_OUTPUT_KINDS = {
    "agentic_point_marker": "normalized_point",
    "binary_color_presence": "boolean",
    "dense_depth_ab": "label",
    "direction_grid": "label",
    "instance_marker_count": "integer_count",
    "region_mask": "mask_path",
    "state_similarity": "choice_label",
    "trajectory": "normalized_polyline",
}
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
SPATIAL_EVIDENCE_TERMS = {
    "arrow",
    "boundary",
    "box",
    "contour",
    "correspondence",
    "depth",
    "distance",
    "dot",
    "extent",
    "heatmap",
    "hidden region",
    "highlight",
    "layout",
    "line",
    "marker",
    "mask",
    "measurement",
    "component",
    "instance",
    "object",
    "outline",
    "overlay",
    "path",
    "point",
    "region",
    "size bar",
    "state",
    "trajectory",
}
VLM_SERIALIZABLE_OUTPUT_KINDS = {
    "boolean",
    "choice_label",
    "integer_count",
    "label",
    "normalized_bbox",
    "normalized_point",
    "normalized_points",
    "normalized_polyline",
    "scalar",
    "scalar_measurement",
    "text",
}


AGENT_SYSTEM_PROMPT = """You are ProVisE's autonomous visual-protocol agent.

You are designing an evaluation interface for an image-generation model. You
are NOT solving the attached benchmark examples, and their ground-truth answers
are deliberately hidden from you.

Choose exactly one decision for this task:

1. reuse
   Reuse an existing protocol only when its visual output, parser contract,
   answer format, input structure, and official metric genuinely match.
2. build
   Build a deterministic protocol. Set build_mode="recipe" to configure one
   registered visual-contract recipe. When no recipe fits, set
   build_mode="parser_ops" and provide a task-grounded generation prompt plus a
   typed graph made only from registered Parser Ops. Never write Python code.
3. fallback
   Use a task-level VLM parser only when recipes, Parser Ops, OCR, and fixed
   similarity models cannot preserve the answer contract.
4. unsupported
   Use only for a hard runtime boundary: unsupported input media, missing task
   output semantics, or a metric/output contract that no available evaluator
   can consume. A deterministic parser gap must use fallback instead.

Principles:
- Prefer reuse over construction when a spatial-evidence protocol truly fits.
- Registered Parser Ops protocols are the primary parsing path. Select and
  configure only the tested operators exposed in the protocol inventory; never
  invent Python code or unregistered operators.
- Prefer build with a recipe, then build with Parser Ops, then fallback. Do not
  skip directly to fallback when a deterministic recipe or registered Parser
  Ops plan can preserve the task.
- A task with metric=unverified is smoke-only; metric verification does not
  determine whether a visual answer can be generated or parsed. Never choose
  unsupported or fallback solely because a metric is unverified.
- For a numeric-choice counting question over one or more views, prefer the
  choice_count_board recipe. It requires visible counted evidence and maps the
  recovered marker count to the numeric choice without an answer code or VLM.
- For an occluded-total counting task, a visible-only marker recipe is invalid.
  If instance_marker_count is used, explicitly set
  target_scope=visible_and_occlusion_inferred so the generated image marks both
  visible objects and inferred hidden positions before counting.
- For a semantic choice over supported images, consider semantic_visual_choice
  before fallback. Its fixed CLIP readout is mechanically testable; long choice
  text alone is not a reason to declare the task unsupported.
- Never construct a protocol merely because the task is multiple choice.
- Spatial evidence must come first. A, B, or other answer labels may be absent
  or may appear only as a secondary cue attached to the evidence.
- Reject pure answer-code mappings such as corners, slots, colored codes, or
  writing the option label without task-relevant marks.
- A check mark, X, traffic-light color, thumbs-up/down, or other generic verdict
  symbol is also an answer code when it alone distinguishes yes from no. For a
  binary spatial task, depict the relation, valid region, fit, collision, or
  resulting state itself so the readout can recover the boolean geometrically.
- Use large, sparse edits that current image editors can generate reliably.
- Do not require tiny coordinates, exact typography, dense diagrams, or a long
  prose explanation.
- The parser must read what the generated image expresses; it must not solve the
  original task itself.
- For a choice task, it is valid to generate the selected option's visible,
  task-relevant object, relation, path, or resulting state and let the parser
  match that generated evidence to the option semantics. This is not an answer
  code. The parser must still avoid inferring the answer from the source alone.
- The protocol must be information-complete for every answer schema listed in
  the task payload. Marking one object can recover an object-choice answer, but
  cannot by itself recover a yes/no comparison; expose comparative spatial
  evidence or another grounded readout that distinguishes both outcomes.
- Follow the official metric contract rather than the annotation's display
  cardinality. For point_in_mask, one recovered point preserves the metric when
  the score accepts any predicted point inside the target mask; a source answer
  listing several valid points does not by itself require a multi-point visual
  protocol. In that case, prefer the reusable deterministic point-marker
  protocol when its other input and output requirements match.
- Do not put a ground-truth answer or an {answer} placeholder in either prompt.
- Use metadata_images when any sample in the task has multiple unified image
  inputs; it also supports the task's single-image samples.
- Use unsupported honestly. Coverage is less important than valid evaluation.

Return ONLY one JSON object, without markdown fences. Use one of these shapes:
{
  "task": "task name exactly as supplied",
  "decision": "reuse | build | fallback | unsupported",
  "build_mode": "recipe | parser_ops; required only for build",
  "confidence": "high | medium | low",
  "reason": "concise decision rationale",
  "reuse": {
    "protocol": "existing protocol name",
    "prompt_variant": "existing prompt variant",
    "protocol_config": {}
  },
  "visual_contract": {
    "recipe": "required for build_mode=recipe",
    "mode": "edit_source | reference_synthesis",
    "primitives": ["task-grounded visible primitive"],
    "parameters": {}
  },
  "readout": {
    "recipe": "same registered recipe for build_mode=recipe",
    "output_kind": "required for build_mode=parser_ops",
    "pipeline": {"steps": [], "output": "step id; required for build_mode=parser_ops"}
  },
  "generation_prompt": "required for build_mode=parser_ops; must contain {question}",
  "fallback": {
    "generation_prompt": "task-grounded prompt containing {question}",
    "parse_prompt": "read only what the generated image expresses",
    "visual_evidence": "required spatial evidence",
    "parser_observation": "concrete visible marks or generated state to inspect",
    "invalid_conditions": ["at least two conditions"]
  }
}

Include only the object matching the chosen decision; omit the other object.
"""


AUTOMATIC_FALLBACK_GENERATION_PROMPT = """Use the supplied source image or ordered images to produce one clear visual answer to this spatial task:
{question}

Candidate answers, when present:
{choices}

Express the answer through large, task-relevant visual evidence in the image itself. Use only the primitives that fit the task, such as object outlines, a region overlay, arrows between referenced objects, a path, measurement marks, correspondence marks, or a visibly rendered next spatial state. Preserve unrelated scene content. Do not merely print an answer label, option text, or prose explanation."""


AUTOMATIC_FALLBACK_PARSE_PROMPT = """Read only the concrete spatial evidence expressed in the generated visual response. Identify the marked objects, regions, arrows, path, measurement, correspondence, or generated state, then recover the answer in the benchmark's required format. Candidate choices may be used only to map that visible evidence to a label. Do not independently solve the original task from the source image and do not accept a bare answer label as evidence."""


AUTOMATIC_FALLBACK_VISUAL_EVIDENCE = (
    "task-grounded object outlines, region overlays, arrows, paths, measurement marks, "
    "correspondence marks, or a visibly changed spatial state"
)


@dataclass
class AgenticProtocolBuildResult:
    benchmark_name: str
    benchmark_config: Dict[str, Any]
    generated_protocols: Dict[str, Any]
    manifest: Dict[str, Any]
    prompt: str
    raw_response: str


class AgenticProtocolBuilder:
    def __init__(
        self,
        items: Iterable[Dict[str, Any]],
        *,
        benchmark_name: str,
        data_file: str,
        benchmark_root: str,
        max_examples_per_task: int = 3,
        max_media_per_task: int = 8,
        protocol_spec_dir: str | Path = "configs/protocol_specs",
        reporter: ProgressReporter | None = None,
    ):
        self.items = [canonicalize_unified_item(item) for item in items]
        self.benchmark_name = benchmark_name
        self.data_file = data_file
        self.benchmark_root = benchmark_root
        self.max_examples_per_task = max(1, int(max_examples_per_task))
        self.max_media_per_task = max(0, int(max_media_per_task))
        self.protocol_specs = load_protocol_catalog(protocol_spec_dir)
        self.reporter = reporter or NullProgressReporter()

    def build(self, *, vlm: Any | None = None, raw_response: str = "") -> AgenticProtocolBuildResult:
        groups = group_items_by_task(self.items)
        self.reporter.emit(
            f"Protocol planning started: {len(groups)} task(s), {len(self.items)} sample(s)",
            event="protocol_planning_started",
            task_count=len(groups),
            sample_count=len(self.items),
        )
        prompts: Dict[str, str] = {}
        contexts: Dict[str, Dict[str, Any]] = {}
        warnings: List[str] = []

        if raw_response:
            for task, task_items in sorted(groups.items()):
                representatives = select_representative_items(task_items, self.max_examples_per_task)
                _, attachments, missing_media = self._visual_context(representatives)
                prompts[task] = self._build_task_prompt(task, task_items, representatives, attachments)
                contexts[task] = {
                    "representative_sample_ids": [str(item.get("id") or "") for item in representatives],
                    "attached_media_count": len(attachments),
                    "attached_media": attachments,
                    "missing_media": missing_media,
                }
            prompt = join_task_artifacts(prompts)
            try:
                payload = normalize_agent_payload(parse_agent_json_response(raw_response))
            except ValueError as exc:
                payload = {"benchmark": self.benchmark_name, "tasks": []}
                warnings.append(f"Agent did not return valid JSON: {type(exc).__name__}: {exc}")
            response_artifact = raw_response
        else:
            if vlm is None:
                raise ValueError("Either vlm or raw_response is required")
            task_rows: List[Dict[str, Any]] = []
            raw_by_task: Dict[str, str] = {}
            for task, task_items in sorted(groups.items()):
                representatives = select_representative_items(task_items, self.max_examples_per_task)
                image_paths, attachments, missing_media = self._visual_context(representatives)
                task_prompt = self._build_task_prompt(task, task_items, representatives, attachments)
                prompts[task] = task_prompt
                contexts[task] = {
                    "representative_sample_ids": [str(item.get("id") or "") for item in representatives],
                    "attached_media_count": len(image_paths),
                    "attached_media": attachments,
                    "missing_media": missing_media,
                }
                try:
                    self.reporter.emit(
                        f"Planning task from {len(representatives)} representative sample(s) and "
                        f"{len(image_paths)} image(s)",
                        event="task_planning_started",
                        task=task,
                        sample_count=len(task_items),
                    )
                    with self.reporter.waiting(
                        "Waiting for protocol agent",
                        event="protocol_agent_call",
                        task=task,
                    ):
                        response = (
                            vlm.predict_multi(image_paths, task_prompt)
                            if hasattr(vlm, "predict_multi")
                            else vlm.predict(image_paths[0] if image_paths else "", task_prompt)
                        )
                    raw_by_task[task] = str(response or "")
                    parsed = parse_agent_json_response(str(response or ""))
                    row = extract_task_row(parsed, task)
                    if row is None:
                        warnings.append(f"Agent response omitted task={task}; marked unsupported")
                    else:
                        task_rows.append(row)
                        agent_decision = infer_decision(row)
                        if agent_decision == "unsupported":
                            decision_message = (
                                "No deterministic route proposed; VLM fallback will be validated"
                            )
                            terminal_decision = "fallback"
                        else:
                            build_mode = str(row.get("build_mode") or "").strip()
                            decision_label = agent_decision or "invalid"
                            if agent_decision == "build" and build_mode:
                                decision_label += f" ({build_mode})"
                            decision_message = f"Agent decision: {decision_label}"
                            terminal_decision = agent_decision
                        self.reporter.emit(
                            decision_message,
                            event="task_agent_decision",
                            status="completed",
                            task=task,
                            decision=terminal_decision,
                            agent_decision=agent_decision,
                            build_mode=str(row.get("build_mode") or ""),
                            confidence=normalize_confidence(row.get("confidence")),
                        )
                except Exception as exc:
                    warnings.append(f"Agent call failed for task={task}: {type(exc).__name__}: {exc}")
                    self.reporter.emit(
                        f"Protocol agent failed: {type(exc).__name__}: {exc}",
                        event="task_agent_failed",
                        status="failed",
                        task=task,
                    )

            payload = {"benchmark": self.benchmark_name, "tasks": task_rows}
            prompt = join_task_artifacts(prompts)
            response_artifact = json.dumps(
                {"benchmark": self.benchmark_name, "task_responses": raw_by_task},
                ensure_ascii=False,
                indent=2,
            )

        generated_protocols, tasks_cfg, rows, validation_warnings = self._validate_payload(payload, groups)
        warnings.extend(validation_warnings)
        decision_counts = dict(sorted(Counter(row["decision"] for row in rows).items()))
        active_sample_count = sum(row["sample_count"] for row in rows if row.get("active"))
        benchmark_config = {
            "benchmark": self.benchmark_name,
            "data_file": self.data_file,
            "benchmark_root": self.benchmark_root,
            "tasks": tasks_cfg,
        }
        manifest = {
            "benchmark": self.benchmark_name,
            "builder": "agentic_protocol_builder_v2",
            "total_samples": len(self.items),
            "task_count": len(groups),
            "active_task_count": len(tasks_cfg),
            "active_sample_count": active_sample_count,
            "decision_counts": decision_counts,
            "route_rows": rows,
            "task_contexts": contexts,
            "task_contracts": {
                task: summarize_task_contract(task, task_items)
                for task, task_items in sorted(groups.items())
            },
            "warnings": warnings,
        }
        self.reporter.emit(
            f"Protocol planning completed: {len(tasks_cfg)} active / {len(groups)} total",
            event="protocol_planning_completed",
            status="completed",
            active_task_count=len(tasks_cfg),
            task_count=len(groups),
            decision_counts=decision_counts,
        )
        return AgenticProtocolBuildResult(
            benchmark_name=self.benchmark_name,
            benchmark_config=benchmark_config,
            generated_protocols={"protocols": generated_protocols},
            manifest=manifest,
            prompt=prompt,
            raw_response=response_artifact,
        )

    def _build_task_prompt(
        self,
        task: str,
        task_items: List[Dict[str, Any]],
        representatives: List[Dict[str, Any]],
        attachments: List[Dict[str, Any]],
    ) -> str:
        examples = []
        attachment_indexes: Dict[str, List[int]] = defaultdict(list)
        for attachment in attachments:
            attachment_indexes[str(attachment["sample_id"])].append(int(attachment["image_index"]))
        for item in representatives:
            examples.append(summarize_item(item, attachment_indexes.get(str(item.get("id") or ""), [])))
        payload = {
            "benchmark": self.benchmark_name,
            "task": task,
            "sample_count": len(task_items),
            "source_metrics": sorted(task_metrics(task_items)),
            "task_answer_schemas": sorted({infer_answer_schema(item) for item in task_items}),
            "observed_input_modes": sorted({infer_input_mode(item) for item in task_items}),
            "inferred_input_mode": infer_task_input_mode(task_items),
            "representative_examples_without_ground_truth": examples,
            "visual_attachment_order": attachments,
        }
        return (
            AGENT_SYSTEM_PROMPT
            + "\nExisting reusable spatial-evidence protocols:\n"
            + json.dumps(protocol_inventory(self.protocol_specs), ensure_ascii=False, indent=2)
            + "\nComposable visual-contract recipes:\n"
            + json.dumps(recipe_inventory(), ensure_ascii=False, indent=2)
            + "\nWhitelisted Parser Ops for build_mode=parser_ops:\n"
            + json.dumps(DEFAULT_REGISTRY.inventory(), ensure_ascii=False, indent=2)
            + "\nTask payload:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )

    def revise_task(
        self,
        *,
        task: str,
        vlm: Any,
        previous_response: str,
        diagnostics: Dict[str, Any],
        generated_paths: Iterable[str] = (),
    ) -> str:
        groups = group_items_by_task(self.items)
        if task not in groups:
            raise ValueError(f"Cannot revise unknown task: {task}")
        task_items = groups[task]
        representatives = select_representative_items(task_items, self.max_examples_per_task)
        image_paths, attachments, _ = self._visual_context(representatives)
        failed_images = [
            str(path)
            for path in generated_paths
            if str(path) and os.path.exists(str(path)) and str(path) not in image_paths
        ]
        available_slots = max(0, self.max_media_per_task - len(image_paths))
        failed_images = failed_images[:available_slots]
        revision_prompt = (
            self._build_task_prompt(task, task_items, representatives, attachments)
            + "\n\nREVISION REQUEST\n"
            + "The first protocol attempt did not pass deterministic framework validation or "
            + "mechanical smoke validation. Ground-truth answers, correctness, and benchmark scores "
            + "are intentionally excluded. Revise the protocol once using only the diagnostics below. "
            + "Fix the visual contract, prompt, parser recipe, or Parser Ops parameters. If no "
            + "deterministic or fixed-model readout can work, return decision=fallback with a "
            + "task-grounded visual response and VLM parse contract. Do not tune toward an answer.\n"
            + "Previous response:\n"
            + str(previous_response or "")[:12000]
            + "\nMechanical diagnostics:\n"
            + json.dumps(sanitize_revision_diagnostics(diagnostics), ensure_ascii=False, indent=2)
        )
        if failed_images:
            revision_prompt += (
                "\nThe final attached image(s) are failed generated outputs in the same order as "
                "diagnostics.failed_samples. Inspect only protocol conformance, not correctness."
            )
        all_images = image_paths + failed_images
        self.reporter.emit(
            "Requesting one bounded protocol revision",
            event="protocol_revision_started",
            task=task,
            failed_image_count=len(failed_images),
        )
        with self.reporter.waiting(
            "Waiting for protocol revision agent",
            event="protocol_revision_agent_call",
            task=task,
        ):
            response = (
                vlm.predict_multi(all_images, revision_prompt)
                if hasattr(vlm, "predict_multi")
                else vlm.predict(all_images[0] if all_images else "", revision_prompt)
            )
        parsed = parse_agent_json_response(str(response or ""))
        row = extract_task_row(parsed, task)
        if row is None:
            raise ValueError(f"Revision response omitted task={task}")
        self.reporter.emit(
            f"Revision decision: {infer_decision(row) or 'invalid'}",
            event="protocol_revision_completed",
            status="completed",
            task=task,
            decision=infer_decision(row),
        )
        return str(response or "")

    def activate_automatic_vlm_fallback(
        self,
        result: AgenticProtocolBuildResult,
        *,
        task: str,
        origin: str,
        reason: str,
    ) -> tuple[bool, List[str]]:
        groups = group_items_by_task(self.items)
        items = groups.get(task) or []
        if not items:
            return False, [f"unknown task: {task}"]
        unsupported_media = unsupported_media_kinds(items)
        if unsupported_media:
            return False, [f"unsupported media: {unsupported_media}"]
        metrics = task_metrics(items)
        if len(metrics) > 1:
            return False, [f"mixed metrics cannot share one fallback: {sorted(metrics)}"]

        current_config = (result.benchmark_config.get("tasks") or {}).get(task) or {}
        current_protocol = next(
            (
                row
                for row in result.generated_protocols.get("protocols", [])
                if row.get("task") == task and row.get("decision") != "fallback"
            ),
            {},
        )
        preserved_generation_prompt = str(
            current_config.get("prompt") or current_protocol.get("generation_prompt") or ""
        ).strip()
        visual_contract = current_protocol.get("visual_contract") or {}
        prior_primitives = visual_contract.get("primitives") or []
        if isinstance(prior_primitives, str):
            prior_primitives = [prior_primitives]
        prior_visual_evidence = normalize_text_or_list(
            current_protocol.get("visual_evidence") or prior_primitives
        )
        prompt_evidence_terms = concrete_spatial_evidence_terms(preserved_generation_prompt)
        if not concrete_spatial_evidence_terms(prior_visual_evidence) and prompt_evidence_terms:
            prior_visual_evidence = normalize_text_or_list(
                [prior_visual_evidence, "visible " + ", ".join(prompt_evidence_terms)]
            )
        fallback_visual_evidence = (
            prior_visual_evidence
            if concrete_spatial_evidence_terms(prior_visual_evidence)
            else AUTOMATIC_FALLBACK_VISUAL_EVIDENCE
        )
        preserve_generation = bool(
            preserved_generation_prompt
            and "{question}" in preserved_generation_prompt
            and prompt_evidence_terms
        )

        raw = {
            "task": task,
            "decision": "fallback",
            "confidence": "medium",
            "reason": reason,
            "fallback": {
                "generation_prompt": (
                    preserved_generation_prompt
                    if preserve_generation
                    else AUTOMATIC_FALLBACK_GENERATION_PROMPT
                ),
                "parse_prompt": AUTOMATIC_FALLBACK_PARSE_PROMPT,
                "visual_strategy": str(
                    current_protocol.get("visual_strategy") or "other_spatial"
                ),
                "visual_evidence": fallback_visual_evidence,
                "parser_observation": (
                    "Read only the visible task-grounded "
                    + fallback_visual_evidence
                    + ", then map that evidence to the required answer schema."
                ),
                "invalid_conditions": [
                    "no task-relevant spatial mark or generated state is visible",
                    "the visible evidence is ambiguous or contradicts itself",
                    "the response contains only an answer label or prose",
                ],
            },
        }
        config, protocol, route, errors = self._validate_v2_fallback(
            task,
            items,
            raw,
            "medium",
            reason,
        )
        if errors:
            return False, errors

        if preserve_generation:
            config["protocol_config"]["fallback_preserved_generation"] = True
            protocol["generation_protocol_preserved"] = True
            route["generation_protocol_preserved"] = True

        prior_route = next(
            (row for row in result.manifest.get("route_rows", []) if row.get("task") == task),
            {},
        )
        route.update(
            {
                "source": "automatic_vlm_fallback",
                "fallback_origin": origin,
                "pre_fallback_decision": prior_route.get("decision", ""),
                "pre_fallback_source": prior_route.get("source", ""),
                "pre_fallback_reason": prior_route.get("reason", ""),
            }
        )
        route_rows = result.manifest.setdefault("route_rows", [])
        replaced = False
        for index, row in enumerate(route_rows):
            if row.get("task") == task:
                route_rows[index] = route
                replaced = True
                break
        if not replaced:
            route_rows.append(route)

        result.benchmark_config.setdefault("tasks", {})[task] = config
        protocols = result.generated_protocols.setdefault("protocols", [])
        protocols[:] = [
            row
            for row in protocols
            if not (row.get("task") == task and row.get("decision") == "fallback")
        ]
        protocols.append(protocol)
        fallback_tasks = set(result.manifest.get("automatic_vlm_fallback_tasks") or [])
        fallback_tasks.add(task)
        result.manifest["automatic_vlm_fallback_tasks"] = sorted(fallback_tasks)
        result.manifest.setdefault("fallback_history", []).append(
            {"task": task, "origin": origin, "reason": reason}
        )
        result.manifest.setdefault("warnings", []).append(
            f"Task {task} entered automatic VLM fallback after {origin}."
        )
        refresh_build_manifest_counts(result)
        return True, []

    def _visual_context(
        self, representatives: List[Dict[str, Any]]
    ) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
        image_paths: List[str] = []
        attachments: List[Dict[str, Any]] = []
        missing: List[str] = []
        for item in representatives:
            sample_id = str(item.get("id") or "")
            entries = input_media_entries(item)
            if not entries:
                legacy = primary_media_path(item)
                entries = [{"path": legacy, "role": "primary"}] if legacy else []
            for media_index, entry in enumerate(entries, 1):
                if len(image_paths) >= self.max_media_per_task:
                    break
                raw_path = str(entry.get("path") or "")
                if not raw_path:
                    continue
                path = resolve_path(self.benchmark_root, raw_path)
                if Path(path).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if not os.path.exists(path):
                    missing.append(path)
                    continue
                image_paths.append(path)
                attachments.append(
                    {
                        "image_index": len(image_paths),
                        "sample_id": sample_id,
                        "media_index": media_index,
                        "role": str(entry.get("role") or ""),
                        "label": str(entry.get("label") or ""),
                    }
                )
        return image_paths, attachments, missing

    def _validate_payload(
        self,
        payload: Dict[str, Any],
        groups: Dict[str, List[Dict[str, Any]]],
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], List[str]]:
        raw_tasks = payload.get("tasks") if isinstance(payload, dict) else []
        if not isinstance(raw_tasks, list):
            raw_tasks = []
        by_task = {
            str(row.get("task") or "").strip(): row
            for row in raw_tasks
            if isinstance(row, dict) and str(row.get("task") or "").strip()
        }
        generated_protocols: List[Dict[str, Any]] = []
        tasks_cfg: Dict[str, Any] = {}
        rows: List[Dict[str, Any]] = []
        warnings: List[str] = []

        for task, items in sorted(groups.items()):
            raw = by_task.get(task)
            metrics = task_metrics(items)
            unsupported_media = unsupported_media_kinds(items)
            if unsupported_media:
                reason = f"current image-generation interface does not support media: {unsupported_media}"
                rows.append(unsupported_route(task, items, reason, "framework_input_validation"))
                warnings.append(f"Task {task} is unsupported: {reason}")
                continue
            if len(metrics) > 1:
                reason = f"task mixes incompatible source metrics: {sorted(metrics)}"
                rows.append(unsupported_route(task, items, reason, "framework_metric_validation"))
                warnings.append(f"Task {task} is unsupported: {reason}")
                continue
            if raw is None:
                reason = "agent returned no valid decision"
                rows.append(unsupported_route(task, items, reason, "agent_missing"))
                warnings.append(f"No agent decision for task={task}; task disabled")
                continue

            decision = infer_decision(raw)
            confidence = normalize_confidence(raw.get("confidence"))
            reason = str(raw.get("reason") or "").strip()
            if decision not in DECISIONS:
                rows.append(unsupported_route(task, items, "invalid agent decision", "framework_validation"))
                warnings.append(f"Invalid agent decision for task={task}; task disabled")
                continue
            if decision == "unsupported":
                rows.append(unsupported_route(task, items, reason or "agent judged task unsuitable", "agent"))
                continue
            if confidence == "low":
                rows.append(
                    unsupported_route(
                        task,
                        items,
                        reason or "agent confidence was low",
                        "low_confidence",
                    )
                )
                warnings.append(f"Low-confidence route for task={task}; task disabled")
                continue

            if decision == "reuse":
                config, route, error = self._validate_reuse(task, items, raw, confidence, reason)
                if error:
                    rows.append(unsupported_route(task, items, error, "framework_validation"))
                    warnings.append(f"Rejected reuse route for task={task}: {error}")
                    continue
                tasks_cfg[task] = config
                rows.append(route)
                continue

            if decision == "build":
                compiled = compile_contract(
                    task=task,
                    items=items,
                    raw=raw,
                    input_mode=infer_task_input_mode(items),
                    confidence=confidence,
                    reason=reason,
                )
                if compiled.errors:
                    error = "; ".join(compiled.errors)
                    rows.append(unsupported_route(task, items, error, "contract_compiler"))
                    warnings.append(f"Rejected visual contract for task={task}: {error}")
                    continue
                tasks_cfg[task] = compiled.task_config
                if compiled.artifact:
                    generated_protocols.append(compiled.artifact)
                rows.append(compiled.route)
                continue

            if decision == "fallback":
                config, protocol, route, errors = self._validate_v2_fallback(
                    task, items, raw, confidence, reason
                )
                if errors:
                    error = "; ".join(errors)
                    rows.append(unsupported_route(task, items, error, "fallback_validation"))
                    warnings.append(f"Rejected VLM fallback for task={task}: {error}")
                    continue
                tasks_cfg[task] = config
                generated_protocols.append(protocol)
                rows.append(route)
                continue

        unknown_tasks = sorted(set(by_task) - set(groups))
        for task in unknown_tasks:
            warnings.append(f"Agent returned unknown task={task!r}; ignored")
        return generated_protocols, tasks_cfg, rows, warnings

    def _validate_reuse(
        self,
        task: str,
        items: List[Dict[str, Any]],
        raw: Dict[str, Any],
        confidence: str,
        reason: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any], str]:
        reuse = raw.get("reuse") if isinstance(raw.get("reuse"), dict) else raw
        protocol_name = str(reuse.get("protocol") or reuse.get("protocol_name") or "").strip()
        prompt_variant = str(reuse.get("prompt_variant") or "").strip()
        if protocol_name in NON_SPATIAL_REUSE_PROTOCOLS:
            return {}, {}, f"{protocol_name} is an answer-code/fallback protocol, not spatial evidence"
        spec = self.protocol_specs.get(protocol_name)
        if spec is None:
            return {}, {}, f"unknown reusable protocol: {protocol_name or '<empty>'}"
        variants = spec.get("prompts") or {}
        if prompt_variant not in variants:
            return {}, {}, f"unknown prompt variant {prompt_variant!r} for {protocol_name}"
        metric_contract = task_metric_contract(items)
        metric = metric_contract["metric"]
        output_kind = str(
            (spec.get("parser_ops") or {}).get("output_kind")
            or PROTOCOL_OUTPUT_KINDS.get(protocol_name)
            or ""
        )
        if metric != "unverified" and (
            not output_kind or not metric_accepts_output(metric, output_kind)
        ):
            return (
                {},
                {},
                f"protocol {protocol_name} output {output_kind or '<unknown>'!r} is incompatible "
                f"with metric {metric!r}",
            )
        input_mode = infer_task_input_mode(items)
        config = {
            "protocol": protocol_name,
            "prompt_variant": prompt_variant,
            "input": {"mode": input_mode},
            "metric": metric,
            "parser_output_kind": output_kind,
            "metric_config": metric_contract["config"],
            "formal_evaluation": metric_contract["formal_evaluation"],
            "protocol_config": dict(reuse.get("protocol_config") or {}),
        }
        route = {
            "task": task,
            "decision": "reuse",
            "active": True,
            "sample_count": len(items),
            "confidence": confidence,
            "protocol": protocol_name,
            "prompt_variant": prompt_variant,
            "metric": metric,
            "formal_evaluation": metric_contract["formal_evaluation"],
            "reason": reason or "agent selected a compatible spatial-evidence protocol",
        }
        return config, route, ""

    def _validate_v2_fallback(
        self,
        task: str,
        items: List[Dict[str, Any]],
        raw: Dict[str, Any],
        confidence: str,
        reason: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], List[str]]:
        fallback = raw.get("fallback") if isinstance(raw.get("fallback"), dict) else raw
        generation_prompt = ensure_choice_context(
            str(fallback.get("generation_prompt") or "").strip(),
            items,
        )
        parse_prompt = str(fallback.get("parse_prompt") or "").strip()
        visual_evidence = normalize_text_or_list(fallback.get("visual_evidence"))
        invalid_conditions = fallback.get("invalid_conditions") or []
        if isinstance(invalid_conditions, str):
            invalid_conditions = [invalid_conditions]
        invalid_conditions = [str(value).strip() for value in invalid_conditions if str(value).strip()]
        errors = []
        if not generation_prompt or "{question}" not in generation_prompt:
            errors.append("fallback generation_prompt must contain {question}")
        if re.search(r"\{\s*answer\s*\}|ground[- ]truth answer", generation_prompt, re.I):
            errors.append("fallback generation_prompt may not reference the ground-truth answer")
        generation_lower = generation_prompt.lower()
        if any(term in generation_lower for term in ("corner", "slot", "code strip")) and any(
            term in generation_lower for term in ("option", "answer", "label")
        ):
            errors.append("fallback may not use a pure answer-code layout")
        if uses_generic_verdict_symbol(generation_prompt, parse_prompt):
            errors.append("fallback may not use a generic verdict symbol as the yes/no answer")
        if not parse_prompt:
            errors.append("fallback parse_prompt is required")
        if re.search(r"\{\s*answer\s*\}|ground[- ]truth answer", parse_prompt, re.I):
            errors.append("fallback parse_prompt may not reference the ground-truth answer")
        if not visual_evidence or not any(
            term in visual_evidence.lower() for term in SPATIAL_EVIDENCE_TERMS
        ):
            errors.append("fallback must require concrete task-grounded spatial evidence")
        if len(invalid_conditions) < 2:
            errors.append("fallback requires at least two invalid_conditions")
        metric_contract = task_metric_contract(items)
        metric = metric_contract["metric"]
        if metric != "unverified" and not any(
            metric_accepts_output(metric, output_kind)
            for output_kind in VLM_SERIALIZABLE_OUTPUT_KINDS
        ):
            errors.append(
                f"VLM fallback cannot return an output compatible with metric {metric!r}"
            )
        if errors:
            return {}, {}, {}, errors

        input_mode = infer_task_input_mode(items)
        protocol_id = sanitize_protocol_id(f"generated_{self.benchmark_name}_{task}_vlm_fallback")
        schemas = sorted({infer_answer_schema(item) for item in items})
        answer_recovery = [
            {
                "answer_schema": schema,
                "visual_readout": "Recover the answer only from the required task-grounded spatial evidence.",
            }
            for schema in schemas
        ]
        protocol_config = {
            "generated_protocol_id": protocol_id,
            "include_source_images": False,
            "parse_prompt": parse_prompt,
            "answer_format": expected_answer_format(items[0]),
            "invalid_conditions": invalid_conditions,
            "visual_strategy": str(fallback.get("visual_strategy") or "other_spatial"),
            "visual_evidence": visual_evidence,
            "parser_observation": str(
                fallback.get("parser_observation")
                or "Read only the visible task-grounded marks or generated spatial state."
            ),
            "answer_recovery": answer_recovery,
            "label_role": "none",
            "rationale": reason,
            "metric": metric_contract["metric"],
            "metric_config": metric_contract["config"],
        }
        config = {
            "protocol": "agentic_vlm_protocol",
            "prompt_variant": "generated",
            "prompt": generation_prompt,
            "input": {"mode": input_mode},
            "metric": metric_contract["metric"],
            "metric_config": metric_contract["config"],
            "formal_evaluation": metric_contract["formal_evaluation"],
            "protocol_config": protocol_config,
        }
        protocol = {
            "id": protocol_id,
            "task": task,
            "decision": "fallback",
            "generation_prompt": generation_prompt,
            "parse_prompt": parse_prompt,
            "parser_inputs": ["generated_response"],
            "visual_evidence": visual_evidence,
            "invalid_conditions": invalid_conditions,
            "metric_contract": metric_contract,
            "rationale": reason,
        }
        route = {
            "task": task,
            "decision": "fallback",
            "active": True,
            "sample_count": len(items),
            "confidence": confidence,
            "protocol_id": protocol_id,
            "protocol": "agentic_vlm_protocol",
            "parser_backend": "vlm_fallback",
            "parser_inputs": ["generated_response"],
            "metric": metric_contract["metric"],
            "formal_evaluation": metric_contract["formal_evaluation"],
            "reason": reason,
        }
        return config, protocol, route, []

def load_protocol_catalog(path: str | Path) -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    root = Path(path)
    if not root.exists():
        return catalog
    for spec_path in sorted(root.glob("*.yaml")):
        payload = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
        name = str(payload.get("name") or spec_path.stem)
        catalog[name] = payload
    return catalog


def protocol_inventory(catalog: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    inventory = []
    for name, spec in sorted(catalog.items()):
        if name in NON_SPATIAL_REUSE_PROTOCOLS or spec.get("catalog_role") == "runtime_adapter":
            continue
        inventory.append(
            {
                "protocol": name,
                "description": str(spec.get("description") or "").strip(),
                "prompt_variants": sorted((spec.get("prompts") or {}).keys()),
                "default_config": spec.get("default_config") or {},
                "visual_contract": spec.get("visual_contract") or {},
                "parser_ops": spec.get("parser_ops") or {},
            }
        )
    return inventory


def select_representative_items(items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    selected: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        fingerprint = item_fingerprint(item)
        if fingerprint in seen:
            continue
        selected.append(item)
        seen.add(fingerprint)
        if len(selected) >= limit:
            return selected
    selected_ids = {id(item) for item in selected}
    for item in items:
        if id(item) in selected_ids:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def item_fingerprint(item: Dict[str, Any]) -> tuple[Any, ...]:
    media = input_media_entries(item)
    return (
        str(item.get("answer_type") or ""),
        len(item.get("choices") or item.get("options") or []),
        str((item.get("evaluation") or {}).get("metric") or item.get("metric") or ""),
        str((item.get("input") or {}).get("type") or ""),
        len(media),
        tuple(sorted(str(entry.get("role") or "") for entry in media)),
        infer_answer_schema(item),
        answer_diversity_bucket(item),
    )


def answer_diversity_bucket(item: Dict[str, Any]) -> str:
    schema = infer_answer_schema(item)
    if schema not in {"binary_boolean", "choice_selection"}:
        return ""
    value = str(item.get("answer") or "").strip().lower()
    if schema == "binary_boolean":
        if value in {"yes", "true", "1"}:
            return "yes"
        if value in {"no", "false", "0"}:
            return "no"
    return value


def summarize_item(item: Dict[str, Any], attachment_indexes: List[int]) -> Dict[str, Any]:
    media = input_media_entries(item)
    return {
        "id": item.get("id", ""),
        "question": item.get("question", ""),
        "answer_type": item.get("answer_type", ""),
        "answer_format": expected_answer_format(item),
        "answer_schema": infer_answer_schema(item),
        "choices": item.get("choices") or item.get("options") or [],
        "evaluation": item.get("evaluation") or {},
        "input_type": (item.get("input") or {}).get("type", ""),
        "media_count": len(media) or (1 if primary_media_path(item) else 0),
        "media_roles": sorted({str(entry.get("role") or "") for entry in media}),
        "media_types": sorted({str(entry.get("type") or "image") for entry in media}),
        "attached_image_indexes": attachment_indexes,
        "metadata_keys": sorted((item.get("metadata") or {}).keys())[:30],
    }


def group_items_by_task(items: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item.get("task") or "default")].append(item)
    return dict(groups)


def task_metrics(items: List[Dict[str, Any]]) -> set[str]:
    return {
        str((item.get("evaluation") or {}).get("metric") or item.get("metric") or "accuracy").strip()
        for item in items
    }


def summarize_task_contract(task: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_contract = task_metric_contract(items)
    return {
        "task": task,
        "sample_count": len(items),
        "answer_schemas": sorted({infer_answer_schema(item) for item in items}),
        "answer_types": sorted({str(item.get("answer_type") or "") for item in items}),
        "input_mode": infer_task_input_mode(items),
        "media_counts": sorted({len(input_media_entries(item)) for item in items}),
        "metric": metric_contract["metric"],
        "metric_config": metric_contract["config"],
        "formal_evaluation": metric_contract["formal_evaluation"],
    }


def unsupported_media_kinds(items: List[Dict[str, Any]]) -> List[str]:
    unsupported = set()
    for item in items:
        for entry in input_media_entries(item):
            media_type = str(entry.get("type") or "image").strip().lower()
            suffix = Path(str(entry.get("path") or "")).suffix.lower()
            if media_type not in {"", "image"}:
                unsupported.add(media_type)
            elif suffix and suffix not in IMAGE_SUFFIXES:
                unsupported.add(suffix.lstrip(".") or "unknown")
    return sorted(unsupported)


def infer_input_mode(item: Dict[str, Any]) -> str:
    if len(input_media_entries(item)) > 1 or item.get("metadata", {}).get("images"):
        return "metadata_images"
    if item.get("metadata", {}).get("file_names"):
        return "file_names_same_dir"
    return "single"


def infer_task_input_mode(items: List[Dict[str, Any]]) -> str:
    modes = {infer_input_mode(item) for item in items}
    if "metadata_images" in modes:
        return "metadata_images"
    if "file_names_same_dir" in modes:
        return "file_names_same_dir"
    return "single"


def infer_decision(row: Dict[str, Any]) -> str:
    decision = str(row.get("decision") or "").strip().lower()
    if decision:
        return decision
    if row.get("fallback"):
        return "fallback"
    if row.get("reuse") or row.get("protocol"):
        return "reuse"
    return "unsupported" if row.get("unsupported") else ""


def normalize_confidence(value: Any) -> str:
    value = str(value or "medium").strip().lower()
    return value if value in {"high", "medium", "low"} else "medium"


def refresh_build_manifest_counts(result: AgenticProtocolBuildResult) -> None:
    tasks = result.benchmark_config.get("tasks") or {}
    rows = result.manifest.get("route_rows") or []
    result.manifest["active_task_count"] = len(tasks)
    result.manifest["active_sample_count"] = sum(
        int(row.get("sample_count") or 0) for row in rows if row.get("active")
    )
    result.manifest["decision_counts"] = dict(
        sorted(Counter(str(row.get("decision") or "unknown") for row in rows).items())
    )


def unsupported_route(
    task: str,
    items: List[Dict[str, Any]],
    reason: str,
    source: str,
) -> Dict[str, Any]:
    return {
        "task": task,
        "decision": "unsupported",
        "active": False,
        "sample_count": len(items),
        "reason": reason,
        "source": source,
    }


def sanitize_revision_diagnostics(value: Any) -> Any:
    """Remove correctness and ground-truth signals before protocol repair."""
    blocked = {
        "answer",
        "correct",
        "correct_count",
        "correct_among_valid",
        "ground_truth",
        "gt",
        "is_correct",
        "mean_score",
        "score",
        "scores",
    }
    if isinstance(value, dict):
        return {
            str(key): sanitize_revision_diagnostics(item)
            for key, item in value.items()
            if str(key).strip().lower() not in blocked
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_revision_diagnostics(item) for item in value]
    if isinstance(value, str):
        return value[:2000]
    return value


def extract_task_row(payload: Dict[str, Any], task: str) -> Dict[str, Any] | None:
    rows = payload.get("tasks") if isinstance(payload, dict) else None
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and str(row.get("task") or "").strip() == task:
                return row
        if len(rows) == 1 and isinstance(rows[0], dict):
            row = dict(rows[0])
            row.setdefault("task", task)
            return row
    if isinstance(payload, dict) and (payload.get("decision") or payload.get("reuse")):
        row = dict(payload)
        row.setdefault("task", task)
        return row
    return None


def normalize_agent_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload.get("tasks"), list):
        return payload
    if isinstance(payload, dict) and (
        payload.get("decision")
        or payload.get("reuse")
        or payload.get("visual_contract")
        or payload.get("fallback")
    ):
        return {
            "benchmark": payload.get("benchmark", ""),
            "tasks": [payload],
        }
    raw_by_task = payload.get("task_responses")
    if not isinstance(raw_by_task, dict):
        return payload
    rows = []
    for task, response in raw_by_task.items():
        try:
            parsed = response if isinstance(response, dict) else parse_agent_json_response(str(response or ""))
        except ValueError:
            continue
        row = extract_task_row(parsed, str(task))
        if row is not None:
            rows.append(row)
    return {"benchmark": payload.get("benchmark", ""), "tasks": rows}


def normalize_text_or_list(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def concrete_spatial_evidence_terms(value: Any) -> List[str]:
    text = normalize_text_or_list(value).lower()
    return sorted(term for term in SPATIAL_EVIDENCE_TERMS if term in text)


def uses_generic_verdict_symbol(generation_prompt: str, parse_prompt: str) -> bool:
    text = " ".join((str(generation_prompt or ""), str(parse_prompt or ""))).lower()
    symbol = (
        r"(?:check\s*mark|checkmark|tick\s*mark|thumbs?[- ]?up|thumbs?[- ]?down|"
        r"traffic[- ]light|(?:green|red|orange)\s+(?:check|x|cross))"
    )
    action = re.compile(
        r"\b(?:add|draw|show|place|use|identify|inspect)\b|"
        r"\b(?:presence\s+of|look\s+for)\b"
    )
    symbol_pattern = re.compile(symbol)
    negated_action = re.compile(
        r"(?:do\s+not|don't|never|must\s+not|should\s+not|avoid)\s+"
        r"(?:\w+\s+){0,2}$"
    )
    for symbol_match in symbol_pattern.finditer(text):
        window_start = max(0, symbol_match.start() - 100)
        actions = list(action.finditer(text, window_start, symbol_match.start()))
        if not actions:
            continue
        nearest_action = actions[-1]
        prefix = text[max(0, nearest_action.start() - 48) : nearest_action.start()]
        if negated_action.search(prefix):
            continue
        return True
    return False


def sanitize_protocol_id(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "generated_protocol").strip())
    return text.strip("_") or "generated_protocol"


def parse_agent_json_response(response: str) -> Dict[str, Any]:
    text = str(response or "").strip()
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE))
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("no JSON object found")


def join_task_artifacts(values: Dict[str, str]) -> str:
    return "\n\n".join(f"===== TASK: {task} =====\n{text}" for task, text in sorted(values.items()))


def write_build_outputs(
    result: AgenticProtocolBuildResult,
    *,
    benchmark_config_path: str | Path,
    protocol_path: str | Path,
    manifest_path: str | Path,
    prompt_path: str | Path,
    raw_response_path: str | Path,
) -> Dict[str, str]:
    outputs = {
        "benchmark_config": Path(benchmark_config_path),
        "agentic_protocols": Path(protocol_path),
        "manifest": Path(manifest_path),
        "prompt": Path(prompt_path),
        "raw_response": Path(raw_response_path),
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    config_text = yaml.safe_dump(
        result.benchmark_config, sort_keys=False, allow_unicode=True
    )
    outputs["benchmark_config"].write_text(config_text, encoding="utf-8")
    outputs["agentic_protocols"].write_text(
        yaml.safe_dump(result.generated_protocols, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    result.manifest["protocol_artifact"] = {
        "schema_version": "provise.protocol.v1",
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
    }
    outputs["manifest"].write_text(
        json.dumps(result.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    outputs["prompt"].write_text(result.prompt, encoding="utf-8")
    outputs["raw_response"].write_text(result.raw_response, encoding="utf-8")
    return {name: str(path) for name, path in outputs.items()}


def namespace_from_args(args: argparse.Namespace) -> str:
    return str(getattr(args, "benchmark_name", "") or Path(str(args.input)).stem)


def load_items(path: str | Path) -> List[Dict[str, Any]]:
    return load_unified_items(path)
