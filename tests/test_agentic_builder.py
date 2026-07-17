import argparse
import json
from pathlib import Path

from PIL import Image

from provise.protocol_agent.builder import (
    AgenticProtocolBuilder,
    infer_answer_schema,
    infer_task_input_mode,
    select_representative_items,
)
from provise.protocol_agent.visual_contract import compile_contract
from provise.protocols import create_protocol, list_protocols
from provise.evaluation.runner import MockProtocolModel, load_protocol_pool, run_task
import provise.protocol_agent.pipeline as build_script
from provise.protocol_agent.pipeline import (
    activate_compile_fallbacks,
    apply_smoke_gate,
    build_tasks_sequentially,
    parser_agreement,
    readout_operational_rate,
    run_automatic_fallback_smoke,
    run_smoke_validation,
    metric_compatibility_rate_among_valid,
    seed_fallback_smoke_images,
    smoke_failure_reasons,
    spatial_evidence_rate,
    smoke_failure_is_external,
)


def _toy_item(image_name="sample.png"):
    return {
        "schema_version": "genbench.v1",
        "id": "sample_1",
        "benchmark": "toy",
        "task": "new_spatial_task",
        "image_path": image_name,
        "question": "Which option best matches the marked spatial relation?",
        "answer": "A",
        "answer_type": "choice",
        "choices": [{"label": "A", "text": "left"}, {"label": "B", "text": "right"}],
        "evaluation": {"metric": "accuracy"},
    }


def _agent_response():
    return json.dumps(
        {
            "benchmark": "toy",
            "tasks": [
                {
                    "task": "new_spatial_task",
                    "decision": "fallback",
                    "confidence": "high",
                    "reason": "A relation arrow exposes the selected spatial relation directly.",
                    "fallback": {
                        "visual_strategy": "object_marking",
                        "visual_evidence": "blue outlines around the referenced objects and one relation arrow between them",
                        "generation_prompt": "Given {question} and {choices}, outline the referenced objects in blue and draw one large arrow between them that depicts the selected spatial relation. An option label may be attached to the arrow only as a secondary cue.",
                        "parse_prompt": "Inspect the blue object outlines and the relation arrow, then recover the option expressed by that spatial evidence.",
                        "parser_observation": "the direction of the arrow connecting the two outlined objects",
                        "invalid_conditions": ["the referenced objects are not outlined", "the relation arrow is absent or ambiguous"],
                    },
                }
            ],
        }
    )


def test_agentic_vlm_protocol_is_registered():
    assert "agentic_vlm_protocol" in list_protocols()
    protocol = create_protocol(
        "agentic_vlm_protocol",
        {
            "parse_prompt": "Return the selected option.",
            "mock_parse_response": '{"status":"valid","prediction":"A","evidence":"mock","confidence":"high"}',
        },
    )
    parsed = protocol.parse("unused.png", _toy_item(), ".")
    assert parsed.parse_success
    assert parsed.prediction == "A"


def test_agentic_live_parse_uses_agentic_response_fields(tmp_path: Path):
    source = tmp_path / "sample.png"
    generated = tmp_path / "generated.png"
    Image.new("RGB", (32, 32), "white").save(source)
    Image.new("RGB", (32, 32), "blue").save(generated)

    class FakeVLM:
        image_paths = []

        def predict_multi(self, image_paths, prompt):
            self.image_paths = list(image_paths)
            return '{"status":"valid","prediction":"A","evidence":"blue outline around the target object","confidence":"high"}'

    protocol = create_protocol(
        "agentic_vlm_protocol",
        {
            "parse_prompt": "Inspect the blue object outline.",
            "visual_strategy": "object_marking",
            "visual_evidence": "blue outline around the target object",
            "parser_observation": "the outlined object",
        },
    )
    fake_vlm = FakeVLM()
    protocol.__class__._eval_vlm = fake_vlm
    try:
        parsed = protocol.parse(str(generated), _toy_item("sample.png"), str(tmp_path))
    finally:
        protocol.__class__._eval_vlm = None

    assert parsed.parse_success
    assert fake_vlm.image_paths == [str(generated)]
    assert parsed.extra["agentic_evidence"].startswith("blue outline")
    assert "fallback_evidence" not in parsed.extra


def test_agentic_builder_generates_benchmark_config():
    result = AgenticProtocolBuilder(
        [_toy_item()],
        benchmark_name="toy",
        data_file="toy.jsonl",
        benchmark_root="assets",
    ).build(raw_response=_agent_response())

    task_cfg = result.benchmark_config["tasks"]["new_spatial_task"]
    assert task_cfg["protocol"] == "agentic_vlm_protocol"
    assert task_cfg["prompt_variant"] == "generated"
    assert task_cfg["protocol_config"]["generated_protocol_id"].endswith("_vlm_fallback")
    assert result.generated_protocols["protocols"][0]["parse_prompt"].startswith("Inspect the blue")


def test_agentic_builder_rejects_fallback_when_metric_requires_a_mask():
    item = _toy_item()
    item["evaluation"] = {"metric": "mask_precision"}
    response = json.loads(_agent_response())
    response["tasks"][0]["metric"] = "accuracy"

    result = AgenticProtocolBuilder(
        [item],
        benchmark_name="toy",
        data_file="toy.jsonl",
        benchmark_root="assets",
    ).build(raw_response=json.dumps(response))

    assert result.benchmark_config["tasks"] == {}
    assert "mask_precision" in result.manifest["route_rows"][0]["reason"]


def test_agentic_builder_can_resume_from_saved_task_response_artifact():
    saved = json.dumps(
        {
            "benchmark": "toy",
            "task_responses": {"new_spatial_task": _agent_response()},
        }
    )

    result = AgenticProtocolBuilder(
        [_toy_item()], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    ).build(raw_response=saved)

    assert result.benchmark_config["tasks"]["new_spatial_task"]["protocol"] == "agentic_vlm_protocol"


def test_agentic_builder_disables_task_on_malformed_response():
    result = AgenticProtocolBuilder(
        [_toy_item()],
        benchmark_name="toy",
        data_file="toy.jsonl",
        benchmark_root="assets",
    ).build(raw_response="not json")

    assert result.benchmark_config["tasks"] == {}
    assert result.manifest["route_rows"][0]["decision"] == "unsupported"
    assert result.manifest["warnings"]


def test_agentic_builder_reuses_existing_spatial_protocol():
    item = _toy_item()
    item["answer_type"] = "mask"
    item["evaluation"] = {"metric": "mask_precision", "mask_path": "target.png"}
    response = {
        "benchmark": "toy",
        "tasks": [
            {
                "task": "new_spatial_task",
                "decision": "reuse",
                "confidence": "high",
                "reason": "The task asks for a target region mask.",
                "reuse": {
                    "protocol": "region_mask",
                    "prompt_variant": "binary_target_mask",
                    "protocol_config": {"success_precision": 0.6},
                },
            }
        ],
    }
    result = AgenticProtocolBuilder(
        [item], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    ).build(raw_response=json.dumps(response))

    task_cfg = result.benchmark_config["tasks"]["new_spatial_task"]
    assert task_cfg["protocol"] == "region_mask"
    assert task_cfg["protocol_config"]["success_precision"] == 0.6
    assert result.manifest["route_rows"][0]["decision"] == "reuse"


def test_agentic_builder_compiles_marker_count_recipe():
    item = _toy_item()
    item.update(
        question="How many cups are closer than the plate?",
        answer=2,
        answer_type="number",
        choices=[],
        evaluation={"metric": "exact_match"},
    )
    response = {
        "benchmark": "toy",
        "tasks": [
            {
                "task": "new_spatial_task",
                "decision": "build",
                "build_mode": "recipe",
                "confidence": "medium",
                "reason": "Visible markers expose the conditionally selected set.",
                "visual_contract": {
                    "recipe": "instance_marker_count",
                    "mode": "edit_source",
                    "primitives": ["one marker per qualifying object"],
                    "parameters": {},
                },
                "readout": {"recipe": "instance_marker_count"},
            }
        ],
    }

    result = AgenticProtocolBuilder(
        [item], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    ).build(raw_response=json.dumps(response))

    task_cfg = result.benchmark_config["tasks"]["new_spatial_task"]
    route = result.manifest["route_rows"][0]
    assert task_cfg["protocol"] == "instance_marker_count"
    assert route["decision"] == "build"
    assert route["build_mode"] == "recipe"
    assert route["recipe"] == "instance_marker_count"


def test_occluded_count_recipe_requires_explicit_amodal_scope():
    item = _toy_item()
    item.update(
        question=(
            "Count all cups as if the black box were not there. Assume the "
            "pattern continues behind the black box."
        ),
        answer=12,
        answer_type="integer_count",
        choices=[],
        evaluation={"metric": "smape"},
    )
    raw = {
        "decision": "build",
        "build_mode": "recipe",
        "visual_contract": {
            "recipe": "instance_marker_count",
            "primitives": ["one marker per target"],
            "parameters": {},
        },
        "readout": {"recipe": "instance_marker_count"},
    }

    result = compile_contract(
        task="new_spatial_task",
        items=[item],
        raw=raw,
        input_mode="single",
        confidence="high",
        reason="fixture",
    )

    assert result.errors == [
        "instance_marker_count.target_scope must be "
        "visible_and_occlusion_inferred for an occluded-total task"
    ]


def test_occluded_count_recipe_uses_amodal_prompt_variant():
    item = _toy_item()
    item.update(
        question="Count all cups as if the black box were not there.",
        answer=12,
        answer_type="integer_count",
        choices=[],
        evaluation={"metric": "smape"},
    )
    raw = {
        "decision": "build",
        "build_mode": "recipe",
        "visual_contract": {
            "recipe": "instance_marker_count",
            "primitives": ["visible and inferred hidden markers"],
            "parameters": {"target_scope": "visible_and_occlusion_inferred"},
        },
        "readout": {"recipe": "instance_marker_count"},
    }

    result = compile_contract(
        task="new_spatial_task",
        items=[item],
        raw=raw,
        input_mode="single",
        confidence="high",
        reason="fixture",
    )

    assert not result.errors
    assert result.task_config["prompt_variant"] == "green_star_amodal_count"
    assert (
        result.task_config["protocol_config"]["target_scope"]
        == "visible_and_occlusion_inferred"
    )


def test_agentic_builder_compiles_relation_zone_boolean_recipe():
    item = _toy_item()
    item.update(
        question="Is the cup left of the plate?",
        answer="yes",
        answer_type="boolean",
        choices=[{"label": "no", "text": "no"}, {"label": "yes", "text": "yes"}],
        evaluation={"metric": "accuracy"},
    )
    response = {
        "task": "new_spatial_task",
        "decision": "build",
        "build_mode": "recipe",
        "confidence": "high",
        "reason": "A visible valid region preserves the binary relation geometrically.",
        "visual_contract": {
            "recipe": "relation_zone_boolean",
            "mode": "edit_source",
            "primitives": ["subject anchor", "reference anchor", "valid relation region"],
            "parameters": {},
        },
        "readout": {"recipe": "relation_zone_boolean"},
    }

    result = AgenticProtocolBuilder(
        [item], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    ).build(raw_response=json.dumps(response))

    config = result.benchmark_config["tasks"]["new_spatial_task"]
    route = result.manifest["route_rows"][0]
    assert config["protocol"] == "agentic_parser_ops_protocol"
    assert config["protocol_config"]["parser_output_kind"] == "boolean"
    assert config["protocol_config"]["parser_pipeline"]["output"] == "relation_holds"
    assert "Do not add a check mark" in config["prompt"]
    assert route["recipe"] == "relation_zone_boolean"


def test_agentic_builder_prompt_explains_point_in_mask_cardinality():
    item = _toy_item()
    item["answer"] = "[(0.2, 0.3), (0.4, 0.5)]"
    item["answer_type"] = "points"
    item["choices"] = []
    item["evaluation"] = {
        "metric": "point_in_mask",
        "mask_path": "target.png",
        "num_points_to_match": 2,
    }
    response = json.dumps(
        {
            "benchmark": "toy",
            "tasks": [
                {
                    "task": "new_spatial_task",
                    "decision": "unsupported",
                    "confidence": "high",
                    "reason": "test response",
                }
            ],
        }
    )

    result = AgenticProtocolBuilder(
        [item], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    ).build(raw_response=response)

    assert "one recovered point preserves the metric" in result.prompt
    assert "reusable deterministic point-marker" in result.prompt


def test_agentic_builder_rejects_answer_code_fallback():
    response = json.loads(_agent_response())
    fallback = response["tasks"][0]["fallback"]
    fallback["generation_prompt"] = (
        "For {question}, put the option label in a corner answer slot and use no other marks."
    )

    result = AgenticProtocolBuilder(
        [_toy_item()], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    ).build(raw_response=json.dumps(response))

    assert result.benchmark_config["tasks"] == {}
    assert "answer-code" in result.manifest["route_rows"][0]["reason"]


def test_agentic_builder_rejects_generic_verdict_symbol_fallback():
    response = json.loads(_agent_response())
    fallback = response["tasks"][0]["fallback"]
    fallback["generation_prompt"] = (
        "For {question}, outline the two objects, then add a green check mark for yes "
        "or an orange X for no."
    )
    fallback["parse_prompt"] = "Inspect the outlines and identify the green check or orange X."

    result = AgenticProtocolBuilder(
        [_toy_item()], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    ).build(raw_response=json.dumps(response))

    assert result.benchmark_config["tasks"] == {}
    assert "verdict symbol" in result.manifest["route_rows"][0]["reason"]


def test_agentic_builder_allows_prohibiting_generic_verdict_symbols():
    response = json.loads(_agent_response())
    fallback = response["tasks"][0]["fallback"]
    fallback["generation_prompt"] = (
        "For {question}, highlight both compared objects and add one vertical measurement bar "
        "to each object. Do not add answer letters, yes/no text, checkmarks, or crosses."
    )
    fallback["parse_prompt"] = (
        "Read only the two highlighted objects and their measurement bars, compare the bar "
        "lengths, and map that visible evidence to a choice."
    )
    fallback["visual_evidence"] = (
        "two highlighted objects with one visible measurement bar attached to each object"
    )
    fallback["invalid_conditions"] = [
        "either compared object is not highlighted",
        "either measurement bar is missing or ambiguous",
    ]

    result = AgenticProtocolBuilder(
        [_toy_item()], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    ).build(raw_response=json.dumps(response))

    config = result.benchmark_config["tasks"]["new_spatial_task"]
    assert config["protocol"] == "agentic_vlm_protocol"
    assert "measurement bar" in config["prompt"]


def test_agentic_builder_fallback_covers_each_observed_answer_schema():
    choice_item = _toy_item()
    boolean_item = _toy_item("sample_2.png")
    boolean_item["id"] = "sample_2"
    boolean_item["question"] = "Is the left object larger than the right object?"
    boolean_item["choices"] = [
        {"label": "A", "text": "no"},
        {"label": "B", "text": "yes"},
    ]

    result = AgenticProtocolBuilder(
        [choice_item, boolean_item],
        benchmark_name="toy",
        data_file="toy.jsonl",
        benchmark_root="assets",
    ).build(raw_response=_agent_response())

    recovery = result.benchmark_config["tasks"]["new_spatial_task"]["protocol_config"][
        "answer_recovery"
    ]
    assert {row["answer_schema"] for row in recovery} == {
        "binary_boolean",
        "choice_selection",
    }


def test_agentic_builder_disables_unsupported_video_input():
    item = _toy_item()
    item["input"] = {
        "type": "video",
        "media": [{"type": "video", "path": "clip.mp4", "role": "primary"}],
    }

    result = AgenticProtocolBuilder(
        [item], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    ).build(raw_response=_agent_response())

    assert result.benchmark_config["tasks"] == {}
    assert "video" in result.manifest["route_rows"][0]["reason"]
    assert activate_compile_fallbacks(
        AgenticProtocolBuilder(
            [item], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
        ),
        result,
    ) == []
    assert result.benchmark_config["tasks"] == {}


def test_answer_schema_recognizes_chinese_boolean_choices():
    item = _toy_item()
    item["choices"] = [{"label": "A", "text": "否"}, {"label": "B", "text": "是"}]

    assert infer_answer_schema(item) == "binary_boolean"


def test_agentic_builder_sends_representative_image_without_ground_truth(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    second_image_path = tmp_path / "sample_2.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    Image.new("RGB", (32, 32), "black").save(second_image_path)
    item = _toy_item("sample.png")
    item["input"] = {
        "type": "multi_image",
        "media": [
            {"type": "image", "path": "sample.png", "role": "start"},
            {"type": "image", "path": "sample_2.png", "role": "candidate"},
        ],
    }

    class RecordingVLM:
        def __init__(self):
            self.calls = []

        def predict_multi(self, image_paths, prompt):
            self.calls.append((image_paths, prompt))
            return _agent_response()

    vlm = RecordingVLM()
    result = AgenticProtocolBuilder(
        [item],
        benchmark_name="toy",
        data_file="toy.jsonl",
        benchmark_root=str(tmp_path),
    ).build(vlm=vlm)

    assert vlm.calls[0][0] == [str(image_path), str(second_image_path)]
    assert '"answer": "A"' not in vlm.calls[0][1]
    assert "representative_examples_without_ground_truth" in vlm.calls[0][1]
    assert result.benchmark_config["tasks"]["new_spatial_task"]["protocol"] == "agentic_vlm_protocol"


def test_task_input_mode_preserves_mixed_single_and_multi_image_samples():
    single = _toy_item()
    multi = _toy_item("first.png")
    multi["id"] = "sample_2"
    multi["input"] = {
        "type": "multi_image",
        "media": [
            {"type": "image", "path": "first.png", "role": "view"},
            {"type": "image", "path": "second.png", "role": "view"},
        ],
    }
    response = json.loads(_agent_response())

    result = AgenticProtocolBuilder(
        [single, multi],
        benchmark_name="toy",
        data_file="toy.jsonl",
        benchmark_root="assets",
    ).build(raw_response=json.dumps(response))

    assert infer_task_input_mode([single, multi]) == "metadata_images"
    assert '"inferred_input_mode": "metadata_images"' in result.prompt
    assert result.benchmark_config["tasks"]["new_spatial_task"]["input"]["mode"] == "metadata_images"


def test_representative_selection_covers_binary_answer_outcomes_without_prompt_leakage():
    yes_item = _toy_item()
    yes_item["answer"] = "yes"
    yes_item["choices"] = [
        {"label": "yes", "text": "yes"},
        {"label": "no", "text": "no"},
    ]
    no_item = json.loads(json.dumps(yes_item))
    no_item["id"] = "sample_2"
    no_item["answer"] = "no"

    selected = select_representative_items([yes_item, no_item], 2)
    result = AgenticProtocolBuilder(
        [yes_item, no_item],
        benchmark_name="toy",
        data_file="toy.jsonl",
        benchmark_root="assets",
    ).build(raw_response='{"decision":"unsupported","reason":"not suitable"}')

    assert {item["answer"] for item in selected} == {"yes", "no"}
    assert '"answer": "yes"' not in result.prompt
    assert '"answer": "no"' not in result.prompt


def test_smoke_gate_disables_failed_task_without_generic_fallback():
    benchmark_cfg = {"tasks": {"new_spatial_task": {"protocol": "agentic_vlm_protocol"}}}
    manifest = {
        "route_rows": [
            {
                "task": "new_spatial_task",
                "decision": "fallback",
                "active": True,
                "sample_count": 3,
            }
        ]
    }
    smoke = {
        "tasks": {
            "new_spatial_task": {
                "status": "failed",
                "failed_checks": ["parser_agreement_rate"],
            }
        }
    }

    apply_smoke_gate(benchmark_cfg, manifest, smoke)

    assert benchmark_cfg["tasks"] == {}
    assert manifest["route_rows"][0]["decision"] == "unsupported"
    assert "generic_vlm_fallback" not in json.dumps(benchmark_cfg)


def test_agent_unsupported_image_task_activates_auditable_vlm_fallback():
    item = _toy_item()
    builder = AgenticProtocolBuilder(
        [item], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    )
    result = builder.build(
        raw_response=json.dumps(
            {
                "task": "new_spatial_task",
                "decision": "unsupported",
                "confidence": "medium",
                "reason": "No deterministic parser recipe fits.",
            }
        )
    )

    activated = activate_compile_fallbacks(builder, result)

    config = result.benchmark_config["tasks"]["new_spatial_task"]
    route = result.manifest["route_rows"][0]
    assert activated == ["new_spatial_task"]
    assert config["protocol"] == "agentic_vlm_protocol"
    assert config["formal_evaluation"] is True
    assert "{question}" in config["prompt"]
    assert "{answer}" not in config["prompt"]
    assert route["source"] == "automatic_vlm_fallback"
    assert route["fallback_origin"] == "compile:agent"
    assert route["parser_backend"] == "vlm_fallback"


def test_failed_deterministic_smoke_runs_vlm_fallback_smoke(monkeypatch, tmp_path: Path):
    item = _toy_item()
    builder = AgenticProtocolBuilder(
        [item], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    )
    result = builder.build(
        raw_response=json.dumps(
            {
                "task": "new_spatial_task",
                "decision": "unsupported",
                "confidence": "medium",
                "reason": "fixture",
            }
        )
    )
    result.benchmark_config["tasks"] = {
        "new_spatial_task": {"protocol": "agentic_parser_ops_protocol"}
    }
    result.manifest["route_rows"][0].update(
        {
            "active": True,
            "decision": "build",
            "build_mode": "recipe",
            "source": "contract_compiler",
        }
    )
    smoke = {
        "tasks": {
            "new_spatial_task": {
                "status": "failed",
                "failed_checks": ["parse_success_rate"],
                "failure_category_counts": {"parser_failure": 1},
            }
        }
    }
    calls = []

    def fake_smoke(_args, config, *, reporter=None):
        calls.append(config)
        assert config["tasks"]["new_spatial_task"]["protocol"] == "agentic_vlm_protocol"
        return {
            "tasks": {
                "new_spatial_task": {
                    "status": "passed",
                    "valid_parse_rate": 100.0,
                }
            }
        }

    monkeypatch.setattr(build_script, "run_smoke_validation", fake_smoke)
    args = argparse.Namespace(
        max_revisions=1,
        smoke_output=str(tmp_path / "smoke"),
        benchmark_name="toy",
        smoke_model="mock-image",
    )

    activated = run_automatic_fallback_smoke(args, builder, result, smoke)

    assert activated == ["new_spatial_task"]
    assert len(calls) == 1
    assert smoke["tasks"]["new_spatial_task"]["status"] == "passed"
    assert smoke["fallback_runs"]
    assert result.manifest["fallback_history"][-1]["origin"].startswith("smoke:")


def test_vlm_readout_fallback_preserves_deterministic_generation_protocol(tmp_path: Path):
    item = _toy_item()
    builder = AgenticProtocolBuilder(
        [item], benchmark_name="toy", data_file="toy.jsonl", benchmark_root="assets"
    )
    result = builder.build(
        raw_response=json.dumps(
            {
                "task": "new_spatial_task",
                "decision": "unsupported",
                "confidence": "medium",
                "reason": "fixture",
            }
        )
    )
    prompt = (
        "Answer {question}. Choices: {choices}. Draw one thick magenta object outline "
        "around the selected object."
    )
    result.benchmark_config["tasks"] = {
        "new_spatial_task": {
            "protocol": "agentic_parser_ops_protocol",
            "prompt": prompt,
        }
    }
    result.generated_protocols["protocols"] = [
        {
            "task": "new_spatial_task",
            "decision": "build",
            "build_mode": "recipe",
            "generation_prompt": prompt,
            "visual_contract": {"primitives": ["subject anchor", "reference anchor"]},
        }
    ]

    activated, errors = builder.activate_automatic_vlm_fallback(
        result,
        task="new_spatial_task",
        origin="smoke:deterministic_protocol_failed",
        reason="CLIP readout was ambiguous",
    )

    config = result.benchmark_config["tasks"]["new_spatial_task"]
    assert activated
    assert errors == []
    assert config["prompt"] == prompt
    assert config["protocol"] == "agentic_vlm_protocol"
    assert config["protocol_config"]["fallback_preserved_generation"] is True
    assert "outline" in config["protocol_config"]["visual_evidence"]


def test_fallback_smoke_seeds_only_preserved_generation_images(tmp_path: Path):
    source_root = tmp_path / "smoke"
    target_root = tmp_path / "smoke_vlm_fallback"
    source_dir = source_root / "task_a"
    source_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), "magenta").save(source_dir / "sample_generated.png")
    config = {
        "tasks": {
            "task_a": {
                "protocol_config": {"fallback_preserved_generation": True},
            },
            "task_b": {
                "protocol_config": {"fallback_preserved_generation": False},
            },
        }
    }

    copied = seed_fallback_smoke_images(
        source_root, target_root, ["task_a", "task_b"], config
    )

    assert copied == 1
    assert (target_root / "task_a" / "sample_generated.png").is_file()


def test_smoke_gate_defers_pure_generation_api_failure():
    benchmark_cfg = {"tasks": {"spatial_task": {"protocol": "agentic_parser_ops_protocol"}}}
    manifest = {
        "route_rows": [
            {
                "task": "spatial_task",
                "decision": "build",
                "build_mode": "recipe",
                "active": True,
                "sample_count": 3,
            }
        ]
    }
    smoke = {
        "tasks": {
            "spatial_task": {
                "status": "failed",
                "failed_checks": ["generation_rate", "parse_success_rate"],
                "failure_category_counts": {"generation_failure": 3},
            }
        }
    }

    assert smoke_failure_is_external(smoke["tasks"]["spatial_task"])
    apply_smoke_gate(benchmark_cfg, manifest, smoke)

    assert benchmark_cfg["tasks"] == {}
    route = manifest["route_rows"][0]
    assert route["decision"] == "deferred"
    assert route["source"] == "smoke_external_failure"
    assert manifest["deferred_external_failure_tasks"] == ["spatial_task"]
    assert manifest["disabled_tasks"] == []


def test_parser_agreement_and_spatial_evidence_metrics():
    first = {
        "detailed_results": [
            {
                "id": "one",
                "parse_success": True,
                "prediction": "A",
                "agentic_evidence": "blue outline around the left object",
                "score_computed": True,
            },
            {
                "id": "two",
                "parse_success": True,
                "prediction": "B",
                "agentic_evidence": "label B",
                "score_computed": True,
            },
        ]
    }
    second = {
        "detailed_results": [
            {"id": "one", "parse_success": True, "prediction": "A"},
            {"id": "two", "parse_success": True, "prediction": "A"},
        ]
    }

    agreement, disagreements = parser_agreement(first, second)

    assert agreement == 50.0
    assert disagreements[0]["id"] == "two"
    assert spatial_evidence_rate(first, required=True) == 50.0


def test_structured_noncompliance_keeps_the_readout_operational():
    first = {
        "detailed_results": [
            {
                "id": "one",
                "parse_success": False,
                "prediction": "no",
                "agentic_status": "ambiguous",
                "agentic_evidence": "No clear task-grounded placement is visible.",
                "model_protocol_noncompliance": True,
                "score_computed": True,
            }
        ]
    }
    second = json.loads(json.dumps(first))

    agreement, disagreements = parser_agreement(first, second)

    assert agreement == 100.0
    assert disagreements == []
    assert readout_operational_rate(first) == 100.0
    assert spatial_evidence_rate(first, required=True) == 100.0
    assert metric_compatibility_rate_among_valid(first) == 100.0


def test_smoke_failure_reasons_deduplicates_double_parse_errors():
    result = {
        "detailed_results": [
            {
                "generation_success": True,
                "parse_success": False,
                "parse_error_type": "ambiguous_clip_match",
            }
        ]
    }

    assert smoke_failure_reasons(result, result) == ["ambiguous_clip_match"]


def test_unverified_metric_still_satisfies_smoke_metric_compatibility():
    results = {
        "detailed_results": [
            {
                "parse_success": True,
                "score_computed": False,
                "metric_unverified": True,
            }
        ]
    }

    assert metric_compatibility_rate_among_valid(results) == 100.0


def test_double_parse_smoke_gate_passes_stable_protocol(tmp_path: Path):
    Image.new("RGB", (64, 64), "white").save(tmp_path / "sample.png")
    data_path = tmp_path / "toy.jsonl"
    data_path.write_text(json.dumps(_toy_item(), ensure_ascii=False) + "\n", encoding="utf-8")
    benchmark_cfg = AgenticProtocolBuilder(
        [_toy_item()],
        benchmark_name="toy",
        data_file=str(data_path),
        benchmark_root=str(tmp_path),
    ).build(raw_response=_agent_response()).benchmark_config
    args = argparse.Namespace(
        smoke_model="mock-copy",
        smoke_limit=1,
        smoke_output=str(tmp_path / "smoke"),
        mock_parse_response=(
            '{"status":"valid","prediction":"A",'
            '"evidence":"blue outlines and a relation arrow","confidence":"high"}'
        ),
        min_generation_rate=66.0,
        min_parse_success_rate=66.0,
        min_parser_agreement_rate=66.0,
        min_spatial_evidence_rate=66.0,
    )

    smoke = run_smoke_validation(args, benchmark_cfg)
    row = smoke["tasks"]["new_spatial_task"]

    assert row["status"] == "passed"
    assert row["parser_agreement_rate"] == 100.0
    assert Path(row["results_pass1"]).exists()
    assert Path(row["results_pass2"]).exists()


def test_fallback_smoke_requires_one_compliant_visual_example(tmp_path: Path):
    Image.new("RGB", (64, 64), "white").save(tmp_path / "sample.png")
    data_path = tmp_path / "toy.jsonl"
    data_path.write_text(json.dumps(_toy_item()) + "\n", encoding="utf-8")
    benchmark_cfg = AgenticProtocolBuilder(
        [_toy_item()],
        benchmark_name="toy",
        data_file=str(data_path),
        benchmark_root=str(tmp_path),
    ).build(raw_response=_agent_response()).benchmark_config
    args = argparse.Namespace(
        smoke_model="mock-copy",
        smoke_limit=1,
        smoke_output=str(tmp_path / "smoke"),
        mock_parse_response=json.dumps(
            {
                "status": "ambiguous",
                "prediction": "A",
                "evidence": "The requested relation arrow is not clearly visible.",
                "confidence": "low",
            }
        ),
        min_generation_rate=66.0,
        min_parse_success_rate=66.0,
        min_parser_agreement_rate=66.0,
        min_spatial_evidence_rate=66.0,
    )

    smoke = run_smoke_validation(args, benchmark_cfg)
    row = smoke["tasks"]["new_spatial_task"]

    assert row["readout_operational_rate"] == 100.0
    assert row["valid_parse_rate"] == 0.0
    assert row["status"] == "failed"
    assert "no_valid_protocol_example" in row["failed_checks"]


def test_smoke_retries_generation_then_reuses_one_image_for_both_parser_passes(
    monkeypatch, tmp_path: Path
):
    class FlakyCountingModel(MockProtocolModel):
        def __init__(self):
            super().__init__("mock-copy")
            self.calls = 0

        def generate_multi(self, image_paths, prompt, save_path):
            self.calls += 1
            if self.calls == 1:
                return False
            return super().generate_multi(image_paths, prompt, save_path)

    Image.new("RGB", (64, 64), "white").save(tmp_path / "sample.png")
    data_path = tmp_path / "toy.jsonl"
    data_path.write_text(json.dumps(_toy_item(), ensure_ascii=False) + "\n", encoding="utf-8")
    benchmark_cfg = AgenticProtocolBuilder(
        [_toy_item()],
        benchmark_name="toy",
        data_file=str(data_path),
        benchmark_root=str(tmp_path),
    ).build(raw_response=_agent_response()).benchmark_config
    model = FlakyCountingModel()
    monkeypatch.setattr(build_script, "ensure_model", lambda _model: model)
    args = argparse.Namespace(
        smoke_model="mock-flaky",
        smoke_limit=1,
        smoke_output=str(tmp_path / "smoke"),
        mock_parse_response=(
            '{"status":"valid","prediction":"A",'
            '"evidence":"blue outlines and a relation arrow","confidence":"high"}'
        ),
        min_generation_rate=66.0,
        min_parse_success_rate=66.0,
        min_parser_agreement_rate=66.0,
        min_spatial_evidence_rate=66.0,
    )

    smoke = run_smoke_validation(args, benchmark_cfg)
    row = smoke["tasks"]["new_spatial_task"]

    assert row["status"] == "passed"
    assert row["initial_generation_failure_count"] == 1
    assert row["generation_retry_count"] == 1
    assert model.calls == 2
    assert Path(row["results_pass1"]).exists()
    assert Path(row["results_pass2"]).exists()


def test_smoke_does_not_run_a_parser_pass_when_generation_never_succeeds(
    monkeypatch, tmp_path: Path
):
    class AlwaysFailModel(MockProtocolModel):
        def __init__(self):
            super().__init__("mock-fail")
            self.calls = 0

        def generate_multi(self, image_paths, prompt, save_path):
            self.calls += 1
            return False

    Image.new("RGB", (64, 64), "white").save(tmp_path / "sample.png")
    data_path = tmp_path / "toy.jsonl"
    data_path.write_text(json.dumps(_toy_item(), ensure_ascii=False) + "\n", encoding="utf-8")
    benchmark_cfg = AgenticProtocolBuilder(
        [_toy_item()],
        benchmark_name="toy",
        data_file=str(data_path),
        benchmark_root=str(tmp_path),
    ).build(raw_response=_agent_response()).benchmark_config
    model = AlwaysFailModel()
    monkeypatch.setattr(build_script, "ensure_model", lambda _model: model)
    args = argparse.Namespace(
        smoke_model="mock-fail",
        smoke_limit=1,
        smoke_output=str(tmp_path / "smoke"),
        mock_parse_response="",
        min_generation_rate=66.0,
        min_parse_success_rate=66.0,
        min_parser_agreement_rate=66.0,
        min_spatial_evidence_rate=66.0,
    )

    smoke = run_smoke_validation(args, benchmark_cfg)
    row = smoke["tasks"]["new_spatial_task"]
    second = json.loads(Path(row["results_pass2"]).read_text(encoding="utf-8"))

    assert row["status"] == "failed"
    assert row["phase"] == "initial"
    assert row["generated_rate"] == 0.0
    assert model.calls == 2
    assert second["detailed_results"][0]["generation_error_type"] != "reused_output_missing"


def test_agentic_protocol_smoke_with_mock_parser(tmp_path: Path):
    Image.new("RGB", (64, 64), "white").save(tmp_path / "sample.png")
    data_path = tmp_path / "toy.jsonl"
    data_path.write_text(json.dumps(_toy_item(), ensure_ascii=False) + "\n", encoding="utf-8")
    task_cfg = AgenticProtocolBuilder(
        [_toy_item()],
        benchmark_name="toy",
        data_file=str(data_path),
        benchmark_root=str(tmp_path),
    ).build(raw_response=_agent_response()).benchmark_config["tasks"]["new_spatial_task"]
    task_cfg["protocol_config"]["mock_parse_response"] = '{"status":"valid","prediction":"A","evidence":"blue outlines and a relation arrow","confidence":"high"}'

    args = argparse.Namespace(
        data_file=str(data_path),
        benchmark_root=str(tmp_path),
        limit=1,
        no_reuse=True,
        protocol="",
        print_prompt=False,
        model="mock-copy",
    )
    results = run_task(
        "new_spatial_task",
        task_cfg,
        load_protocol_pool("configs/protocol_specs"),
        args,
        MockProtocolModel("mock-copy"),
        tmp_path / "out",
    )
    assert results["valid_parse_count"] == 1
    assert results["correct_count"] == 1


def test_task_major_builder_finishes_each_task_before_starting_next(
    monkeypatch, tmp_path: Path
):
    first = _toy_item("first.png")
    first["id"] = "first"
    first["task"] = "alpha"
    second = _toy_item("second.png")
    second["id"] = "second"
    second["task"] = "beta"
    Image.new("RGB", (32, 32), "white").save(tmp_path / "first.png")
    Image.new("RGB", (32, 32), "white").save(tmp_path / "second.png")
    data_path = tmp_path / "toy.jsonl"
    data_path.write_text(
        "".join(json.dumps(item) + "\n" for item in (first, second)),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "configs" / "toy.manifest.json"

    class TaskAwareVLM:
        def __init__(self):
            self.tasks = []
            self.checkpoint_counts = []

        def predict_multi(self, _image_paths, prompt):
            task = "alpha" if '"task": "alpha"' in prompt else "beta"
            if task == "beta":
                checkpoint = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.checkpoint_counts.append(checkpoint["completed_task_count"])
            self.tasks.append(task)
            return json.dumps(
                {
                    "task": task,
                    "decision": "fallback",
                    "confidence": "medium",
                    "reason": "fixture uses a visible relation arrow",
                    "fallback": {
                        "generation_prompt": (
                            "For {question} and {choices}, draw one large relation arrow "
                            "between the referenced objects."
                        ),
                        "parse_prompt": "Read only the visible relation arrow.",
                        "visual_evidence": "a relation arrow between referenced objects",
                        "invalid_conditions": [
                            "the relation arrow is missing",
                            "the relation arrow is ambiguous",
                        ],
                    },
                }
            )

    args = argparse.Namespace(
        benchmark_name="toy_agentic",
        input=str(data_path),
        data_file=str(data_path),
        benchmark_root=str(tmp_path),
        max_examples_per_task=3,
        max_media_per_task=8,
        protocol_spec_dir="configs/protocol_specs",
        max_revisions=1,
        no_smoke=False,
        smoke_model="mock-copy",
        smoke_limit=1,
        smoke_output=str(tmp_path / "smoke"),
        reuse_smoke_images=False,
        mock_parse_response=json.dumps(
            {
                "status": "valid",
                "prediction": "A",
                "evidence": "a visible relation arrow between two objects",
                "confidence": "high",
            }
        ),
        min_generation_rate=66.0,
        min_parse_success_rate=66.0,
        min_parser_agreement_rate=66.0,
        min_spatial_evidence_rate=66.0,
    )
    events_path = tmp_path / "progress.jsonl"
    reporter = build_script.ProgressReporter(events_path, enabled=False)
    vlm = TaskAwareVLM()
    model_loads = []

    def load_model_once(model_name):
        model_loads.append(model_name)
        return MockProtocolModel(model_name)

    monkeypatch.setattr(build_script, "ensure_model", load_model_once)

    result, smoke, _ = build_tasks_sequentially(
        args,
        [first, second],
        vlm=vlm,
        raw_response="",
        reporter=reporter,
        output_paths={
            "benchmark_config_path": tmp_path / "configs" / "toy.yaml",
            "protocol_path": tmp_path / "generated" / "toy.yaml",
            "manifest_path": manifest_path,
            "prompt_path": tmp_path / "configs" / "toy.prompt.txt",
            "raw_response_path": tmp_path / "configs" / "toy.response.txt",
        },
    )

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    lifecycle = [
        (row["event"], row.get("task"))
        for row in events
        if row["event"] in {"task_workflow_started", "task_workflow_completed"}
    ]
    assert lifecycle == [
        ("task_workflow_started", "alpha"),
        ("task_workflow_completed", "alpha"),
        ("task_workflow_started", "beta"),
        ("task_workflow_completed", "beta"),
    ]
    assert vlm.tasks == ["alpha", "beta"]
    assert vlm.checkpoint_counts == [1]
    assert model_loads == ["mock-copy"]
    assert result.manifest["completed_task_count"] == 2
    assert set(result.benchmark_config["tasks"]) == {"alpha", "beta"}
    assert set(smoke["tasks"]) == {"alpha", "beta"}


def test_task_runner_retries_transient_generation_failure(tmp_path: Path):
    class TransientFailureModel(MockProtocolModel):
        def __init__(self):
            super().__init__("mock-copy")
            self.calls = 0
            self.last_error_type = ""
            self.last_error_message = ""

        def generate_multi(self, image_paths, prompt, save_path):
            self.calls += 1
            if self.calls == 1:
                self.last_error_type = "ChunkedEncodingError"
                self.last_error_message = "connection closed"
                return False
            return super().generate_multi(image_paths, prompt, save_path)

    Image.new("RGB", (64, 64), "white").save(tmp_path / "sample.png")
    data_path = tmp_path / "toy.jsonl"
    data_path.write_text(json.dumps(_toy_item()) + "\n", encoding="utf-8")
    task_cfg = AgenticProtocolBuilder(
        [_toy_item()],
        benchmark_name="toy",
        data_file=str(data_path),
        benchmark_root=str(tmp_path),
    ).build(raw_response=_agent_response()).benchmark_config["tasks"]["new_spatial_task"]
    task_cfg["protocol_config"]["mock_parse_response"] = (
        '{"status":"valid","prediction":"A",'
        '"evidence":"blue outlines and a relation arrow","confidence":"high"}'
    )
    model = TransientFailureModel()
    args = argparse.Namespace(
        data_file=str(data_path),
        benchmark_root=str(tmp_path),
        limit=1,
        no_reuse=True,
        reuse_only=False,
        generation_retries=1,
        protocol="",
        print_prompt=False,
        model="mock-transient",
    )

    results = run_task(
        "new_spatial_task",
        task_cfg,
        load_protocol_pool("configs/protocol_specs"),
        args,
        model,
        tmp_path / "out",
    )

    assert model.calls == 2
    assert results["generated_count"] == 1
    assert results["valid_parse_count"] == 1


def test_task_runner_retries_retryable_http_generation_failure(tmp_path: Path):
    class TransientHttpFailureModel(MockProtocolModel):
        def __init__(self):
            super().__init__("mock-copy")
            self.calls = 0
            self.last_error_type = ""
            self.last_error_message = ""

        def generate_multi(self, image_paths, prompt, save_path):
            self.calls += 1
            if self.calls == 1:
                self.last_error_type = "api_http_error"
                self.last_error_message = "HTTP 502: Bad Gateway"
                return False
            return super().generate_multi(image_paths, prompt, save_path)

    Image.new("RGB", (64, 64), "white").save(tmp_path / "sample.png")
    data_path = tmp_path / "toy.jsonl"
    data_path.write_text(json.dumps(_toy_item()) + "\n", encoding="utf-8")
    task_cfg = AgenticProtocolBuilder(
        [_toy_item()],
        benchmark_name="toy",
        data_file=str(data_path),
        benchmark_root=str(tmp_path),
    ).build(raw_response=_agent_response()).benchmark_config["tasks"]["new_spatial_task"]
    task_cfg["protocol_config"]["mock_parse_response"] = (
        '{"status":"valid","prediction":"A",'
        '"evidence":"blue outlines and a relation arrow","confidence":"high"}'
    )
    model = TransientHttpFailureModel()
    args = argparse.Namespace(
        data_file=str(data_path),
        benchmark_root=str(tmp_path),
        limit=1,
        no_reuse=True,
        reuse_only=False,
        generation_retries=1,
        generation_retry_backoff=0,
        protocol="",
        print_prompt=False,
        model="mock-transient-http",
    )

    results = run_task(
        "new_spatial_task",
        task_cfg,
        load_protocol_pool("configs/protocol_specs"),
        args,
        model,
        tmp_path / "out",
    )

    assert model.calls == 2
    assert results["generated_count"] == 1
    assert results["detailed_results"][0]["generation_retry_count"] == 1
