import argparse
import json

from provise.evaluation.runner import (
    allocate_task_sample_limits,
    ensure_model,
    formal_evaluation_blocked_tasks,
    load_benchmark_runtime,
    resolve_frozen_runtime_path,
)
from provise.benchmark.schema import canonicalize_unified_item


def test_canonicalize_unified_item_sets_image_path_from_primary_media():
    item = {
        "id": "interaction_trajectory_1005",
        "input": {
            "type": "image",
            "media": [
                {"type": "image", "path": "interaction/trajectory/1005_frame_0.png", "role": "primary"}
            ],
        },
    }

    normalized = canonicalize_unified_item(item)

    assert normalized["image_path"] == "interaction/trajectory/1005_frame_0.png"


def test_canonicalize_unified_item_keeps_existing_image_path():
    item = {
        "image_path": "already/set.png",
        "input": {
            "type": "image",
            "media": [{"type": "image", "path": "should/not/replace.png", "role": "primary"}],
        },
    }

    normalized = canonicalize_unified_item(item)

    assert normalized["image_path"] == "already/set.png"


def test_canonicalize_unified_item_supports_nested_media_entries():
    item = {
        "id": "interaction_trajectory_1005",
        "input": {
            "type": "image",
            "media": [
                {
                    "role": "primary",
                    "media": {"type": "image", "path": "interaction/trajectory/1005_frame_0.png"},
                }
            ],
        },
    }

    normalized = canonicalize_unified_item(item)

    assert normalized["image_path"] == "interaction/trajectory/1005_frame_0.png"


def test_load_benchmark_runtime_reads_explicit_mapping(tmp_path):
    config_path = tmp_path / "toy.yaml"
    config_path.write_text(
        "benchmark: toy\n"
        "data_file: toy.jsonl\n"
        "benchmark_root: assets\n"
        "tasks:\n"
        "  counting:\n"
        "    protocol: instance_marker_count\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config="",
        benchmark_config=str(config_path),
    )

    benchmark_cfg, tasks_cfg, benchmark_name = load_benchmark_runtime(args)

    assert benchmark_name == "toy"
    assert benchmark_cfg["data_file"] == "toy.jsonl"
    assert tasks_cfg["counting"]["protocol"] == "instance_marker_count"


def test_formal_evaluation_gate_blocks_only_explicitly_unverified_tasks():
    tasks_cfg = {
        "legacy": {"protocol": "label_code"},
        "verified": {"formal_evaluation": True},
        "smoke_only": {"formal_evaluation": False},
    }

    blocked = formal_evaluation_blocked_tasks(
        ["legacy", "verified", "smoke_only"], tasks_cfg
    )

    assert blocked == ["smoke_only"]


def test_frozen_runtime_path_is_relative_to_artifact_root(tmp_path):
    config_path = tmp_path / "artifact" / "configs" / "toy_agentic.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("benchmark: toy\n", encoding="utf-8")

    resolved = resolve_frozen_runtime_path("benchmark/data.jsonl", str(config_path))

    assert resolved == str(tmp_path / "artifact" / "benchmark" / "data.jsonl")


def test_ensure_model_loads_registered_model(monkeypatch):
    class DummyModel:
        loaded = False

        def load_model(self):
            self.loaded = True

    model = DummyModel()
    monkeypatch.setattr(
        "provise.models.generative.create_model", lambda model_key: model
    )

    resolved = ensure_model("gpt-image-2")

    assert resolved is model
    assert model.loaded is True


def test_total_sample_budget_is_balanced_and_redistributed(tmp_path):
    data_path = tmp_path / "data.jsonl"
    rows = [
        {"id": f"a{index}", "task": "a"} for index in range(2)
    ] + [
        {"id": f"b{index}", "task": "b"} for index in range(10)
    ] + [
        {"id": f"c{index}", "task": "c"} for index in range(10)
    ]
    data_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    tasks = ["a", "b", "c"]
    tasks_cfg = {
        task: {"task_field": "task", "task_value": task} for task in tasks
    }

    limits = allocate_task_sample_limits(
        str(data_path),
        tasks,
        tasks_cfg,
        total_budget=10,
        per_task_limit=None,
    )

    assert limits == {"a": 2, "b": 4, "c": 4}
