import hashlib
import json
from pathlib import Path

import pytest
import yaml

import provise.commands.evaluate as evaluation_cli
from provise.commands.evaluate import resolve_frozen_protocol
from provise.evaluation.runner import load_protocol_pool, resolve_task_config
from provise.protocol_agent.builder import (
    AgenticProtocolBuildResult,
    write_build_outputs,
)
from provise.protocols import create_protocol


def _write_artifact(root: Path) -> Path:
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    config = {
        "benchmark": "toy_agentic",
        "data_file": "data.jsonl",
        "benchmark_root": ".",
        "protocol_build": {
            "schema_version": "provise.protocol.v1",
            "frozen": True,
            "agent_model": "agent",
            "parser_model": "parser",
            "validation_model": "image",
        },
        "tasks": {},
    }
    config_text = yaml.safe_dump(config, sort_keys=False)
    config_path = config_dir / "toy_agentic.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    manifest = {
        "protocol_artifact": {
            "schema_version": "provise.protocol.v1",
            "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        }
    }
    (config_dir / "toy_agentic.agentic_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return config_path


def test_resolve_frozen_protocol_accepts_build_directory(tmp_path: Path):
    config_path = _write_artifact(tmp_path)

    artifact = resolve_frozen_protocol(tmp_path)

    assert artifact.config_path == config_path
    assert artifact.config["protocol_build"]["parser_model"] == "parser"


def test_resolve_frozen_protocol_rejects_modified_config(tmp_path: Path):
    config_path = _write_artifact(tmp_path)
    config_path.write_text(config_path.read_text() + "# modified\n", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity check failed"):
        resolve_frozen_protocol(tmp_path)


def test_build_writer_produces_a_verifiable_frozen_artifact(tmp_path: Path):
    config_dir = tmp_path / "configs"
    result = AgenticProtocolBuildResult(
        benchmark_name="toy_agentic",
        benchmark_config={
            "benchmark": "toy_agentic",
            "data_file": "data.jsonl",
            "benchmark_root": ".",
            "protocol_build": {
                "schema_version": "provise.protocol.v1",
                "frozen": True,
                "agent_model": "agent",
                "parser_model": "parser",
                "validation_model": "image",
            },
            "tasks": {},
        },
        generated_protocols={"protocols": []},
        manifest={},
        prompt="prompt",
        raw_response="{}",
    )
    write_build_outputs(
        result,
        benchmark_config_path=config_dir / "toy_agentic.yaml",
        protocol_path=tmp_path / "generated" / "toy_agentic.yaml",
        manifest_path=config_dir / "toy_agentic.agentic_manifest.json",
        prompt_path=config_dir / "toy_agentic.prompt.txt",
        raw_response_path=config_dir / "toy_agentic.response.txt",
    )

    artifact = resolve_frozen_protocol(tmp_path)

    assert artifact.config["benchmark"] == "toy_agentic"


def test_evaluate_restores_parser_from_frozen_artifact(monkeypatch, tmp_path: Path):
    config_path = _write_artifact(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["tasks"] = {"spatial": {"formal_evaluation": True}}
    config_text = yaml.safe_dump(config, sort_keys=False)
    config_path.write_text(config_text, encoding="utf-8")
    manifest_path = config_path.with_name("toy_agentic.agentic_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol_artifact"]["config_sha256"] = hashlib.sha256(
        config_text.encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    captured = {}

    def fake_evaluate(args):
        captured["args"] = args
        captured["parser_model"] = evaluation_cli.os.environ.get(
            "PROVISE_PARSER_MODEL"
        )
        return 0

    monkeypatch.setattr(evaluation_cli, "run_protocol_eval", fake_evaluate)
    monkeypatch.setenv("PROVISE_PARSER_MODEL", "stale-parser")

    code = evaluation_cli.main(
        [
            "--protocol",
            str(tmp_path),
            "--model",
            "candidate",
            "--reuse-only",
            "--output",
            str(tmp_path / "results"),
        ]
    )

    assert code == 0
    assert captured["args"].model == "candidate"
    assert captured["args"].max_samples == 0
    assert captured["args"].reuse_only is True
    assert captured["parser_model"] == "parser"


def test_published_protocols_are_portable_and_verifiable():
    root = Path(__file__).resolve().parents[1] / "configs" / "protocols"
    artifacts = sorted(path for path in root.iterdir() if path.is_dir())

    assert [path.name for path in artifacts] == [
        "embspatial",
        "omnispatial",
        "q_spatial_plus",
        "roboafford",
        "robospatial_home",
        "sat",
        "spatialgen_bench",
    ]
    for artifact_root in artifacts:
        artifact = resolve_frozen_protocol(artifact_root)
        assert not Path(artifact.config["data_file"]).is_absolute()
        assert not Path(artifact.config["benchmark_root"]).is_absolute()
        generated = list((artifact_root / "generated").glob("*.agentic_protocols.yaml"))
        assert len(generated) == 1
        public_paths = list(artifact_root.glob("*.*"))
        for directory in ("configs", "generated"):
            public_paths.extend((artifact_root / directory).rglob("*"))
        for path in public_paths:
            if path.is_file():
                text = path.read_text(encoding="utf-8")
                assert "/home/" not in text
                assert "metric_provenance" not in text
                assert '"metric_status"' not in text


def test_omnispatial_image_embedded_choice_task_is_formally_evaluable():
    root = Path(__file__).resolve().parents[1] / "configs" / "protocols" / "omnispatial"

    artifact = resolve_frozen_protocol(root)
    task = "complex_logic__structured_or_text__accuracy"
    generated_path = next((root / "generated").glob("*.agentic_protocols.yaml"))
    generated = yaml.safe_load(generated_path.read_text(encoding="utf-8"))

    assert artifact.config["tasks"][task]["formal_evaluation"] is True
    assert task in evaluation_cli._evaluation_tasks(artifact.config, "")
    assert task in {row["task"] for row in generated["protocols"]}


def test_spatialgen_published_protocol_preserves_manual_and_agentic_origins():
    root = Path(__file__).resolve().parents[1]
    artifact_root = root / "configs" / "protocols" / "spatialgen_bench"
    artifact = resolve_frozen_protocol(artifact_root)
    tasks = artifact.config["tasks"]

    assert len(tasks) == 14
    assert sum(task["protocol_origin"] == "manual" for task in tasks.values()) == 11
    assert sum(task["protocol_origin"] == "agentic" for task in tasks.values()) == 3
    assert {
        name for name, task in tasks.items() if task["protocol_origin"] == "agentic"
    } == {"size", "grounding", "feasibility"}

    generated_path = next((artifact_root / "generated").glob("*.agentic_protocols.yaml"))
    generated = yaml.safe_load(generated_path.read_text(encoding="utf-8"))
    generated_ids = {row["id"] for row in generated["protocols"]}
    pool = load_protocol_pool(str(root / "configs" / "protocol_specs"))
    for task_name, task_config in tasks.items():
        protocol_name, protocol_config, prompt = resolve_task_config(
            task_name, task_config, pool
        )
        create_protocol(protocol_name, protocol_config)
        assert prompt
        generated_id = (task_config.get("protocol_config") or {}).get(
            "generated_protocol_id"
        )
        if generated_id:
            assert generated_id in generated_ids


def test_evaluate_passes_total_sample_budget(monkeypatch, tmp_path: Path):
    config_path = _write_artifact(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["tasks"] = {"spatial": {"formal_evaluation": True}}
    config_text = yaml.safe_dump(config, sort_keys=False)
    config_path.write_text(config_text, encoding="utf-8")
    manifest_path = config_path.with_name("toy_agentic.agentic_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol_artifact"]["config_sha256"] = hashlib.sha256(
        config_text.encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    captured = {}

    def fake_evaluate(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(evaluation_cli, "run_protocol_eval", fake_evaluate)

    code = evaluation_cli.main(
        [
            "--protocol",
            str(tmp_path),
            "--model",
            "candidate",
            "--max-samples",
            "24",
            "--output",
            str(tmp_path / "results"),
        ]
    )

    assert code == 0
    assert captured["args"].max_samples == 24


def test_evaluate_forwards_progress_settings(monkeypatch, tmp_path: Path):
    config_path = _write_artifact(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["tasks"] = {"spatial": {"formal_evaluation": True}}
    config_text = yaml.safe_dump(config, sort_keys=False)
    config_path.write_text(config_text, encoding="utf-8")
    manifest_path = config_path.with_name("toy_agentic.agentic_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol_artifact"]["config_sha256"] = hashlib.sha256(
        config_text.encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    captured = {}

    def fake_evaluate(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(evaluation_cli, "run_protocol_eval", fake_evaluate)
    event_path = tmp_path / "progress.jsonl"

    code = evaluation_cli.main(
        [
            "--protocol",
            str(tmp_path),
            "--progress-events",
            str(event_path),
            "--heartbeat-seconds",
            "2.5",
            "--output",
            str(tmp_path / "results"),
        ]
    )

    assert code == 0
    assert captured["args"].progress_events == str(event_path)
    assert captured["args"].heartbeat_seconds == 2.5
