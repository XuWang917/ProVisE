import json
from pathlib import Path
import sys

import pytest
import yaml
from PIL import Image

import provise.commands.agentic as agentic_runner


def _load_script_module():
    return agentic_runner


def test_evaluation_subprocess_env_uses_requested_parser_model(monkeypatch):
    module = _load_script_module()
    monkeypatch.setenv("PROVISE_PARSER_MODEL", "stale-model")

    env = module.evaluation_subprocess_env("gpt-5.4")

    assert env["PROVISE_PARSER_MODEL"] == "gpt-5.4"


def test_evaluation_task_sets_excludes_unverified_metrics_from_formal_scoring():
    module = _load_script_module()
    config = {
        "tasks": {
            "verified": {"formal_evaluation": True},
            "smoke_only": {"formal_evaluation": False},
        }
    }

    active, formal, unverified = module.evaluation_task_sets(config)

    assert active == ["verified", "smoke_only"]
    assert formal == ["verified"]
    assert unverified == ["smoke_only"]


def test_cli_accepts_saved_protocol_response(monkeypatch):
    module = _load_script_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "provise",
            "--source-root",
            "/tmp/bench",
            "--benchmark-name",
            "toy",
            "--protocol-response-file",
            "/tmp/protocol.json",
            "--reuse-smoke-images",
        ],
    )

    args = module.parse_args()

    assert args.protocol_response_file == "/tmp/protocol.json"
    assert args.reuse_smoke_images is True


def test_cli_minimal_surface_accepts_source_and_model():
    module = _load_script_module()

    args = module.parse_args(
        ["--source", "/tmp/new-benchmark", "--model", "gpt-image-2"]
    )

    assert args.source_root == "/tmp/new-benchmark"
    assert args.evaluation_model == "gpt-image-2"
    assert args.protocol_model == "gpt-image-2"
    assert args.benchmark_name == ""
    assert args.workspace == ""
    assert args.evaluate_limit is None
    assert args.verbose is False
    assert args.heartbeat_seconds == 1.0
    assert args.max_tasks == 0


def test_cli_accepts_verbose_output_mode():
    module = _load_script_module()

    args = module.parse_args(["--source", "/tmp/new-benchmark", "--verbose"])

    assert args.verbose is True


def test_final_message_reports_task_coverage():
    module = _load_script_module()

    message = module.final_run_message(
        "completed",
        active_task_count=5,
        selected_task_count=6,
        total_task_count=6,
    )

    assert message == "Run completed: 5/6 tasks active"


def test_representative_task_selection_prefers_contract_diversity(tmp_path: Path):
    module = _load_script_module()
    data_path = tmp_path / "data.jsonl"
    rows = []
    for task, count, metric, answer_type in (
        ("large_choice", 4, "accuracy", "choice"),
        ("small_choice", 3, "accuracy", "choice"),
        ("numeric", 1, "numeric_absolute_error", "number"),
    ):
        for index in range(count):
            rows.append(
                {
                    "task": task,
                    "answer_type": answer_type,
                    "choices": (
                        [{"label": "A", "text": "left"}, {"label": "B", "text": "right"}]
                        if answer_type == "choice"
                        else []
                    ),
                    "evaluation": {"metric": metric},
                    "id": f"{task}_{index}",
                }
            )
    data_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    selected = module.select_representative_tasks(data_path, 2)

    assert selected == ["large_choice", "numeric"]


def test_final_message_marks_explicit_task_subset_as_partial_coverage():
    module = _load_script_module()

    message = module.final_run_message(
        "protocol_ready",
        active_task_count=1,
        selected_task_count=1,
        total_task_count=6,
    )

    assert message == (
        "Run completed: 1/1 selected tasks active; benchmark coverage 1/6"
    )


def test_final_exit_code_propagates_evaluation_failure():
    module = _load_script_module()

    assert module.final_exit_code("completed", active_task_count=2) == 0
    assert module.final_exit_code("evaluation_failed", active_task_count=2) == 4
    assert module.final_exit_code("no_active_tasks", active_task_count=0) == 3


def test_evaluation_summary_separates_model_results_from_pipeline_failures():
    module = _load_script_module()

    message = module.evaluation_summary_message(
        {
            "total_samples": 10,
            "correct_count": 4,
            "accuracy": 40.0,
            "valid_parse_count": 5,
            "generation_failed_count": 3,
            "parser_failure_count": 2,
        }
    )

    assert message == (
        "Pilot: 4/10 correct (40.0%); 5 valid; "
        "3 generation failures; 2 parser failures"
    )


def test_cli_rejects_more_than_one_protocol_revision(monkeypatch):
    module = _load_script_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "provise",
            "--source-root",
            "/tmp/bench",
            "--benchmark-name",
            "toy",
            "--max-revisions",
            "2",
        ],
    )

    with pytest.raises(SystemExit):
        module.parse_args()


def test_one_command_pipeline_ingests_and_smokes_automatic_vlm_fallback(
    monkeypatch, tmp_path: Path
):
    module = _load_script_module()
    source = tmp_path / "unseen_benchmark"
    data = source / "data"
    data.mkdir(parents=True)
    Image.new("RGB", (64, 48), "white").save(data / "scene.png")
    (data / "test.jsonl").write_text(
        json.dumps(
            {
                "question": "Is the cup left of the plate?",
                "answer": "A",
                "A": "yes",
                "B": "no",
                "image": "scene.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "eval.py").write_text(
        "def accuracy(prediction, answer):\n    return float(prediction == answer)\n",
        encoding="utf-8",
    )
    ingestion_response = tmp_path / "ingestion.json"
    ingestion_response.write_text(
        json.dumps(
            {
                "benchmark": "blind_toy",
                "decision": "ingest",
                "reason": "Official image, answer, choices, and accuracy are explicit.",
                "sources": [
                    {
                        "source": "data/test.jsonl",
                        "split": "test",
                        "id": {"mode": "row_index", "prefix": "blind"},
                        "task": {"constant": "spatial_relation"},
                        "question": {"field": "question", "transform": "text"},
                        "answer": {"mode": "field", "field": "answer", "transform": "text"},
                        "answer_type": "choice",
                        "choices": {
                            "mode": "fields",
                            "fields": ["A", "B"],
                            "labels": ["A", "B"],
                        },
                        "media": [{"field": "image", "mode": "path", "role": "primary"}],
                        "evaluation": {
                            "metric": "accuracy",
                            "metric_config": {},
                        },
                        "metadata_fields": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    protocol_response = tmp_path / "protocol.json"
    protocol_response.write_text(
        json.dumps(
            {
                "task": "spatial_relation",
                "decision": "unsupported",
                "confidence": "medium",
                "reason": "No deterministic readout was proposed by this fixture.",
            }
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "provise",
            "--source-root",
            str(source),
            "--benchmark-name",
            "blind_toy",
            "--workspace",
            str(workspace),
            "--ingestion-response-file",
            str(ingestion_response),
            "--protocol-response-file",
            str(protocol_response),
            "--smoke-model",
            "mock-copy",
            "--smoke-limit",
            "1",
            "--evaluate-limit",
            "0",
            "--mock-parse-response",
            json.dumps(
                {
                    "status": "valid",
                    "prediction": "A",
                    "evidence": "a visible object outline and relation arrow",
                    "confidence": "high",
                }
            ),
        ],
    )

    assert module.main() == 0

    config = yaml.safe_load(
        (workspace / "configs" / "blind_toy_agentic.yaml").read_text(encoding="utf-8")
    )
    task = config["tasks"]["spatial_relation"]
    assert task["protocol"] == "agentic_vlm_protocol"
    assert task["formal_evaluation"] is True
    assert "metric_provenance" not in json.dumps(config)
    manifest = json.loads(
        (workspace / "configs" / "blind_toy_agentic.agentic_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["automatic_vlm_fallback_tasks"] == ["spatial_relation"]
    assert manifest["smoke_validation"]["tasks"]["spatial_relation"]["status"] == "passed"
    assert (workspace / "blind_toy.ingestion_attempts.json").exists()
    assert (workspace / "blind_toy.agentic_run_manifest.json").exists()


def test_one_command_uses_normalized_package_without_loading_ingestion_agent(
    monkeypatch, tmp_path: Path
):
    module = _load_script_module()
    source = tmp_path / "ready_benchmark"
    assets = source / "assets"
    assets.mkdir(parents=True)
    Image.new("RGB", (48, 32), "white").save(assets / "scene.png")
    item = {
        "schema_version": "genbench.v1",
        "id": "ready_1",
        "benchmark": "ready_benchmark",
        "task": "spatial_relation",
        "split": "test",
        "input": {
            "type": "image",
            "media": [{"type": "image", "path": "assets/scene.png", "role": "primary"}],
        },
        "question": "Is the cup left of the plate?",
        "answer": "A",
        "answer_type": "choice",
        "choices": [{"label": "A", "text": "yes"}, {"label": "B", "text": "no"}],
        "evaluation": {"metric": "unverified"},
        "metadata": {},
    }
    (source / "data.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
    (source / "benchmark.yaml").write_text(
        "schema_version: provise.benchmark.v1\n"
        "benchmark: ready_benchmark\n"
        "data_file: data.jsonl\n"
        "benchmark_root: .\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"

    def fail_if_loaded(*_args, **_kwargs):
        raise AssertionError("normalized packages must bypass the ingestion VLM")

    monkeypatch.setattr(module, "create_eval_vlm", fail_if_loaded)

    assert module.main(
        [
            "--source",
            str(source),
            "--model",
            "gpt-image-2",
            "--workspace",
            str(workspace),
            "--ingest-only",
        ]
    ) == 0

    manifest = json.loads(
        (workspace / "ready_benchmark.ingestion_manifest.json").read_text(encoding="utf-8")
    )
    progress = [
        json.loads(line)
        for line in (workspace / "ready_benchmark.progress.jsonl").read_text().splitlines()
    ]
    assert manifest["mode"] == "normalized_package"
    assert manifest["validation"]["valid"] is True
    assert any(
        row.get("stage") == 2
        and row["message"] == "Ingestion agent skipped: normalized package detected"
        for row in progress
    )
