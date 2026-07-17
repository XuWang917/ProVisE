import json
from pathlib import Path

from provise.commands import suite


def _write_suite(path: Path) -> None:
    path.write_text(
        "schema_version: provise.suite.v1\n"
        "name: toy_spatial\n"
        "benchmarks:\n"
        "  - id: present\n"
        "    source: PresentBench\n"
        "    family: relation\n"
        "  - id: absent\n"
        "    source: AbsentBench\n"
        "    family: grounding\n",
        encoding="utf-8",
    )


def test_load_suite_validates_and_preserves_entries(tmp_path: Path):
    path = tmp_path / "suite.yaml"
    _write_suite(path)

    name, entries = suite.load_suite(path)

    assert name == "toy_spatial"
    assert [entry.benchmark_id for entry in entries] == ["present", "absent"]
    assert entries[0].family == "relation"


def test_default_suite_matches_the_published_protocol_pool():
    _, entries = suite.load_suite(suite.DEFAULT_SUITE)
    protocol_root = suite.PROJECT_ROOT / "configs" / "protocols"
    published = sorted(path.name for path in protocol_root.iterdir() if path.is_dir())

    assert sorted(entry.benchmark_id for entry in entries) == published


def test_source_environment_override_is_optional(monkeypatch, tmp_path: Path):
    entry = suite.BenchmarkEntry(
        "toy", "ToyBench", source_env="PROVISE_TEST_BENCH_SOURCE"
    )

    assert suite.resolve_source(tmp_path, entry) == (tmp_path / "ToyBench").resolve()

    override = tmp_path / "prepared" / "ToyBench"
    monkeypatch.setenv("PROVISE_TEST_BENCH_SOURCE", str(override))
    assert suite.resolve_source(tmp_path, entry) == override.resolve()


def test_suite_command_uses_fixed_build_models_and_task_cap(tmp_path: Path):
    args = suite.parse_args(
        [
            "--benchmark-root",
            str(tmp_path),
            "--model",
            "candidate-image-model",
            "--agent-model",
            "agent-model",
            "--parser-model",
            "parser-model",
            "--protocol-model",
            "validation-image-model",
            "--max-tasks-per-benchmark",
            "4",
        ]
    )
    entry = suite.BenchmarkEntry("toy", "ToyBench")

    command = suite.build_benchmark_command(
        args,
        entry,
        tmp_path / "ToyBench",
        tmp_path / "workspace",
    )

    assert command[:4] == [suite.sys.executable, "-m", "provise.cli", "run"]
    assert command[command.index("--model") + 1] == "candidate-image-model"
    assert command[command.index("--agent-model") + 1] == "agent-model"
    assert command[command.index("--parser-model") + 1] == "parser-model"
    assert command[command.index("--protocol-model") + 1] == "validation-image-model"
    assert command[command.index("--max-tasks") + 1] == "4"
    assert command[command.index("--evaluate-samples") + 1] == "24"


def test_suite_dry_run_skips_missing_benchmark_and_writes_summary(tmp_path: Path):
    suite_path = tmp_path / "suite.yaml"
    _write_suite(suite_path)
    benchmark_root = tmp_path / "benchmarks"
    (benchmark_root / "PresentBench").mkdir(parents=True)
    output = tmp_path / "output"

    code = suite.main(
        [
            "--suite",
            str(suite_path),
            "--benchmark-root",
            str(benchmark_root),
            "--output",
            str(output),
            "--dry-run",
        ]
    )

    assert code == 0
    manifest = json.loads((output / "suite_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed_with_skips"
    assert [row["status"] for row in manifest["benchmarks"]] == ["planned", "missing"]
    assert (output / "summary.csv").is_file()
    assert (output / "summary.md").is_file()


def test_suite_classifies_action_required_as_blocked(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "action_required.json").write_text(
        json.dumps(
            {
                "stage": "benchmark_ingestion",
                "reason": "episode images are missing",
            }
        ),
        encoding="utf-8",
    )
    entry = suite.BenchmarkEntry("episodic", "OpenEQA")

    row = suite.summarize_benchmark_run(
        entry,
        tmp_path / "OpenEQA",
        workspace,
        return_code=2,
        elapsed_seconds=1.0,
        run_manifest={},
    )

    assert row["status"] == "blocked"
    assert row["error"] == "episode images are missing"
    totals = suite.aggregate_suite([row])
    assert totals["blocked_benchmarks"] == 1
    assert totals["failed_benchmarks"] == 0
    assert suite.suite_status([row], strict_missing=False) == "completed_with_blocks"


def test_suite_main_returns_success_when_available_benchmark_is_blocked(
    monkeypatch, tmp_path: Path
):
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "schema_version: provise.suite.v1\n"
        "name: blocked_suite\n"
        "benchmarks:\n"
        "  - id: episodic\n"
        "    source: OpenEQA\n",
        encoding="utf-8",
    )
    benchmark_root = tmp_path / "benchmarks"
    (benchmark_root / "OpenEQA").mkdir(parents=True)
    output = tmp_path / "output"

    def fake_run(command, *, cwd, timeout_seconds):
        workspace = Path(command[command.index("--workspace") + 1])
        workspace.mkdir(parents=True)
        (workspace / "action_required.json").write_text(
            json.dumps(
                {"stage": "benchmark_ingestion", "reason": "episode frames missing"}
            ),
            encoding="utf-8",
        )
        return 2, ""

    monkeypatch.setattr(suite, "run_command", fake_run)

    code = suite.main(
        [
            "--suite",
            str(suite_path),
            "--benchmark-root",
            str(benchmark_root),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    manifest = json.loads((output / "suite_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed_with_blocks"
