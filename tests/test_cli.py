from provise import cli
from provise.reporting import compact_path, display_task_name


def test_cli_help_is_small_and_names_public_commands(capsys):
    assert cli.main(["--help"]) == 0

    output = capsys.readouterr().out
    assert "provise build --source BENCHMARK" in output
    assert "provise evaluate --protocol BUILD" in output
    assert "provise run --source BENCHMARK" in output
    assert "provise suite --benchmark-root DIRECTORY" in output
    assert "provise baseline --suite-output DIRECTORY" in output


def test_cli_run_forwards_the_minimal_arguments(monkeypatch):
    received = []

    def fake_run(args, **_kwargs):
        received.extend(args)
        return 7

    monkeypatch.setattr(cli, "run_agentic_benchmark", fake_run)

    code = cli.main(["run", "--source", "/tmp/bench", "--model", "image-model"])

    assert code == 7
    assert received == ["--source", "/tmp/bench", "--model", "image-model"]


def test_cli_build_forwards_build_only(monkeypatch):
    received = []

    def fake_run(args, **kwargs):
        received.extend(args)
        received.append(kwargs.get("command"))
        return 0

    monkeypatch.setattr(cli, "run_agentic_benchmark", fake_run)

    assert cli.main(["build", "--source", "/tmp/bench"]) == 0
    assert received == ["--source", "/tmp/bench", "build"]


def test_cli_evaluate_forwards_frozen_protocol(monkeypatch):
    received = []

    def fake_evaluate(args):
        received.extend(args)
        return 0

    monkeypatch.setattr(cli, "evaluate_frozen_protocol", fake_evaluate)

    assert cli.main(
        ["evaluate", "--protocol", "/tmp/build", "--model", "image-model"]
    ) == 0
    assert received == ["--protocol", "/tmp/build", "--model", "image-model"]


def test_cli_suite_forwards_suite_arguments(monkeypatch):
    received = []

    def fake_suite(args):
        received.extend(args)
        return 0

    monkeypatch.setattr(cli, "run_spatial_suite", fake_suite)

    assert cli.main(
        ["suite", "--benchmark-root", "/tmp/benchmarks", "--model", "image-model"]
    ) == 0
    assert received == [
        "--benchmark-root",
        "/tmp/benchmarks",
        "--model",
        "image-model",
    ]


def test_cli_baseline_forwards_arguments(monkeypatch):
    received = []

    def fake_baseline(args):
        received.extend(args)
        return 0

    monkeypatch.setattr(cli, "run_text_baseline", fake_baseline)

    assert cli.main(
        ["baseline", "--suite-output", "/tmp/suite", "--model", "vlm"]
    ) == 0
    assert received == ["--suite-output", "/tmp/suite", "--model", "vlm"]


def test_compact_path_shortens_default_output_workspace(tmp_path):
    project = tmp_path / "ProVisE"
    run = project / "outputs" / "agentic_runs" / "toy" / "run_1"
    run.mkdir(parents=True)

    assert compact_path(run, project) == "outputs/agentic_runs/toy/run_1"


def test_display_task_name_humanizes_contract_partition_slug():
    assert (
        display_task_name("spatial_reasoning__binary_boolean__accuracy")
        == "Spatial Reasoning (boolean)"
    )
    assert display_task_name("spatial_relation") == "Spatial Relation"
    assert display_task_name("Multi-view Reasoning") == "Multi-view Reasoning"
