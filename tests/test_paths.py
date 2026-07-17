from pathlib import Path

from provise.paths import (
    benchmark_suite_path,
    protocol_spec_dir,
    runtime_root,
    source_root,
)


def test_source_checkout_resources_are_discoverable():
    root = source_root()

    assert root is not None
    assert (root / "pyproject.toml").is_file()
    assert (protocol_spec_dir() / "agentic_point_marker.yaml").is_file()
    assert benchmark_suite_path().is_file()


def test_runtime_root_honors_provise_home(monkeypatch, tmp_path: Path):
    home = tmp_path / "provise-home"
    monkeypatch.setenv("PROVISE_HOME", str(home))

    assert runtime_root() == home.resolve()
