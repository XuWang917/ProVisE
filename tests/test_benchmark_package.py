import json
from pathlib import Path

from PIL import Image

from provise.benchmark.package import (
    PACKAGE_SCHEMA_VERSION,
    infer_benchmark_name,
    probe_benchmark_package,
    write_package_manifest,
)
from provise.benchmark.schema import canonicalize_unified_item


def _item(image_path: str = "assets/scene.png") -> dict:
    return {
        "schema_version": "genbench.v1",
        "id": "sample_1",
        "benchmark": "package_toy",
        "task": "relation",
        "split": "test",
        "input": {
            "type": "image",
            "media": [{"type": "image", "path": image_path, "role": "primary"}],
        },
        "question": "Is the cup left of the plate?",
        "answer": "A",
        "answer_type": "choice",
        "choices": [
            {"label": "A", "text": "yes"},
            {"label": "B", "text": "no"},
        ],
        "evaluation": {"metric": "accuracy"},
        "metadata": {},
    }


def test_canonicalization_strips_legacy_metric_metadata():
    item = _item()
    item["evaluation"]["metric_provenance"] = {"source": "legacy/path.py"}

    normalized = canonicalize_unified_item(item)

    assert normalized["evaluation"] == {"metric": "accuracy"}


def test_probe_loads_manifested_normalized_package(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    Image.new("RGB", (24, 16), "white").save(assets / "scene.png")
    data_file = tmp_path / "data.jsonl"
    data_file.write_text(json.dumps(_item()) + "\n", encoding="utf-8")
    manifest = write_package_manifest(
        tmp_path / "benchmark.yaml",
        benchmark_name="package_toy",
        data_file="data.jsonl",
        benchmark_root=".",
    )

    probe = probe_benchmark_package(tmp_path)

    assert probe.status == "ready"
    assert probe.package is not None
    assert probe.package.benchmark_name == "package_toy"
    assert probe.package.data_file == data_file
    assert probe.package.benchmark_root == tmp_path
    assert probe.package.manifest_path == manifest
    assert probe.package.validation["valid"] is True


def test_probe_infers_assets_root_for_existing_normalized_layout(tmp_path: Path):
    assets = tmp_path / "assets"
    assets.mkdir()
    Image.new("RGB", (24, 16), "white").save(assets / "scene.png")
    data_file = tmp_path / "legacy.jsonl"
    data_file.write_text(json.dumps(_item("scene.png")) + "\n", encoding="utf-8")

    probe = probe_benchmark_package(tmp_path)

    assert probe.status == "ready"
    assert probe.package is not None
    assert probe.package.benchmark_root == assets


def test_probe_distinguishes_raw_data_from_invalid_normalized_data(tmp_path: Path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"question": "Where?", "image": "scene.png"}) + "\n")

    raw_probe = probe_benchmark_package(tmp_path)

    assert raw_probe.status == "absent"

    raw.write_text(json.dumps(_item("assets/missing.png")) + "\n", encoding="utf-8")
    invalid_probe = probe_benchmark_package(tmp_path)

    assert invalid_probe.status == "invalid"
    assert invalid_probe.validation["missing_media_count"] == 1


def test_package_manifest_has_public_schema_and_name_inference(tmp_path: Path):
    path = write_package_manifest(
        tmp_path / "benchmark.yaml",
        benchmark_name="my_benchmark",
        data_file="data.jsonl",
    )

    text = path.read_text(encoding="utf-8")
    assert f"schema_version: {PACKAGE_SCHEMA_VERSION}" in text
    assert infer_benchmark_name("/tmp/SpaCE-10") == "space_10"
    assert infer_benchmark_name("/tmp/example.unified.jsonl") == "example"
