import io
import json
import base64
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
from tfrecord.writer import TFRecordWriter

from provise.benchmark.ingestion import (
    AgenticBenchmarkIngestor,
    discover_metric_evidence,
    discover_sources,
    metric_evidence_supports,
    normalize_answer,
    parse_inline_choices,
    write_ingestion_outputs,
)
from provise.benchmark.package import probe_benchmark_package
from provise.benchmark.tasks import partition_heterogeneous_tasks


def _image_bytes(color: str, *, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (24, 16), color).save(buffer, format=fmt)
    return buffer.getvalue()


def _base_source_mapping(source: str) -> dict:
    return {
        "source": source,
        "split": "test",
        "id": {"mode": "row_index", "prefix": "toy"},
        "task": {"field": "category", "transform": "slug", "default": "default"},
        "question": {"field": "question", "transform": "text"},
        "answer": {"field": "answer", "transform": "text"},
        "answer_type": "choice",
        "choices": {"mode": "inline_mcq"},
        "media": [{"field": "image", "mode": "path", "role": "primary"}],
        "evaluation": {"metric": "accuracy"},
        "metadata_fields": ["category"],
    }


def _response(source_mapping: dict, benchmark: str = "toy") -> str:
    return json.dumps(
        {
            "benchmark": benchmark,
            "decision": "ingest",
            "reason": "The official fields and accuracy metric are explicit.",
            "sources": [source_mapping],
        }
    )


class _SequenceVLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def predict(self, _image_path, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_jsonl_ingestion_extracts_choices_and_deduplicates_media(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    Image.new("RGB", (24, 16), "red").save(data_dir / "scene.png")
    rows = [
        {
            "category": "Spatial Relation",
            "question": "Where is the cup? Choices: A. left. B. right. Please answer only the letter.",
            "answer": "A",
            "image": "scene.png",
        },
        {
            "category": "Spatial Relation",
            "question": "Where is the plate? Choices: A. front. B. behind. Please answer only the letter.",
            "answer": "B",
            "image": "scene.png",
        },
    ]
    data_path = data_dir / "test.jsonl"
    data_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    mapping = _base_source_mapping("data/test.jsonl")
    output_root = tmp_path / "out"

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="toy",
        output_root=output_root,
    ).build(raw_response=_response(mapping))

    assert result.decision == "ingest"
    assert len(result.items) == 2
    assert result.items[0]["task"] == "spatial_relation"
    assert [choice["label"] for choice in result.items[0]["choices"]] == ["A", "B"]
    assert result.items[0]["input"]["media"][0]["path"] == result.items[1]["input"]["media"][0]["path"]
    assert result.manifest["validation"]["valid"]
    paths = write_ingestion_outputs(result, output_root)
    assert Path(paths["unified_data"]).read_text(encoding="utf-8").count("\n") == 2
    assert Path(paths["package"]).name == "benchmark.yaml"
    assert probe_benchmark_package(output_root).status == "ready"


def test_jsonl_ingestion_extracts_chat_style_text_and_images(tmp_path: Path):
    source_root = tmp_path / "source"
    json_dir = source_root / "json"
    frame_dir = source_root / "data" / "episode"
    json_dir.mkdir(parents=True)
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (24, 16), "red").save(frame_dir / "00000.jpg")
    Image.new("RGB", (24, 16), "blue").save(frame_dir / "00015.jpg")
    row = {
        "id": 7,
        "task_type": "distance_compare",
        "conversation": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "episode/00000.jpg"},
                    {"type": "image", "image": "episode/00015.jpg"},
                    {"type": "text", "text": "Which object is closer?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "The chair is closer."}],
            },
        ],
    }
    (json_dir / "spatial.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    mapping = {
        "source": "json/spatial.jsonl",
        "split": "test",
        "id": {"mode": "field", "field": "id", "prefix": "chat"},
        "task": {"field": "task_type", "transform": "text"},
        "question": {"field": "conversation", "transform": "conversation_user_text"},
        "answer": {
            "mode": "field",
            "field": "conversation",
            "transform": "conversation_assistant_text",
        },
        "answer_type": "text",
        "choices": {"mode": "none"},
        "media": [
            {"field": "conversation", "mode": "conversation_images", "role": "frame"}
        ],
        "evaluation": {"metric": "exact_match"},
        "metadata_fields": ["task_type"],
    }

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="chat_toy",
        output_root=tmp_path / "out",
    ).build(raw_response=_response(mapping, benchmark="chat_toy"))

    assert result.decision == "ingest"
    assert result.items[0]["question"] == "Which object is closer?"
    assert result.items[0]["answer"] == "The chair is closer."
    assert len(result.items[0]["input"]["media"]) == 2
    assert (tmp_path / "out" / result.items[0]["input"]["media"][0]["path"]).is_file()

    (frame_dir / "00015.jpg").unlink()
    missing_result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="chat_toy",
        output_root=tmp_path / "out_missing",
    ).build(raw_response=_response(mapping, benchmark="chat_toy"))

    assert missing_result.decision == "unsupported"
    assert missing_result.manifest["blocker_type"] == "missing_media"
    assert "resolvable input media" in missing_result.manifest["reason"]
    assert missing_result.manifest["source_extraction"]["json/spatial.jsonl"][
        "skipped_count"
    ] == 1

    (frame_dir / "00000.jpg").unlink()
    frame_dir.rmdir()
    frame_dir.parent.rmdir()
    vlm = _SequenceVLM([_response(mapping, benchmark="chat_toy")])
    agent_result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="chat_toy",
        output_root=tmp_path / "out_agent",
        max_revisions=1,
    ).build(vlm=vlm)

    assert agent_result.manifest["blocker_type"] == "missing_media"
    assert agent_result.manifest["revision_count"] == 0
    assert len(vlm.prompts) == 1


def test_json_ingestion_resolves_field_derived_image_path_template(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    image_dir = data_dir / "Dynamic_Reasoning"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (24, 16), "teal").save(image_dir / "17.jpg")
    (data_dir / "data.json").write_text(
        json.dumps(
            [
                {
                    "id": "17_2",
                    "question": "Which direction will the object move?",
                    "options": ["left", "right"],
                    "answer": 1,
                    "task_type": "Dynamic_Reasoning",
                }
            ]
        ),
        encoding="utf-8",
    )
    mapping = {
        "source": "data/data.json",
        "split": "test",
        "id": {"mode": "field", "field": "id", "prefix": "derived"},
        "task": {"field": "task_type", "transform": "slug"},
        "question": {"field": "question", "transform": "text"},
        "answer": {
            "field": "answer",
            "transform": "raw",
            "choice_index_base": 0,
        },
        "answer_type": "choice",
        "choices": {"mode": "field", "field": "options"},
        "media": [
            {
                "field": "id",
                "mode": "path_template",
                "template": "{task_type}/{value}",
                "value_transform": "first_underscore",
                "role": "primary",
            }
        ],
        "evaluation": {"metric": "accuracy"},
        "metadata_fields": ["task_type"],
    }
    output_root = tmp_path / "out"

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="derived_media",
        output_root=output_root,
    ).build(raw_response=_response(mapping, benchmark="derived_media"))

    assert result.decision == "ingest"
    assert result.items[0]["task"] == "dynamic_reasoning"
    assert result.items[0]["answer"] == "B"
    assert (output_root / result.items[0]["input"]["media"][0]["path"]).is_file()
    assert "path_template" in result.prompt


def test_json_ingestion_normalizes_image_embedded_choice_index(tmp_path: Path):
    source_root = tmp_path / "source"
    image_dir = source_root / "Visual_Choice"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (24, 16), "teal").save(image_dir / "0.png")
    (source_root / "data.json").write_text(
        json.dumps(
            [
                {
                    "id": "0_0",
                    "question": "Which option printed in the image is correct?",
                    "options": [],
                    "answer": 3,
                    "task_type": "Visual_Choice",
                }
            ]
        ),
        encoding="utf-8",
    )
    mapping = {
        "source": "data.json",
        "split": "test",
        "id": {"mode": "field", "field": "id", "prefix": "embedded"},
        "task": {"field": "task_type", "transform": "slug"},
        "question": {"field": "question", "transform": "text"},
        "answer": {
            "field": "answer",
            "transform": "raw",
            "choice_index_base": 0,
        },
        "answer_type": "choice",
        "choices": {
            "mode": "field",
            "field": "options",
            "labels": ["A", "B", "C", "D"],
        },
        "media": [
            {
                "field": "id",
                "mode": "path_template",
                "template": "{task_type}/{value}",
                "value_transform": "first_underscore",
                "role": "primary",
            }
        ],
        "evaluation": {"metric": "accuracy"},
    }

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="image_embedded_choices",
        output_root=tmp_path / "out",
    ).build(raw_response=_response(mapping, benchmark="image_embedded_choices"))

    assert result.decision == "ingest"
    assert result.items[0]["choices"] == [
        {"label": "A", "text": ""},
        {"label": "B", "text": ""},
        {"label": "C", "text": ""},
        {"label": "D", "text": ""},
    ]
    assert result.items[0]["answer"] == "D"


def test_numeric_choice_without_choices_requires_explicit_labels():
    with pytest.raises(ValueError, match="explicit choices.labels"):
        normalize_answer(3, "choice", [], choice_index_base=0)


def test_label_only_numeric_choice_requires_explicit_index_base():
    choices = [{"label": str(index), "text": ""} for index in range(1, 5)]
    with pytest.raises(ValueError, match="explicit choice_index_base"):
        normalize_answer(3, "choice", choices)
    assert normalize_answer(3, "choice", choices, choice_index_base=0) == "4"


def test_json_ingestion_repairs_zero_coverage_path_template_transform(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    image_dir = data_dir / "Dynamic_Reasoning"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (24, 16), "teal").save(image_dir / "17.jpg")
    (data_dir / "data.json").write_text(
        json.dumps(
            [
                {
                    "id": "17_2",
                    "question": "Which direction will the object move?",
                    "options": ["left", "right"],
                    "answer": 1,
                    "task_type": "Dynamic_Reasoning",
                }
            ]
        ),
        encoding="utf-8",
    )
    mapping = {
        "source": "data/data.json",
        "split": "test",
        "id": {"mode": "field", "field": "id", "prefix": "derived"},
        "task": {"field": "task_type", "transform": "slug"},
        "question": {"field": "question", "transform": "text"},
        "answer": {
            "field": "answer",
            "transform": "raw",
            "choice_index_base": 0,
        },
        "answer_type": "choice",
        "choices": {"mode": "field", "field": "options"},
        "media": [
            {
                "field": "id",
                "mode": "path_template",
                "template": "{task_type}/{value}",
                "value_transform": "text",
                "role": "primary",
            }
        ],
        "evaluation": {"metric": "accuracy"},
        "metadata_fields": ["task_type"],
    }

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="repaired_media",
        output_root=tmp_path / "out",
    ).build(raw_response=_response(mapping, benchmark="repaired_media"))

    assert result.decision == "ingest"
    assert result.mapping["sources"][0]["media"][0]["value_transform"] == "first_underscore"
    assert result.manifest["deterministic_mapping_repairs"] == [
        {
            "source": "data/data.json",
            "media_index": 0,
            "from_transform": "text",
            "to_transform": "first_underscore",
            "probe_count": 1,
        }
    ]


def test_path_template_repair_does_not_guess_between_distinct_full_coverage_paths(
    tmp_path: Path,
):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    Image.new("RGB", (24, 16), "red").save(data_dir / "scene.jpg")
    Image.new("RGB", (24, 16), "blue").save(data_dir / "scene.png")
    (data_dir / "data.jsonl").write_text(
        json.dumps(
            {
                "id": "nested/scene.jpg",
                "category": "relation",
                "question": "Where? Choices: A. left. B. right.",
                "answer": "A",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    mapping = _base_source_mapping("data/data.jsonl")
    mapping["media"] = [
        {
            "field": "id",
            "mode": "path_template",
            "template": "{value}",
            "value_transform": "text",
            "role": "primary",
        }
    ]

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="ambiguous_media",
        output_root=tmp_path / "out",
        max_revisions=0,
    ).build(raw_response=_response(mapping, benchmark="ambiguous_media"))

    assert result.decision == "unsupported"
    assert result.manifest["deterministic_mapping_repairs"] == []


def test_ingestion_rejects_silently_dropped_source_rows(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    data_path = data_dir / "test.jsonl"
    data_path.write_text(
        json.dumps(
            {
                "category": "relation",
                "question": "Where is the cup? Choices: A. left. B. right.",
                "answer": "A",
                "image": "missing.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    mapping = _base_source_mapping("data/test.jsonl")

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="incomplete_toy",
        output_root=tmp_path / "out",
        max_revisions=0,
    ).build(raw_response=_response(mapping, benchmark="incomplete_toy"))

    assert result.decision == "unsupported"
    assert result.items == []
    assert not result.manifest["validation"]["valid"]
    assert "silently dropped" in result.manifest["validation"]["errors"][0]
    assert result.manifest["source_extraction"]["data/test.jsonl"] == {
        "record_count": 1,
        "converted_count": 0,
        "skipped_count": 1,
        "coverage_rate": 0.0,
    }


def test_tsv_ingestion_supports_base64_images_and_separate_choice_columns(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    image = base64.b64encode(_image_bytes("orange")).decode("ascii")
    (data_dir / "test.tsv").write_text(
        "index\tcategory\tquestion\tanswer\tA\tB\timage\n"
        f"7\trelation\tWhere is the cup?\tB\tleft\tright\t{image}\n",
        encoding="utf-8",
    )
    mapping = {
        "source": "data/test.tsv",
        "split": "test",
        "id": {"mode": "field", "field": "index", "prefix": "tsv"},
        "task": {"field": "category", "transform": "slug"},
        "question": {"field": "question", "transform": "text"},
        "answer": {"field": "answer", "transform": "text"},
        "answer_type": "choice",
        "choices": {"mode": "fields", "fields": ["A", "B"], "labels": ["A", "B"]},
        "media": [{"field": "image", "mode": "embedded_images", "role": "primary"}],
        "evaluation": {"metric": "unverified"},
        "metadata_fields": [],
    }

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="tsv_toy",
        output_root=tmp_path / "out",
    ).build(raw_response=_response(mapping, benchmark="tsv_toy"))

    assert result.decision == "ingest"
    assert result.items[0]["id"] == "tsv_7"
    assert result.items[0]["answer"] == "B"
    assert [choice["text"] for choice in result.items[0]["choices"]] == ["left", "right"]
    assert '"type": "base64_image"' in result.prompt


def test_nested_json_prefers_official_test_collection_over_train(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    Image.new("RGB", (24, 16), "blue").save(data_dir / "scene.png")
    train = [
        {
            "category": "train",
            "question": f"Train question {index}",
            "answer": "A",
            "image": "scene.png",
        }
        for index in range(5)
    ]
    test = [
        {
            "category": "relation",
            "question": "Official test question",
            "answer": "yes",
            "image": "scene.png",
        }
    ]
    (data_dir / "benchmark.json").write_text(
        json.dumps({"splits": {"train": train, "test": test}}), encoding="utf-8"
    )
    mapping = {
        "source": "data/benchmark.json",
        "split": "test",
        "id": {"mode": "row_index", "prefix": "nested"},
        "task": {"field": "category", "transform": "slug"},
        "question": {"field": "question", "transform": "text"},
        "answer": {"field": "answer", "transform": "text"},
        "answer_type": "boolean",
        "choices": {"mode": "boolean"},
        "media": [{"field": "image", "mode": "path", "role": "primary"}],
        "evaluation": {"metric": "unverified"},
        "metadata_fields": [],
    }

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="nested_toy",
        output_root=tmp_path / "out",
    ).build(raw_response=_response(mapping, benchmark="nested_toy"))

    assert result.decision == "ingest"
    assert len(result.items) == 1
    assert result.items[0]["question"] == "Official test question"
    assert result.items[0]["task"] == "relation"
    assert result.manifest["inventory_summary"]["record_collections"] == {
        "data/benchmark.json": "splits.test"
    }


def test_live_ingestion_repairs_a_semantically_invalid_mapping_once(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    Image.new("RGB", (24, 16), "red").save(data_dir / "scene.png")
    (data_dir / "test.jsonl").write_text(
        json.dumps(
            {
                "category": "relation",
                "question": "Where is the cup?",
                "answer": "B",
                "option_left": "left",
                "option_right": "right",
                "image": "scene.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    first = _base_source_mapping("data/test.jsonl")
    first["choices"] = {"mode": "field", "field": "option_left"}
    second = _base_source_mapping("data/test.jsonl")
    second["choices"] = {
        "mode": "fields",
        "fields": ["option_left", "option_right"],
        "labels": ["A", "B"],
    }
    vlm = _SequenceVLM([_response(first), _response(second)])

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="repair_toy",
        output_root=tmp_path / "out",
        max_revisions=1,
    ).build(vlm=vlm)

    assert result.decision == "ingest"
    assert result.manifest["revision_count"] == 1
    assert len(result.attempts) == 2
    assert result.attempts[0]["diagnostics"]["validation_errors"] == [
        {"message": "answer is not one of the choice labels", "reported_count": 1}
    ]
    assert "INGESTION MAPPING REPAIR" in vlm.prompts[1]
    assert [choice["label"] for choice in result.items[0]["choices"]] == ["A", "B"]


def test_multi_source_ingestion_namespaces_row_ids(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    Image.new("RGB", (24, 16), "red").save(data_dir / "scene.png")
    row = {
        "category": "Spatial Relation",
        "question": "Where is the cup? Choices: A. left. B. right.",
        "answer": "A",
        "image": "scene.png",
    }
    (data_dir / "first.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (data_dir / "second.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    first_mapping = _base_source_mapping("data/first.jsonl")
    second_mapping = _base_source_mapping("data/second.jsonl")
    response = json.dumps(
        {
            "benchmark": "toy",
            "decision": "ingest",
            "reason": "Both official sources share one schema.",
            "sources": [first_mapping, second_mapping],
        }
    )

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="toy",
        output_root=tmp_path / "out",
    ).build(raw_response=response)

    assert result.decision == "ingest"
    assert len({item["id"] for item in result.items}) == 2
    assert result.items[0]["id"].startswith("toy_data_first_")
    assert result.items[1]["id"].startswith("toy_data_second_")


def test_single_source_ingestion_namespaces_duplicate_ids_by_task(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    Image.new("RGB", (24, 16), "red").save(data_dir / "scene.png")
    rows = [
        {
            "id": "0_0",
            "category": task,
            "question": "Where is the cup? Choices: A. left. B. right.",
            "answer": "A",
            "image": "scene.png",
        }
        for task in ("Dynamic Reasoning", "Perspective Taking")
    ]
    (data_dir / "test.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    mapping = _base_source_mapping("data/test.jsonl")
    mapping["id"] = {"mode": "field", "field": "id", "prefix": "sample"}

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="toy",
        output_root=tmp_path / "out",
    ).build(raw_response=_response(mapping))

    assert result.decision == "ingest"
    assert [item["id"] for item in result.items] == [
        "sample_0_0_dynamic_reasoning",
        "sample_0_0_perspective_taking",
    ]
    assert result.manifest["deterministic_id_repair"] == {
        "strategy": "task_namespace_then_stable_ordinal",
        "duplicate_id_count": 1,
        "affected_sample_count": 2,
        "examples": [
            {
                "original_id": "sample_0_0",
                "task": "dynamic_reasoning",
                "id": "sample_0_0_dynamic_reasoning",
            },
            {
                "original_id": "sample_0_0",
                "task": "perspective_taking",
                "id": "sample_0_0_perspective_taking",
            },
        ],
    }


def test_saved_mapping_inspects_only_referenced_sources(monkeypatch, tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    Image.new("RGB", (24, 16), "red").save(data_dir / "scene.png")
    row = {
        "category": "relation",
        "question": "Where? Choices: A. left. B. right.",
        "answer": "A",
        "image": "scene.png",
    }
    selected = data_dir / "test.jsonl"
    unrelated = data_dir / "large_train.jsonl"
    selected.write_text(json.dumps(row) + "\n", encoding="utf-8")
    unrelated.write_text(json.dumps(row) + "\n", encoding="utf-8")
    import provise.benchmark.ingestion as ingestion_module

    original = ingestion_module.diverse_record_examples
    inspected = []

    def recording_examples(path, *args, **kwargs):
        inspected.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(ingestion_module, "diverse_record_examples", recording_examples)
    mapping = _base_source_mapping("data/test.jsonl")

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="saved_mapping",
        output_root=tmp_path / "out",
    ).build(raw_response=_response(mapping, benchmark="saved_mapping"))

    assert result.decision == "ingest"
    assert inspected == ["test.jsonl"]


def test_inline_mcq_parser_supports_common_option_separators_and_empty_text():
    colon_choices = parse_inline_choices(
        "You have the following options. A: [638 481] B: [643 397] C: [453 562] D: [518 394] "
        "Please answer directly with only the letter."
    )
    parenthesis_choices = parse_inline_choices(
        "Which point is valid? A) [633 567] B) [840 745] C) [715 223] D) [725 237] "
        "Please answer directly with only the letter."
    )
    sparse_choices = parse_inline_choices(
        "Did the robot pick up an orange? Choices: A. B. No. C. Yes. D. "
        "Please answer directly with only the letter."
    )
    visual_label_choices = parse_inline_choices(
        "Which object fits? Choices: A. A. B. B. C. C. D. D. "
        "Please answer directly with only the letter."
    )

    assert [choice["label"] for choice in colon_choices] == ["A", "B", "C", "D"]
    assert colon_choices[0]["text"] == "[638 481]"
    assert [choice["label"] for choice in parenthesis_choices] == ["A", "B", "C", "D"]
    assert parenthesis_choices[-1]["text"] == "[725 237]"
    assert sparse_choices == [
        {"label": "A", "text": ""},
        {"label": "B", "text": "No."},
        {"label": "C", "text": "Yes."},
        {"label": "D", "text": ""},
    ]
    assert visual_label_choices == [
        {"label": "A", "text": "A."},
        {"label": "B", "text": "B."},
        {"label": "C", "text": "C."},
        {"label": "D", "text": "D."},
    ]


def test_parquet_ingestion_extracts_embedded_image_and_target_mask(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    image_bytes = _image_bytes("blue")
    mask_bytes = _image_bytes("white")
    table = pa.Table.from_pylist(
        [
            {
                "category": "context",
                "question": "Pinpoint a location in the free region.",
                "answer": "[(0.5, 0.5)]",
                "img": {"bytes": image_bytes, "path": "scene.jpg"},
                "mask": {"bytes": mask_bytes, "path": "mask.png"},
            }
        ]
    )
    pq.write_table(table, data_dir / "context.parquet")
    mapping = {
        "source": "data/context.parquet",
        "split": "test",
        "id": {"mode": "row_index", "prefix": "context"},
        "task": {"field": "category", "transform": "slug"},
        "question": {"field": "question", "transform": "text"},
        "answer": {"field": "answer", "transform": "text"},
        "answer_type": "points",
        "choices": {"mode": "none"},
        "media": [{"field": "img", "mode": "hf_image", "role": "primary"}],
        "evaluation": {"metric": "point_in_mask", "mask_field": "mask"},
        "metadata_fields": ["category"],
    }
    output_root = tmp_path / "out"

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="robo_toy",
        output_root=output_root,
    ).build(raw_response=_response(mapping, benchmark="robo_toy"))

    assert result.decision == "ingest"
    item = result.items[0]
    assert item["answer_type"] == "points"
    assert item["evaluation"]["metric"] == "point_in_mask"
    assert (output_root / item["evaluation"]["mask_path"]).exists()
    assert (output_root / item["input"]["media"][0]["path"]).exists()


def test_json_ingestion_resolves_target_mask_path_template(tmp_path: Path):
    source_root = tmp_path / "source"
    (source_root / "images").mkdir(parents=True)
    (source_root / "masks").mkdir()
    Image.new("RGB", (24, 16), "blue").save(source_root / "images" / "00.jpg")
    Image.new("L", (24, 16), 255).save(source_root / "masks" / "target.png")
    (source_root / "annotations.json").write_text(
        json.dumps(
            [
                {
                    "category": "object affordance",
                    "question": "Point to the graspable region.",
                    "answer": [[[0.4, 0.5], [0.6, 0.5]]],
                    "img": "00.jpg",
                    "mask": "target.png",
                }
            ]
        ),
        encoding="utf-8",
    )
    mapping = {
        "source": "annotations.json",
        "split": "test",
        "id": {"mode": "row_index", "prefix": "affordance"},
        "task": {"field": "category", "transform": "slug"},
        "question": {"field": "question", "transform": "text"},
        "answer": {"mode": "field", "field": "answer", "transform": "raw"},
        "answer_type": "points",
        "choices": {"mode": "none"},
        "media": [
            {
                "field": "img",
                "mode": "path_template",
                "template": "images/{value}",
                "role": "primary",
            }
        ],
        "evaluation": {
            "metric": "point_in_mask",
            "mask": {
                "field": "mask",
                "mode": "path_template",
                "template": "masks/{value}",
            },
        },
        "metadata_fields": ["category"],
    }
    output_root = tmp_path / "out"

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="roboafford_toy",
        output_root=output_root,
    ).build(raw_response=_response(mapping, benchmark="roboafford_toy"))

    assert result.decision == "ingest"
    item = result.items[0]
    assert (output_root / item["evaluation"]["mask_path"]).is_file()
    assert [entry["role"] for entry in item["input"]["media"]] == ["primary"]
    assert "evaluation.mask" in result.prompt


def test_parquet_ingestion_preserves_official_null_bbox_sentinel(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    image_bytes = _image_bytes("blue")
    table = pa.Table.from_pylist(
        [
            {
                "category": "Spatial",
                "description": "the cup left of the plate",
                "bbox": [1, 2, 10, 12],
                "image": {"bytes": image_bytes, "path": "scene.jpg"},
            },
            {
                "category": "Rejection",
                "description": "an object that is not present",
                "bbox": None,
                "image": {"bytes": image_bytes, "path": "scene.jpg"},
            },
        ]
    )
    pq.write_table(table, data_dir / "grounding.parquet")
    mapping = {
        "source": "data/grounding.parquet",
        "split": "test",
        "id": {"mode": "row_index", "prefix": "grounding"},
        "task": {"field": "category", "transform": "slug"},
        "question": {"field": "description", "transform": "text"},
        "answer": {
            "mode": "field",
            "field": "bbox",
            "transform": "raw",
            "null_value": [0, 0, 0, 0],
        },
        "answer_type": "bbox",
        "choices": {"mode": "none"},
        "media": [{"field": "image", "mode": "hf_image", "role": "primary"}],
        "evaluation": {"metric": "bbox_iou", "metric_config": {"threshold": 0.5}},
        "metadata_fields": ["category"],
    }

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="grounding_toy",
        output_root=tmp_path / "out",
    ).build(raw_response=_response(mapping, benchmark="grounding_toy"))

    assert result.decision == "ingest"
    assert result.items[0]["answer"] == [1, 2, 10, 12]
    assert result.items[1]["answer"] == [0, 0, 0, 0]
    assert result.items[1]["answer_type"] == "bbox"


def test_parquet_ingestion_supports_choices_in_separate_fields(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    table = pa.Table.from_pylist(
        [
            {
                "index": 7,
                "category": "relation",
                "question": "Where is the cup?",
                "answer": "B",
                "A": "left of the plate",
                "B": "right of the plate",
                "C": "behind the plate",
                "image": [_image_bytes("blue"), _image_bytes("green")],
            }
        ]
    )
    pq.write_table(table, data_dir / "separate_choices.parquet")
    mapping = {
        "source": "data/separate_choices.parquet",
        "split": "test",
        "id": {"mode": "field", "field": "index", "prefix": "separate"},
        "task": {"field": "category", "transform": "slug"},
        "question": {"field": "question", "transform": "text"},
        "answer": {"field": "answer", "transform": "text"},
        "answer_type": "choice",
        "choices": {
            "mode": "fields",
            "fields": ["A", "B", "C"],
            "labels": ["A", "B", "C"],
        },
        "media": [{"field": "image", "mode": "embedded_images", "role": "view"}],
        "evaluation": {"metric": "unverified"},
        "metadata_fields": ["category"],
    }

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="separate_choices",
        output_root=tmp_path / "out",
    ).build(raw_response=_response(mapping, benchmark="separate_choices"))

    assert result.decision == "ingest"
    assert result.items[0]["answer"] == "B"
    assert result.items[0]["choices"] == [
        {"label": "A", "text": "left of the plate"},
        {"label": "B", "text": "right of the plate"},
        {"label": "C", "text": "behind the plate"},
    ]


def test_parquet_ingestion_maps_choice_text_answer_to_generated_label(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    table = pa.Table.from_pylist(
        [
            {
                "question": "Which person is taller?",
                "answer": "The right.",
                "option": ["the left", "the right"],
                "image": [_image_bytes("blue")],
            }
        ]
    )
    pq.write_table(table, data_dir / "choice_text_answer.parquet")
    mapping = {
        "source": "data/choice_text_answer.parquet",
        "split": "test",
        "id": {"mode": "row_index", "prefix": "choice-text"},
        "task": {"constant": "size"},
        "question": {"field": "question", "transform": "text"},
        "answer": {"field": "answer", "transform": "raw"},
        "answer_type": "choice",
        "choices": {"mode": "field", "field": "option"},
        "media": [{"field": "image", "mode": "embedded_images", "role": "primary"}],
        "evaluation": {"metric": "accuracy"},
        "metadata_fields": [],
    }

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="choice_text_answer",
        output_root=tmp_path / "out",
    ).build(raw_response=_response(mapping, benchmark="choice_text_answer"))

    assert result.decision == "ingest"
    assert result.items[0]["answer"] == "B"
    assert result.items[0]["choices"] == [
        {"label": "A", "text": "the left"},
        {"label": "B", "text": "the right"},
    ]


def test_ingestion_repairs_scalar_choice_field_when_letter_columns_are_present(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    table = pa.Table.from_pylist(
        [
            {
                "question": "Choose the relation.",
                "answer": "B",
                "A": "left",
                "B": "right",
                "C": "above",
                "image": [_image_bytes("blue")],
            }
        ]
    )
    pq.write_table(table, data_dir / "letter_columns.parquet")
    mapping = {
        "source": "data/letter_columns.parquet",
        "split": "test",
        "id": {"mode": "row_index", "prefix": "letter"},
        "task": {"constant": "relation"},
        "question": {"field": "question", "transform": "text"},
        "answer": {"field": "answer", "transform": "text"},
        "answer_type": "choice",
        "choices": {"mode": "field", "field": "A"},
        "media": [{"field": "image", "mode": "embedded_images", "role": "view"}],
        "evaluation": {"metric": "unverified"},
        "metadata_fields": [],
    }

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="letter_columns",
        output_root=tmp_path / "out",
    ).build(raw_response=_response(mapping, benchmark="letter_columns"))

    assert result.decision == "ingest"
    assert result.mapping["sources"][0]["choices"] == {
        "mode": "fields",
        "fields": ["A", "B", "C"],
        "labels": ["A", "B", "C"],
    }
    assert [choice["label"] for choice in result.items[0]["choices"]] == ["A", "B", "C"]


def test_metric_evidence_prioritizes_task_level_evaluation_config(tmp_path: Path):
    generic = tmp_path / "evaluator.py"
    generic.write_text("# Generic evaluation framework with accuracy metrics.\n", encoding="utf-8")
    task_dir = tmp_path / "code" / "tasks" / "spatial"
    task_dir.mkdir(parents=True)
    task_config = task_dir / "ep.yaml"
    task_config.write_text(
        "doc_to_target: answer\n"
        "process_results: !function cc_utils.process_results\n"
        "metric_list:\n"
        "  - metric: task_score\n"
        "    aggregation: !function cc_utils.aggregate_results\n",
        encoding="utf-8",
    )
    utils = task_dir / "cc_utils.py"
    utils.write_text(
        "def aggregate_results(rows):\n"
        "    overall_acc = sum(row['score'] for row in rows) / len(rows)\n"
        "    return overall_acc * 100\n",
        encoding="utf-8",
    )

    evidence = discover_metric_evidence(tmp_path, limit=8000)

    assert evidence[0]["source"] == "code/tasks/spatial/ep.yaml"
    accuracy_sources = [row for row in evidence if metric_evidence_supports("accuracy", row["excerpt"])]
    assert any(row["source"] == "code/tasks/spatial/cc_utils.py" for row in accuracy_sources)


def test_inventory_separates_data_sources_from_code_artifacts(tmp_path: Path):
    data_dir = tmp_path / "data"
    code_dir = tmp_path / "code"
    data_dir.mkdir()
    code_dir.mkdir()
    (data_dir / "test.jsonl").write_text('{"question":"q"}\n', encoding="utf-8")
    (code_dir / "test_fixture.json").write_text('{"not":"benchmark data"}', encoding="utf-8")
    (code_dir / "evaluator.py").write_text(
        "def calculate_accuracy(correct, total):\n    return correct / total\n",
        encoding="utf-8",
    )

    sources = discover_sources(tmp_path, limit=30)
    evidence = discover_metric_evidence(tmp_path)

    assert [path.relative_to(tmp_path).as_posix() for path in sources] == ["data/test.jsonl"]
    assert any(row["source"] == "code/evaluator.py" for row in evidence)


def test_ingestion_promotes_choice_normalizer_metric_to_verified_accuracy(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    code_dir = source_root / "code" / "tasks" / "choice"
    data_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)
    Image.new("RGB", (24, 16), "navy").save(data_dir / "scene.png")
    (data_dir / "test.jsonl").write_text(
        json.dumps(
            {
                "category": "counting",
                "question": "How many cups are visible?",
                "answer": "B",
                "A": "1",
                "B": "2",
                "C": "3",
                "image": "scene.png",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (code_dir / "evaluator.py").write_text(
        "def evaluate(rows):\n"
        "    hits = [row['prediction'] == row['answer'] for row in rows]\n"
        "    overall_acc = sum(hits) / len(hits)\n"
        "    return overall_acc * 100\n",
        encoding="utf-8",
    )
    mapping = {
        "source": "data/test.jsonl",
        "split": "test",
        "id": {"mode": "row_index", "prefix": "choice"},
        "task": {"field": "category", "transform": "slug"},
        "question": {"field": "question", "transform": "text"},
        "answer": {"mode": "field", "field": "answer", "transform": "text"},
        "answer_type": "choice",
        "choices": {
            "mode": "fields",
            "fields": ["A", "B", "C"],
            "labels": ["A", "B", "C"],
        },
        "media": [{"field": "image", "mode": "path", "role": "primary"}],
        "evaluation": {"metric": "unverified"},
        "metadata_fields": [],
    }

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="choice_normalizer",
        output_root=tmp_path / "out",
    ).build(raw_response=_response(mapping, benchmark="choice_normalizer"))

    evaluation = result.mapping["sources"][0]["evaluation"]
    assert result.decision == "ingest"
    assert evaluation["metric"] == "accuracy"
    assert "metric_provenance" not in evaluation
    assert result.items[0]["evaluation"]["metric"] == "accuracy"
    assert "metric_provenance" not in result.items[0]["evaluation"]


def test_json_ingestion_materializes_a_base64_embedded_image(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    rows = [
        {
            "category": "relation",
            "question": "Where is the cup? Choices: A. left. B. right.",
            "answer": "A",
            "image": base64.b64encode(_image_bytes("purple")).decode("ascii"),
        }
    ]
    (data_dir / "benchmark.json").write_text(json.dumps(rows), encoding="utf-8")
    mapping = _base_source_mapping("data/benchmark.json")
    mapping["media"] = [
        {"field": "image", "mode": "embedded_images", "role": "primary"}
    ]
    output_root = tmp_path / "out"

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="base64_toy",
        output_root=output_root,
    ).build(raw_response=_response(mapping, benchmark="base64_toy"))

    assert result.decision == "ingest"
    assert len(result.items) == 1
    image_path = output_root / result.items[0]["input"]["media"][0]["path"]
    assert image_path.exists()
    with Image.open(image_path) as image:
        assert image.size == (24, 16)


def test_parquet_ingestion_supports_mask_only_evaluation_target(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    table = pa.Table.from_pylist(
        [
            {
                "id": 7,
                "prompt": "Point to the free region.",
                "image": {"bytes": _image_bytes("blue"), "path": "scene.png"},
                "mask": {"bytes": _image_bytes("white"), "path": "mask.png"},
            }
        ]
    )
    pq.write_table(table, data_dir / "target_only.parquet")
    mapping = {
        "source": "data/target_only.parquet",
        "split": "test",
        "id": {"mode": "field", "field": "id", "prefix": "target"},
        "task": {"constant": "placement"},
        "question": {"field": "prompt", "transform": "text"},
        "answer": {"mode": "evaluation_target"},
        "answer_type": "points",
        "choices": {"mode": "none"},
        "media": [{"field": "image", "mode": "hf_image", "role": "primary"}],
        "evaluation": {"metric": "point_in_mask", "mask_field": "mask"},
        "metadata_fields": [],
    }
    output_root = tmp_path / "out"

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="target_only",
        output_root=output_root,
    ).build(raw_response=_response(mapping, benchmark="target_only"))

    assert result.decision == "ingest"
    item = result.items[0]
    assert item["answer"] == {
        "type": "evaluation_target",
        "metric": "point_in_mask",
        "path": item["evaluation"]["mask_path"],
    }
    assert [entry["role"] for entry in item["input"]["media"]] == ["primary"]


def test_tfrecord_ingestion_preserves_multi_image_order(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    first = _image_bytes("red")
    second = _image_bytes("green")
    tfrecord_path = data_dir / "test.tfrecord"
    writer = TFRecordWriter(str(tfrecord_path))
    writer.write(
        {
            "question": (b"Which view matches? Choices: A. first. B. second.", "byte"),
            "answer": (b"B", "byte"),
            "question_type": ([b"Multi-view Reasoning", b"correspondence"], "byte"),
            "image/encoded": ([first, second], "byte"),
            "visual_indices": ([9, 0], "int"),
        }
    )
    writer.close()
    mapping = {
        "source": "data/test.tfrecord",
        "split": "test",
        "id": {"mode": "row_index", "prefix": "erqa"},
        "task": {"field": "question_type", "transform": "first_slug"},
        "question": {"field": "question", "transform": "text"},
        "answer": {"field": "answer", "transform": "text"},
        "answer_type": "choice",
        "choices": {"mode": "inline_mcq"},
        "media": [
            {
                "field": "image/encoded",
                "mode": "embedded_images",
                "role": "view",
                "order_field": "visual_indices",
            }
        ],
        "evaluation": {"metric": "accuracy"},
        "metadata_fields": ["question_type", "visual_indices"],
    }
    output_root = tmp_path / "out"

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="erqa_toy",
        output_root=output_root,
    ).build(raw_response=_response(mapping, benchmark="erqa_toy"))

    assert result.decision == "ingest"
    item = result.items[0]
    assert item["task"] == "multi_view_reasoning"
    assert len(item["input"]["media"]) == 2
    with Image.open(output_root / item["input"]["media"][0]["path"]) as image:
        assert image.convert("RGB").getpixel((0, 0))[1] > image.convert("RGB").getpixel((0, 0))[0]


def test_ingestion_rejects_agent_mapping_with_unknown_field(tmp_path: Path):
    source_root = tmp_path / "source"
    data_dir = source_root / "data"
    data_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(data_dir / "scene.png")
    (data_dir / "test.jsonl").write_text(
        json.dumps({"category": "test", "question": "Q", "answer": "A", "image": "scene.png"}) + "\n",
        encoding="utf-8",
    )
    mapping = _base_source_mapping("data/test.jsonl")
    mapping["question"]["field"] = "hallucinated_question"

    result = AgenticBenchmarkIngestor(
        source_root=source_root,
        benchmark_name="toy",
        output_root=tmp_path / "out",
    ).build(raw_response=_response(mapping))

    assert result.decision == "unsupported"
    assert result.items == []
    assert "hallucinated_question" in " ".join(result.manifest["warnings"])


def test_heterogeneous_task_is_partitioned_by_answer_and_metric_contract():
    common = {
        "benchmark": "toy",
        "task": "spatial_reasoning",
        "image_path": "scene.png",
        "metadata": {},
    }
    choice = {
        **common,
        "id": "choice",
        "question": "Is the cup left of the plate?",
        "answer": "yes",
        "answer_type": "choice",
        "choices": [{"label": "yes", "text": "yes"}, {"label": "no", "text": "no"}],
        "evaluation": {"metric": "accuracy"},
    }
    point = {
        **common,
        "id": "point",
        "question": "Point to the free region.",
        "answer": [0.5, 0.5],
        "answer_type": "points",
        "choices": [],
        "evaluation": {"metric": "point_in_mask", "mask_path": "mask.png"},
    }

    partitioned, manifest = partition_heterogeneous_tasks([choice, point])

    assert len({item["task"] for item in partitioned}) == 2
    assert all(item["task"].startswith("spatial_reasoning__") for item in partitioned)
    assert all(item["metadata"]["original_task"] == "spatial_reasoning" for item in partitioned)
    assert manifest[0]["partition_count"] == 2
