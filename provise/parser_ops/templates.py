from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


_CYAN_POINT_MARKER_PIPELINE: Dict[str, Any] = {
    "steps": [
        {"id": "image", "op": "load_generated"},
        {
            "id": "mask",
            "op": "hsv_color_mask",
            "inputs": ["image"],
            "params": {
                "lower": {"$config": "cyan_lower", "default": [80, 70, 70]},
                "upper": {"$config": "cyan_upper", "default": [100, 255, 255]},
            },
        },
        {
            "id": "components",
            "op": "connected_components",
            "inputs": ["mask"],
        },
        {
            "id": "candidates",
            "op": "filter_components",
            "inputs": ["components"],
            "params": {
                "min_area": {"$config": "min_pixels", "default": 10},
                "min_fill_ratio": {"$config": "min_fill_ratio", "default": 0.2},
            },
        },
        {
            "id": "marker",
            "op": "select_largest_compact",
            "inputs": ["candidates"],
            "params": {
                "ambiguity_ratio": {"$config": "ambiguity_ratio", "default": 0.85},
                "empty_error": "cyan point marker not found",
                "ambiguous_error": "multiple similarly prominent cyan point markers found",
            },
        },
        {
            "id": "centroid",
            "op": "component_centroid",
            "inputs": ["marker"],
        },
        {
            "id": "point",
            "op": "normalize_point",
            "inputs": ["centroid", "image"],
        },
    ],
    "output": "point",
}


def cyan_point_marker_pipeline() -> Dict[str, Any]:
    return deepcopy(_CYAN_POINT_MARKER_PIPELINE)


def cyan_point_marker_edit_pipeline() -> Dict[str, Any]:
    """Return a source-aware marker pipeline that suppresses pre-existing cyan."""
    pipeline = cyan_point_marker_pipeline()
    steps = pipeline["steps"]
    generated_mask = next(step for step in steps if step["id"] == "mask")
    generated_mask["id"] = "generated_mask"
    mask_index = steps.index(generated_mask)
    steps[mask_index + 1 : mask_index + 1] = [
        {"id": "source", "op": "load_source", "params": {"index": 0}},
        {
            "id": "source_mask",
            "op": "hsv_color_mask",
            "inputs": ["source"],
            "params": deepcopy(generated_mask["params"]),
        },
        {
            "id": "source_guard",
            "op": "morphology",
            "inputs": ["source_mask"],
            "params": {"operation": "dilate", "kernel_size": 7, "iterations": 1},
        },
        {
            "id": "mask",
            "op": "mask_subtract",
            "inputs": ["generated_mask", "source_guard"],
        },
    ]
    return pipeline


def green_instance_marker_count_pipeline() -> Dict[str, Any]:
    return {
        "steps": [
            {"id": "image", "op": "load_generated"},
            {"id": "source", "op": "load_source", "params": {"index": 0}},
            {
                "id": "generated_mask",
                "op": "hsv_color_mask",
                "inputs": ["image"],
                "params": {
                    "lower": {"$config": "green_lower", "default": [35, 80, 80]},
                    "upper": {"$config": "green_upper", "default": [90, 255, 255]},
                },
            },
            {
                "id": "source_mask",
                "op": "hsv_color_mask",
                "inputs": ["source"],
                "params": {
                    "lower": {"$config": "green_lower", "default": [35, 80, 80]},
                    "upper": {"$config": "green_upper", "default": [90, 255, 255]},
                },
            },
            {
                "id": "source_guard",
                "op": "morphology",
                "inputs": ["source_mask"],
                "params": {"operation": "dilate", "kernel_size": 7, "iterations": 1},
            },
            {
                "id": "novel_mask",
                "op": "mask_subtract",
                "inputs": ["generated_mask", "source_guard"],
            },
            {
                "id": "opened_mask",
                "op": "morphology",
                "inputs": ["novel_mask"],
                "params": {"operation": "open", "kernel_size": 3, "iterations": 1},
            },
            {
                "id": "clean_mask",
                "op": "morphology",
                "inputs": ["opened_mask"],
                "params": {"operation": "close", "kernel_size": 3, "iterations": 1},
            },
            {
                "id": "components",
                "op": "connected_components",
                "inputs": ["clean_mask"],
            },
            {
                "id": "candidates",
                "op": "filter_components",
                "inputs": ["components"],
                "params": {
                    "min_area": {"$config": "min_area", "default": 30},
                    "min_fill_ratio": {"$config": "min_fill_ratio", "default": 0.2},
                    "min_compactness": {"$config": "min_compactness", "default": 0.4},
                },
            },
            {
                "id": "count",
                "op": "count_components",
                "inputs": ["candidates"],
                "params": {"min_relative_area": 0.25, "max_relative_area": 3.0},
            },
        ],
        "output": "count",
    }


def green_choice_count_board_pipeline() -> Dict[str, Any]:
    """Count uniform evidence markers on a synthesized board and map to a numeric choice."""
    return {
        "steps": [
            {"id": "image", "op": "load_generated"},
            {
                "id": "marker_mask",
                "op": "hsv_color_mask",
                "inputs": ["image"],
                "params": {
                    "lower": {"$config": "green_lower", "default": [35, 100, 100]},
                    "upper": {"$config": "green_upper", "default": [90, 255, 255]},
                },
            },
            {
                "id": "opened_mask",
                "op": "morphology",
                "inputs": ["marker_mask"],
                "params": {"operation": "open", "kernel_size": 5, "iterations": 1},
            },
            {
                "id": "clean_mask",
                "op": "morphology",
                "inputs": ["opened_mask"],
                "params": {"operation": "close", "kernel_size": 5, "iterations": 1},
            },
            {
                "id": "components",
                "op": "connected_components",
                "inputs": ["clean_mask"],
            },
            {
                "id": "markers",
                "op": "filter_components",
                "inputs": ["components"],
                "params": {
                    "min_area": {"$config": "min_area", "default": 80},
                    "min_fill_ratio": {"$config": "min_fill_ratio", "default": 0.45},
                    "min_compactness": {"$config": "min_compactness", "default": 0.55},
                },
            },
            {
                "id": "count",
                "op": "count_components",
                "inputs": ["markers"],
                "params": {"min_relative_area": 0.35, "max_relative_area": 2.8},
            },
            {
                "id": "choice",
                "op": "map_count_to_choice",
                "inputs": ["count"],
            },
        ],
        "output": "choice",
    }


def dual_anchor_relation_pipeline(*, mode: str = "dominant_axis", map_to_choices: bool = True) -> Dict[str, Any]:
    steps = [
        {"id": "generated", "op": "load_generated"},
        {"id": "source", "op": "load_source", "params": {"index": 0}},
        {
            "id": "aligned",
            "op": "align_to_reference",
            "inputs": ["generated", "source"],
            "params": {"motion": "euclidean"},
        },
    ]
    steps.extend(_source_aware_role_component("subject", "aligned", "source", [140, 70, 70], [179, 255, 255]))
    steps.extend(_source_aware_role_component("reference", "aligned", "source", [80, 70, 70], [100, 255, 255]))
    steps.append(
        {
            "id": "relation",
            "op": "component_relation",
            "inputs": ["subject_component", "reference_component"],
            "params": {
                "mode": {"$config": "relation_mode", "default": mode},
                "overlap_threshold": {"$config": "overlap_threshold", "default": 0.25},
                "near_threshold": {"$config": "near_threshold", "default": 0.2},
            },
        }
    )
    output = "relation"
    if map_to_choices:
        steps.append(
            {
                "id": "choice",
                "op": "map_relation_to_choice",
                "inputs": ["relation"],
            }
        )
        output = "choice"
    return {"steps": steps, "output": output}


def relation_zone_boolean_pipeline() -> Dict[str, Any]:
    steps = [
        {"id": "generated", "op": "load_generated"},
        {"id": "source", "op": "load_source", "params": {"index": 0}},
        {
            "id": "aligned",
            "op": "align_to_reference",
            "inputs": ["generated", "source"],
            "params": {"motion": "euclidean"},
        },
    ]
    steps.extend(
        _source_aware_role_component(
            "subject", "aligned", "source", [140, 80, 80], [179, 255, 255], min_area=40
        )
    )
    steps.extend(
        _source_aware_role_component(
            "reference", "aligned", "source", [80, 80, 80], [100, 255, 255], min_area=40
        )
    )
    steps.extend(
        [
            {
                "id": "target_region_generated_mask",
                "op": "hsv_color_mask",
                "inputs": ["aligned"],
                "params": {"lower": [35, 50, 50], "upper": [90, 255, 255]},
            },
            {
                "id": "target_region_source_mask",
                "op": "hsv_color_mask",
                "inputs": ["source"],
                "params": {"lower": [35, 50, 50], "upper": [90, 255, 255]},
            },
            {
                "id": "target_region_source_guard",
                "op": "morphology",
                "inputs": ["target_region_source_mask"],
                "params": {"operation": "dilate", "kernel_size": 5, "iterations": 1},
            },
            {
                "id": "target_region_novel_mask",
                "op": "mask_subtract",
                "inputs": ["target_region_generated_mask", "target_region_source_guard"],
            },
            {
                "id": "target_region_mask",
                "op": "morphology",
                "inputs": ["target_region_novel_mask"],
                "params": {"operation": "close", "kernel_size": 7, "iterations": 1},
            },
            {
                "id": "subject_point",
                "op": "component_centroid",
                "inputs": ["subject_component"],
            },
            {
                "id": "relation_holds",
                "op": "point_in_mask",
                "inputs": ["subject_point", "target_region_mask"],
                "params": {
                    "radius": {"$config": "membership_radius", "default": 24},
                    "min_fraction": {
                        "$config": "membership_min_fraction",
                        "default": 0.03,
                    },
                    "min_mask_pixels": {
                        "$config": "target_region_min_pixels",
                        "default": 200,
                    },
                },
            },
        ]
    )
    return {"steps": steps, "output": "relation_holds"}


def grounded_dimension_pipeline(*, anchor_count: int = 2) -> Dict[str, Any]:
    if anchor_count not in {1, 2}:
        raise ValueError("grounded_dimension anchor_count must be 1 or 2")
    steps = [
        {"id": "generated", "op": "load_generated"},
        {"id": "source", "op": "load_source", "params": {"index": 0}},
        {
            "id": "aligned",
            "op": "align_to_reference",
            "inputs": ["generated", "source"],
            "params": {"motion": "euclidean"},
        },
    ]
    steps.extend(_source_aware_role_component("subject", "aligned", "source", [140, 70, 70], [179, 255, 255]))
    if anchor_count == 2:
        steps.extend(_source_aware_role_component("reference", "aligned", "source", [80, 70, 70], [100, 255, 255]))
    steps.extend(_source_aware_role_component("dimension", "aligned", "source", [18, 80, 100], [40, 255, 255], min_area=20))
    steps.extend(
        [
            {
                "id": "label_crop",
                "op": "crop_dimension_label",
                "inputs": ["aligned", "dimension_component"],
                "params": {
                    "minor_pad": 4.0,
                    "major_pad": 0.75,
                    "min_minor": 512,
                    "min_major": 192,
                },
            },
            {
                "id": "measurement",
                "op": "ocr_measurement",
                "inputs": ["label_crop"],
                "params": {
                    "min_confidence": {"$config": "ocr_min_confidence", "default": 0.35},
                    "unit_hint": {"$config": "unit", "default": "cm"},
                },
            },
        ]
    )
    if anchor_count == 2:
        steps.append(
            {
                "id": "grounded_measurement",
                "op": "validate_dimension_between",
                "inputs": [
                    "measurement",
                    "subject_component",
                    "reference_component",
                    "dimension_component",
                ],
                "params": {"min_line_elongation": 2.0, "min_anchor_area": 30},
            }
        )
    else:
        steps.append(
            {
                "id": "grounded_measurement",
                "op": "validate_dimension_extent",
                "inputs": ["measurement", "subject_component", "dimension_component"],
                "params": {"min_line_elongation": 2.0, "min_anchor_area": 30},
            }
        )
    return {"steps": steps, "output": "grounded_measurement"}


def semantic_choice_clip_pipeline() -> Dict[str, Any]:
    return {
        "steps": [
            {"id": "generated", "op": "load_generated"},
            {
                "id": "choice",
                "op": "clip_match_choices",
                "inputs": ["generated"],
                "params": {
                    "model": {
                        "$config": "clip_model",
                        "default": "",
                    },
                    "min_score": {"$config": "clip_min_score", "default": 0.0},
                    "min_margin": {"$config": "clip_min_margin", "default": 0.015},
                },
            },
        ],
        "output": "choice",
    }


def marked_object_choice_pipeline() -> Dict[str, Any]:
    steps = [
        {"id": "generated", "op": "load_generated"},
        {"id": "source", "op": "load_source", "params": {"index": 0}},
        {
            "id": "aligned",
            "op": "align_to_reference",
            "inputs": ["generated", "source"],
            "params": {"motion": "euclidean"},
        },
    ]
    steps.extend(
        _source_aware_role_component(
            "selected_object", "aligned", "source", [140, 70, 70], [179, 255, 255]
        )
    )
    steps.extend(
        [
            {
                "id": "object_crop",
                "op": "crop_around_component",
                "inputs": ["source", "selected_object_component"],
                "params": {"pad_x": 0.15, "pad_y": 0.15, "min_width": 96, "min_height": 96},
            },
            {
                "id": "choice",
                "op": "clip_match_choices",
                "inputs": ["object_crop"],
                "params": {
                    "model": {"$config": "clip_model", "default": ""},
                    "min_score": {"$config": "clip_min_score", "default": 0.0},
                    "min_margin": {"$config": "clip_min_margin", "default": 0.0},
                },
            },
        ]
    )
    return {"steps": steps, "output": "choice"}


def _source_aware_role_component(
    role: str,
    generated_id: str,
    source_id: str,
    lower: list[int],
    upper: list[int],
    *,
    min_area: int = 30,
) -> list[Dict[str, Any]]:
    return [
        {
            "id": f"{role}_generated_mask",
            "op": "hsv_color_mask",
            "inputs": [generated_id],
            "params": {"lower": lower, "upper": upper},
        },
        {
            "id": f"{role}_source_mask",
            "op": "hsv_color_mask",
            "inputs": [source_id],
            "params": {"lower": lower, "upper": upper},
        },
        {
            "id": f"{role}_source_guard",
            "op": "morphology",
            "inputs": [f"{role}_source_mask"],
            "params": {"operation": "dilate", "kernel_size": 5, "iterations": 1},
        },
        {
            "id": f"{role}_novel_mask",
            "op": "mask_subtract",
            "inputs": [f"{role}_generated_mask", f"{role}_source_guard"],
        },
        {
            "id": f"{role}_clean_mask",
            "op": "morphology",
            "inputs": [f"{role}_novel_mask"],
            "params": {"operation": "close", "kernel_size": 5, "iterations": 1},
        },
        {
            "id": f"{role}_components",
            "op": "connected_components",
            "inputs": [f"{role}_clean_mask"],
        },
        {
            "id": f"{role}_candidates",
            "op": "filter_components",
            "inputs": [f"{role}_components"],
            "params": {"min_area": min_area},
        },
        {
            "id": f"{role}_component",
            "op": "select_largest",
            "inputs": [f"{role}_candidates"],
            "params": {"ambiguity_ratio": 1.0},
        },
    ]
