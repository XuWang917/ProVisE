from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from ..reporting import concise_output_enabled
from .core import ParserContext, ParserOpError, ParserOpSpec, ParserRegistry, ParserValue


def create_default_registry() -> ParserRegistry:
    registry = ParserRegistry()
    registry.register(
        ParserOpSpec(
            name="load_generated",
            input_kinds=(),
            output_kind="image_bgr",
            function=_load_generated,
            description="Load the generated image as a BGR array.",
        )
    )
    registry.register(
        ParserOpSpec(
            name="load_source",
            input_kinds=(),
            output_kind="image_bgr",
            function=_load_source,
            description="Load one original benchmark input image as a BGR array.",
            allowed_params=frozenset({"index"}),
            validate_params=_validate_load_source_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="hsv_color_mask",
            input_kinds=("image_bgr",),
            output_kind="binary_mask",
            function=_hsv_color_mask,
            description="Threshold a BGR image with explicit HSV lower and upper bounds.",
            allowed_params=frozenset({"lower", "upper"}),
            required_params=frozenset({"lower", "upper"}),
            validate_params=_validate_hsv_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="connected_components",
            input_kinds=("binary_mask",),
            output_kind="components",
            function=_connected_components,
            description="Extract foreground connected components and geometric statistics.",
            allowed_params=frozenset({"connectivity"}),
            validate_params=_validate_connected_components_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="morphology",
            input_kinds=("binary_mask",),
            output_kind="binary_mask",
            function=_morphology,
            description="Apply a whitelisted OpenCV morphology operation to a mask.",
            allowed_params=frozenset({"operation", "kernel_size", "iterations"}),
            required_params=frozenset({"operation"}),
            validate_params=_validate_morphology_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="mask_subtract",
            input_kinds=("binary_mask", "binary_mask"),
            output_kind="binary_mask",
            function=_mask_subtract,
            description="Remove pixels present in a reference mask from a candidate mask.",
        )
    )
    registry.register(
        ParserOpSpec(
            name="filter_components",
            input_kinds=("components",),
            output_kind="components",
            function=_filter_components,
            description="Filter components by area, fill ratio, and compactness.",
            allowed_params=frozenset(
                {"min_area", "max_area", "min_fill_ratio", "min_compactness"}
            ),
            validate_params=_validate_filter_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="count_components",
            input_kinds=("components",),
            output_kind="integer_count",
            function=_count_components,
            description="Count similarly sized components with median-relative area bounds.",
            allowed_params=frozenset({"min_relative_area", "max_relative_area"}),
            validate_params=_validate_count_components_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="map_count_to_choice",
            input_kinds=("integer_count",),
            output_kind="choice_label",
            function=_map_count_to_choice,
            description=(
                "Map a deterministically recovered count to the unique numeric benchmark choice."
            ),
            allowed_params=frozenset({"ambiguity_error"}),
        )
    )
    registry.register(
        ParserOpSpec(
            name="select_largest_compact",
            input_kinds=("components",),
            output_kind="component",
            function=_select_largest_compact,
            description="Select the strongest compact component and reject ambiguous ties.",
            allowed_params=frozenset({"ambiguity_ratio", "empty_error", "ambiguous_error"}),
            validate_params=_validate_select_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="component_centroid",
            input_kinds=("component",),
            output_kind="point_pixels",
            function=_component_centroid,
            description="Read a component centroid in generated-image pixel coordinates.",
        )
    )
    registry.register(
        ParserOpSpec(
            name="normalize_point",
            input_kinds=("point_pixels", "image_bgr"),
            output_kind="normalized_point",
            function=_normalize_point,
            description="Normalize a pixel point to [0, 1] image coordinates.",
        )
    )
    registry.register(
        ParserOpSpec(
            name="point_in_mask",
            input_kinds=("point_pixels", "binary_mask"),
            output_kind="boolean",
            function=_point_in_mask,
            description=(
                "Test whether a grounded point lies in or immediately beside a visible "
                "target region."
            ),
            allowed_params=frozenset({"radius", "min_fraction", "min_mask_pixels"}),
            validate_params=_validate_point_in_mask_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="align_to_reference",
            input_kinds=("image_bgr", "image_bgr"),
            output_kind="image_bgr",
            function=_align_to_reference,
            description="Align an edited image to a source image with deterministic ECC registration.",
            allowed_params=frozenset({"motion", "iterations", "epsilon"}),
            validate_params=_validate_alignment_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="image_difference_mask",
            input_kinds=("image_bgr", "image_bgr"),
            output_kind="binary_mask",
            function=_image_difference_mask,
            description="Detect changed pixels between aligned generated and source images.",
            allowed_params=frozenset({"threshold", "blur_kernel"}),
            validate_params=_validate_difference_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="mask_union",
            input_kinds=("binary_mask", "binary_mask"),
            output_kind="binary_mask",
            function=_mask_union,
            description="Compute the union of two binary masks.",
        )
    )
    registry.register(
        ParserOpSpec(
            name="mask_intersection",
            input_kinds=("binary_mask", "binary_mask"),
            output_kind="binary_mask",
            function=_mask_intersection,
            description="Compute the intersection of two binary masks.",
        )
    )
    registry.register(
        ParserOpSpec(
            name="select_largest",
            input_kinds=("components",),
            output_kind="component",
            function=_select_largest,
            description="Select the largest component with an optional ambiguity check.",
            allowed_params=frozenset({"ambiguity_ratio", "empty_error", "ambiguous_error"}),
            validate_params=_validate_select_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="component_bbox",
            input_kinds=("component",),
            output_kind="pixel_bbox",
            function=_component_bbox,
            description="Convert one component to an [x1,y1,x2,y2] pixel box.",
        )
    )
    registry.register(
        ParserOpSpec(
            name="crop_around_component",
            input_kinds=("image_bgr", "component"),
            output_kind="image_bgr",
            function=_crop_around_component,
            description="Crop an image around a component with relative horizontal and vertical padding.",
            allowed_params=frozenset({"pad_x", "pad_y", "min_width", "min_height"}),
            validate_params=_validate_crop_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="crop_dimension_label",
            input_kinds=("image_bgr", "component"),
            output_kind="image_bgr",
            function=_crop_dimension_label,
            description=(
                "Crop around a dimension line with orientation-aware space for an adjacent label."
            ),
            allowed_params=frozenset(
                {"minor_pad", "major_pad", "min_minor", "min_major"}
            ),
            validate_params=_validate_dimension_crop_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="component_relation",
            input_kinds=("component", "component"),
            output_kind="relation_label",
            function=_component_relation,
            description="Recover a 2D spatial relation from two role-colored components.",
            allowed_params=frozenset({"mode", "overlap_threshold", "near_threshold"}),
            validate_params=_validate_relation_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="map_relation_to_choice",
            input_kinds=("relation_label",),
            output_kind="choice_label",
            function=_map_relation_to_choice,
            description="Map a recovered relation to the semantically matching benchmark choice.",
            allowed_params=frozenset({"ambiguity_error"}),
        )
    )
    registry.register(
        ParserOpSpec(
            name="ocr_measurement",
            input_kinds=("image_bgr",),
            output_kind="scalar_measurement",
            function=_ocr_measurement,
            description="Read a scalar and physical unit with a fixed local OCR backend.",
            allowed_params=frozenset({"min_confidence", "unit_hint", "numeric_pattern"}),
            validate_params=_validate_ocr_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="validate_dimension_between",
            input_kinds=("scalar_measurement", "component", "component", "component"),
            output_kind="scalar_measurement",
            function=_validate_dimension_between,
            description="Require two object anchors and one elongated dimension-line component.",
            allowed_params=frozenset({"min_line_elongation", "min_anchor_area"}),
            validate_params=_validate_dimension_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="validate_dimension_extent",
            input_kinds=("scalar_measurement", "component", "component"),
            output_kind="scalar_measurement",
            function=_validate_dimension_extent,
            description="Require one object outline and one elongated dimension-line component.",
            allowed_params=frozenset({"min_line_elongation", "min_anchor_area"}),
            validate_params=_validate_dimension_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="clip_match_choices",
            input_kinds=("image_bgr",),
            output_kind="choice_label",
            function=_clip_match_choices,
            description="Match generated visual evidence to textual choices with a fixed CLIP model.",
            allowed_params=frozenset({"model", "min_score", "min_margin"}),
            validate_params=_validate_clip_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="components_centroids",
            input_kinds=("components",),
            output_kind="points_pixels",
            function=_components_centroids,
            description="Read all component centroids as a pixel-coordinate point set.",
        )
    )
    registry.register(
        ParserOpSpec(
            name="normalize_points",
            input_kinds=("points_pixels", "image_bgr"),
            output_kind="normalized_points",
            function=_normalize_points,
            description="Normalize a pixel-coordinate point set to [0,1] image coordinates.",
        )
    )
    registry.register(
        ParserOpSpec(
            name="normalize_bbox",
            input_kinds=("pixel_bbox", "image_bgr"),
            output_kind="normalized_bbox",
            function=_normalize_bbox,
            description="Normalize a pixel bounding box to [0,1] image coordinates.",
        )
    )
    registry.register(
        ParserOpSpec(
            name="mask_bbox",
            input_kinds=("binary_mask",),
            output_kind="pixel_bbox",
            function=_mask_bbox,
            description="Recover the tight pixel bounding box of a non-empty mask.",
        )
    )
    registry.register(
        ParserOpSpec(
            name="skeletonize_mask",
            input_kinds=("binary_mask",),
            output_kind="binary_mask",
            function=_skeletonize_mask,
            description="Reduce a path mask to a deterministic one-pixel morphology skeleton.",
            allowed_params=frozenset({"max_iterations"}),
            validate_params=_validate_skeleton_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="mask_endpoints",
            input_kinds=("binary_mask",),
            output_kind="points_pixels",
            function=_mask_endpoints,
            description="Recover the farthest endpoint pair from a skeleton mask.",
        )
    )
    registry.register(
        ParserOpSpec(
            name="normalize_polyline",
            input_kinds=("points_pixels", "image_bgr"),
            output_kind="normalized_polyline",
            function=_normalize_polyline,
            description="Normalize an ordered or endpoint polyline to [0,1] coordinates.",
        )
    )
    registry.register(
        ParserOpSpec(
            name="hough_lines",
            input_kinds=("binary_mask",),
            output_kind="lines_pixels",
            function=_hough_lines,
            description="Detect line segments in a binary mask with probabilistic Hough transform.",
            allowed_params=frozenset({"threshold", "min_line_length", "max_line_gap"}),
            validate_params=_validate_hough_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="select_longest_line",
            input_kinds=("lines_pixels",),
            output_kind="line_pixels",
            function=_select_longest_line,
            description="Select the longest detected line segment and reject an empty set.",
        )
    )
    registry.register(
        ParserOpSpec(
            name="line_endpoints",
            input_kinds=("line_pixels",),
            output_kind="points_pixels",
            function=_line_endpoints,
            description="Read the two endpoints of one line segment.",
        )
    )
    registry.register(
        ParserOpSpec(
            name="line_angle",
            input_kinds=("line_pixels",),
            output_kind="scalar",
            function=_line_angle,
            description="Read one line orientation in degrees in [-180,180].",
        )
    )
    registry.register(
        ParserOpSpec(
            name="mask_area_ratio",
            input_kinds=("binary_mask",),
            output_kind="scalar",
            function=_mask_area_ratio,
            description="Measure foreground area as a fraction of mask pixels.",
        )
    )
    registry.register(
        ParserOpSpec(
            name="color_presence",
            input_kinds=("binary_mask",),
            output_kind="boolean",
            function=_color_presence,
            description="Convert a color mask to a boolean using an explicit pixel threshold.",
            allowed_params=frozenset({"min_pixels"}),
            validate_params=_validate_presence_params,
        )
    )
    registry.register(
        ParserOpSpec(
            name="clip_match_source_images",
            input_kinds=("image_bgr",),
            output_kind="choice_label",
            function=_clip_match_source_images,
            description="Match a generated state to ordered source candidate images with fixed CLIP.",
            allowed_params=frozenset(
                {"model", "candidate_start_index", "min_score", "min_margin"}
            ),
            validate_params=_validate_clip_source_params,
        )
    )
    return registry


def _load_generated(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    image = cv2.imread(context.generated_path)
    if image is None:
        raise ParserOpError(
            "generated image unreadable",
            code="image_unreadable",
            diagnostics={"path": context.generated_path},
        )
    height, width = image.shape[:2]
    return ParserValue(
        "image_bgr",
        image,
        {"width": int(width), "height": int(height), "channels": int(image.shape[2])},
    )


def _load_source(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    index = int(params.get("index", 0))
    if index >= len(context.source_paths):
        raise ParserOpError(
            f"source image index {index} is unavailable",
            code="source_unavailable",
            diagnostics={"source_count": len(context.source_paths), "requested_index": index},
        )
    path = context.source_paths[index]
    image = cv2.imread(path)
    if image is None:
        raise ParserOpError(
            "source image unreadable",
            code="source_unreadable",
            diagnostics={"path": path, "source_index": index},
        )
    height, width = image.shape[:2]
    return ParserValue(
        "image_bgr",
        image,
        {
            "path": path,
            "source_index": index,
            "width": int(width),
            "height": int(height),
            "channels": int(image.shape[2]),
        },
    )


def _hsv_color_mask(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    image = np.asarray(inputs[0].value)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.asarray(params["lower"], dtype=np.uint8)
    upper = np.asarray(params["upper"], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return ParserValue(
        "binary_mask",
        mask,
        {
            "pixel_count": int(np.count_nonzero(mask)),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        },
    )


def _morphology(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    mask = np.asarray(inputs[0].value, dtype=np.uint8)
    operation = str(params["operation"]).strip().lower()
    kernel_size = int(params.get("kernel_size", 3))
    iterations = int(params.get("iterations", 1))
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    if operation == "dilate":
        output = cv2.dilate(mask, kernel, iterations=iterations)
    elif operation == "erode":
        output = cv2.erode(mask, kernel, iterations=iterations)
    else:
        op_code = cv2.MORPH_OPEN if operation == "open" else cv2.MORPH_CLOSE
        output = cv2.morphologyEx(mask, op_code, kernel, iterations=iterations)
    return ParserValue(
        "binary_mask",
        output,
        {
            "operation": operation,
            "kernel_size": kernel_size,
            "iterations": iterations,
            "input_pixel_count": int(np.count_nonzero(mask)),
            "pixel_count": int(np.count_nonzero(output)),
        },
    )


def _mask_subtract(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    candidate = np.asarray(inputs[0].value, dtype=np.uint8)
    reference = np.asarray(inputs[1].value, dtype=np.uint8)
    if candidate.shape != reference.shape:
        raise ParserOpError(
            f"mask shapes do not match: {candidate.shape} vs {reference.shape}",
            code="mask_shape_mismatch",
        )
    output = cv2.bitwise_and(candidate, cv2.bitwise_not(reference))
    return ParserValue(
        "binary_mask",
        output,
        {
            "candidate_pixel_count": int(np.count_nonzero(candidate)),
            "reference_pixel_count": int(np.count_nonzero(reference)),
            "pixel_count": int(np.count_nonzero(output)),
        },
    )


def _connected_components(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    mask = np.asarray(inputs[0].value, dtype=np.uint8)
    connectivity = int(params.get("connectivity", 8))
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
    )
    components = []
    for component_index in range(1, component_count):
        x, y, width, height, area = [int(value) for value in stats[component_index]]
        fill_ratio = float(area / max(1, width * height))
        compactness = float(min(width, height) / max(1, max(width, height)))
        components.append(
            {
                "index": component_index,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "area": area,
                "centroid": [
                    float(centroids[component_index][0]),
                    float(centroids[component_index][1]),
                ],
                "fill_ratio": fill_ratio,
                "compactness": compactness,
                "selection_score": float(area * fill_ratio * compactness),
            }
        )
    return ParserValue(
        "components",
        components,
        {"component_count": len(components), "connectivity": connectivity},
    )


def _filter_components(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    components = list(inputs[0].value)
    min_area = float(params.get("min_area", 0))
    max_area = float(params.get("max_area", float("inf")))
    min_fill_ratio = float(params.get("min_fill_ratio", 0))
    min_compactness = float(params.get("min_compactness", 0))
    filtered = [
        component
        for component in components
        if min_area <= float(component["area"]) <= max_area
        and float(component["fill_ratio"]) >= min_fill_ratio
        and float(component["compactness"]) >= min_compactness
    ]
    return ParserValue(
        "components",
        filtered,
        {
            "input_component_count": len(components),
            "component_count": len(filtered),
            "rejected_component_count": len(components) - len(filtered),
        },
    )


def _count_components(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    components = list(inputs[0].value)
    if not components:
        return ParserValue(
            "integer_count",
            0,
            {"input_component_count": 0, "component_count": 0, "median_area": 0.0},
        )
    areas = [float(component["area"]) for component in components]
    median_area = float(np.median(areas))
    min_relative_area = float(params.get("min_relative_area", 0.25))
    max_relative_area = float(params.get("max_relative_area", 3.0))
    kept = [
        component
        for component in components
        if min_relative_area * median_area
        <= float(component["area"])
        <= max_relative_area * median_area
    ]
    return ParserValue(
        "integer_count",
        len(kept),
        {
            "input_component_count": len(components),
            "component_count": len(kept),
            "median_area": median_area,
            "min_relative_area": min_relative_area,
            "max_relative_area": max_relative_area,
        },
    )


def _select_largest_compact(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    candidates = sorted(
        list(inputs[0].value),
        key=lambda row: float(row["selection_score"]),
        reverse=True,
    )
    if not candidates:
        raise ParserOpError(
            str(params.get("empty_error") or "no matching component found"),
            code="no_component",
            diagnostics={"component_count": 0},
        )

    candidate_scores = [round(float(row["selection_score"]), 3) for row in candidates[:3]]
    ambiguity_ratio = float(params.get("ambiguity_ratio", 0.85))
    if len(candidates) > 1 and float(candidates[1]["selection_score"]) >= (
        float(candidates[0]["selection_score"]) * ambiguity_ratio
    ):
        raise ParserOpError(
            str(params.get("ambiguous_error") or "multiple similarly prominent components found"),
            code="ambiguous_components",
            diagnostics={
                "component_count": len(candidates),
                "candidate_scores": candidate_scores,
                "ambiguity_ratio": ambiguity_ratio,
            },
        )

    selected = candidates[0]
    return ParserValue(
        "component",
        selected,
        {
            "component_count": len(candidates),
            "candidate_scores": candidate_scores,
            "area": int(selected["area"]),
            "fill_ratio": float(selected["fill_ratio"]),
            "compactness": float(selected["compactness"]),
            "selection_score": float(selected["selection_score"]),
        },
    )


def _component_centroid(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    centroid = inputs[0].value.get("centroid")
    if not isinstance(centroid, (list, tuple)) or len(centroid) < 2:
        raise ParserOpError("component has no valid centroid", code="invalid_component")
    point = [float(centroid[0]), float(centroid[1])]
    return ParserValue("point_pixels", point, {"x": point[0], "y": point[1]})


def _normalize_point(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    point = inputs[0].value
    image = np.asarray(inputs[1].value)
    height, width = image.shape[:2]
    normalized = [
        float(point[0] / max(1, width - 1)),
        float(point[1] / max(1, height - 1)),
    ]
    return ParserValue(
        "normalized_point",
        normalized,
        {"prediction": normalized, "source_width": int(width), "source_height": int(height)},
    )


def _point_in_mask(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    point = np.asarray(inputs[0].value, dtype=np.float64).reshape(-1)
    mask = np.asarray(inputs[1].value)
    if point.size != 2 or mask.ndim != 2:
        raise ParserOpError("point or target-region mask is malformed", code="invalid_geometry")
    foreground = mask > 0
    mask_pixels = int(np.count_nonzero(foreground))
    min_mask_pixels = int(params.get("min_mask_pixels", 200))
    if mask_pixels < min_mask_pixels:
        raise ParserOpError(
            "target relation region is missing or too small",
            code="target_region_missing",
            diagnostics={"mask_pixels": mask_pixels, "min_mask_pixels": min_mask_pixels},
        )

    height, width = mask.shape
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    if not (0 <= x < width and 0 <= y < height):
        raise ParserOpError(
            "grounded point lies outside the generated image",
            code="point_out_of_bounds",
            diagnostics={"x": x, "y": y, "width": width, "height": height},
        )
    radius = int(params.get("radius", 24))
    x1, x2 = max(0, x - radius), min(width, x + radius + 1)
    y1, y2 = max(0, y - radius), min(height, y + radius + 1)
    patch = foreground[y1:y2, x1:x2]
    nearby_pixels = int(np.count_nonzero(patch))
    nearby_fraction = float(nearby_pixels / max(1, patch.size))
    min_fraction = float(params.get("min_fraction", 0.03))
    inside = nearby_pixels > 0 and nearby_fraction >= min_fraction
    return ParserValue(
        "boolean",
        inside,
        {
            "prediction": inside,
            "x": x,
            "y": y,
            "radius": radius,
            "nearby_mask_pixels": nearby_pixels,
            "nearby_mask_fraction": nearby_fraction,
            "min_fraction": min_fraction,
            "mask_pixels": mask_pixels,
        },
    )


def _align_to_reference(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    candidate = np.asarray(inputs[0].value)
    reference = np.asarray(inputs[1].value)
    height, width = reference.shape[:2]
    if candidate.shape[:2] != reference.shape[:2]:
        candidate = cv2.resize(candidate, (width, height), interpolation=cv2.INTER_LANCZOS4)
    motion_name = str(params.get("motion", "euclidean")).strip().lower()
    motion_codes = {
        "translation": cv2.MOTION_TRANSLATION,
        "euclidean": cv2.MOTION_EUCLIDEAN,
        "affine": cv2.MOTION_AFFINE,
    }
    warp = np.eye(2, 3, dtype=np.float32)
    iterations = int(params.get("iterations", 100))
    epsilon = float(params.get("epsilon", 1e-5))
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, epsilon)
    correlation = None
    aligned = candidate
    error = ""
    try:
        correlation, warp = cv2.findTransformECC(
            cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY),
            warp,
            motion_codes[motion_name],
            criteria,
        )
        aligned = cv2.warpAffine(
            candidate,
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT,
        )
    except cv2.error as exc:
        error = str(exc).split("\n", 1)[0]
    return ParserValue(
        "image_bgr",
        aligned,
        {
            "motion": motion_name,
            "correlation": float(correlation) if correlation is not None else None,
            "used_identity_fallback": bool(error),
            "alignment_error": error,
            "width": int(width),
            "height": int(height),
        },
    )


def _image_difference_mask(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    candidate = np.asarray(inputs[0].value)
    reference = np.asarray(inputs[1].value)
    if candidate.shape != reference.shape:
        raise ParserOpError(
            f"image shapes do not match: {candidate.shape} vs {reference.shape}",
            code="image_shape_mismatch",
        )
    blur_kernel = int(params.get("blur_kernel", 3))
    threshold = int(params.get("threshold", 28))
    if blur_kernel > 1:
        candidate = cv2.GaussianBlur(candidate, (blur_kernel, blur_kernel), 0)
        reference = cv2.GaussianBlur(reference, (blur_kernel, blur_kernel), 0)
    difference = cv2.absdiff(candidate, reference)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    return ParserValue(
        "binary_mask",
        mask,
        {"pixel_count": int(np.count_nonzero(mask)), "threshold": threshold, "blur_kernel": blur_kernel},
    )


def _mask_union(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    left = np.asarray(inputs[0].value, dtype=np.uint8)
    right = np.asarray(inputs[1].value, dtype=np.uint8)
    if left.shape != right.shape:
        raise ParserOpError("mask shapes do not match", code="mask_shape_mismatch")
    output = cv2.bitwise_or(left, right)
    return ParserValue("binary_mask", output, {"pixel_count": int(np.count_nonzero(output))})


def _mask_intersection(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    left = np.asarray(inputs[0].value, dtype=np.uint8)
    right = np.asarray(inputs[1].value, dtype=np.uint8)
    if left.shape != right.shape:
        raise ParserOpError("mask shapes do not match", code="mask_shape_mismatch")
    output = cv2.bitwise_and(left, right)
    return ParserValue("binary_mask", output, {"pixel_count": int(np.count_nonzero(output))})


def _select_largest(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    candidates = sorted(list(inputs[0].value), key=lambda row: float(row["area"]), reverse=True)
    if not candidates:
        raise ParserOpError(
            str(params.get("empty_error") or "no matching component found"),
            code="no_component",
            diagnostics={"component_count": 0},
        )
    ambiguity_ratio = float(params.get("ambiguity_ratio", 1.0))
    if (
        ambiguity_ratio < 1.0
        and len(candidates) > 1
        and float(candidates[1]["area"]) >= float(candidates[0]["area"]) * ambiguity_ratio
    ):
        raise ParserOpError(
            str(params.get("ambiguous_error") or "multiple similarly large components found"),
            code="ambiguous_components",
            diagnostics={
                "component_count": len(candidates),
                "candidate_areas": [int(row["area"]) for row in candidates[:3]],
                "ambiguity_ratio": ambiguity_ratio,
            },
        )
    selected = candidates[0]
    return ParserValue(
        "component",
        selected,
        {
            "component_count": len(candidates),
            "area": int(selected["area"]),
            "width": int(selected["width"]),
            "height": int(selected["height"]),
        },
    )


def _component_bbox(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    row = inputs[0].value
    x1, y1 = int(row["x"]), int(row["y"])
    x2, y2 = x1 + int(row["width"]), y1 + int(row["height"])
    return ParserValue("pixel_bbox", [x1, y1, x2, y2], {"bbox": [x1, y1, x2, y2]})


def _crop_around_component(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    image = np.asarray(inputs[0].value)
    row = inputs[1].value
    height, width = image.shape[:2]
    x, y = int(row["x"]), int(row["y"])
    box_width, box_height = int(row["width"]), int(row["height"])
    pad_x = max(int(round(box_width * float(params.get("pad_x", 0.5)))), int(params.get("min_width", 0)) // 2)
    pad_y = max(int(round(box_height * float(params.get("pad_y", 2.0)))), int(params.get("min_height", 0)) // 2)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(width, x + box_width + pad_x), min(height, y + box_height + pad_y)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ParserOpError("component crop is empty", code="empty_crop")
    return ParserValue(
        "image_bgr",
        crop,
        {"crop_bbox": [x1, y1, x2, y2], "width": int(x2 - x1), "height": int(y2 - y1)},
    )


def _crop_dimension_label(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    image = np.asarray(inputs[0].value)
    row = inputs[1].value
    height, width = image.shape[:2]
    x, y = int(row["x"]), int(row["y"])
    box_width, box_height = int(row["width"]), int(row["height"])
    minor_pad = float(params.get("minor_pad", 4.0))
    major_pad = float(params.get("major_pad", 0.75))
    min_minor = int(params.get("min_minor", 512))
    min_major = int(params.get("min_major", 192))
    vertical = box_height > box_width
    if vertical:
        target_width = max(min_minor, int(round(box_width * (1.0 + 2.0 * minor_pad))))
        target_height = max(min_major, int(round(box_height * (1.0 + 2.0 * major_pad))))
    else:
        target_width = max(min_major, int(round(box_width * (1.0 + 2.0 * major_pad))))
        target_height = max(min_minor, int(round(box_height * (1.0 + 2.0 * minor_pad))))
    center_x = x + box_width / 2.0
    center_y = y + box_height / 2.0
    x1 = max(0, int(round(center_x - target_width / 2.0)))
    y1 = max(0, int(round(center_y - target_height / 2.0)))
    x2 = min(width, x1 + target_width)
    y2 = min(height, y1 + target_height)
    x1 = max(0, x2 - target_width)
    y1 = max(0, y2 - target_height)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ParserOpError("dimension-label crop is empty", code="empty_crop")
    return ParserValue(
        "image_bgr",
        crop,
        {
            "crop_bbox": [x1, y1, x2, y2],
            "width": int(x2 - x1),
            "height": int(y2 - y1),
            "dimension_orientation": "vertical" if vertical else "horizontal",
        },
    )


def _component_relation(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    subject, reference = inputs[0].value, inputs[1].value
    sx, sy = [float(value) for value in subject["centroid"]]
    rx, ry = [float(value) for value in reference["centroid"]]
    sbox = _component_box(subject)
    rbox = _component_box(reference)
    intersection = _box_intersection(sbox, rbox)
    min_area = max(1.0, min(float(subject["width"] * subject["height"]), float(reference["width"] * reference["height"])))
    overlap = intersection / min_area
    mode = str(params.get("mode", "dominant_axis")).strip().lower()
    if overlap >= float(params.get("overlap_threshold", 0.25)):
        relation = "overlapping"
    else:
        dx, dy = sx - rx, sy - ry
        image = cv2.imread(context.generated_path)
        diagonal = float(np.hypot(image.shape[1], image.shape[0])) if image is not None else 1.0
        distance = float(np.hypot(dx, dy) / max(diagonal, 1.0))
        if mode == "distance":
            relation = "near" if distance <= float(params.get("near_threshold", 0.2)) else "far"
        elif mode == "horizontal":
            relation = "left" if dx < 0 else "right"
        elif mode == "vertical":
            relation = "above" if dy < 0 else "below"
        elif abs(dx) >= abs(dy):
            relation = "left" if dx < 0 else "right"
        else:
            relation = "above" if dy < 0 else "below"
    return ParserValue(
        "relation_label",
        relation,
        {
            "relation": relation,
            "subject_centroid": [sx, sy],
            "reference_centroid": [rx, ry],
            "overlap_ratio": overlap,
        },
    )


def _map_relation_to_choice(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    relation = str(inputs[0].value).strip().lower()
    choices = list(context.item.get("choices") or context.item.get("options") or [])
    if not choices:
        return ParserValue("choice_label", relation, {"relation": relation, "choice_count": 0})
    aliases = {
        "left": ("left", "to the left"),
        "right": ("right", "to the right"),
        "above": ("above", "over", "on top"),
        "below": ("below", "under", "beneath"),
        "overlapping": ("overlap", "intersect", "blocking", "occlud"),
        "near": ("near", "close", "shortest", "closest"),
        "far": ("far", "farthest", "furthest", "longest"),
    }
    matches = []
    for index, choice in enumerate(choices):
        text = str(choice.get("text") if isinstance(choice, Mapping) else choice).strip().lower()
        if any(term in text for term in aliases.get(relation, (relation,))):
            matches.append(index)
    if len(matches) != 1:
        raise ParserOpError(
            str(params.get("ambiguity_error") or "relation does not map to exactly one choice"),
            code="ambiguous_choice_mapping",
            diagnostics={"relation": relation, "matching_choice_indexes": matches},
        )
    index = matches[0]
    prediction = _choice_prediction(context.item, choices, index)
    return ParserValue(
        "choice_label",
        prediction,
        {"relation": relation, "choice_index": index, "prediction": prediction},
    )


def _map_count_to_choice(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    count = int(inputs[0].value)
    choices = list(context.item.get("choices") or context.item.get("options") or [])
    number_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }
    matches = []
    parsed_values = []
    for index, choice in enumerate(choices):
        text = str(choice.get("text") if isinstance(choice, Mapping) else choice).strip().lower()
        numeric = re.search(r"(?<![\d.])-?\d+(?![\d.])", text.replace(",", ""))
        value = int(numeric.group(0)) if numeric else number_words.get(text)
        parsed_values.append(value)
        if value == count:
            matches.append(index)
    if len(matches) != 1:
        raise ParserOpError(
            str(params.get("ambiguity_error") or "count does not map to exactly one choice"),
            code="ambiguous_choice_mapping",
            diagnostics={
                "count": count,
                "choice_values": parsed_values,
                "matching_choice_indexes": matches,
            },
        )
    index = matches[0]
    prediction = _choice_prediction(context.item, choices, index)
    return ParserValue(
        "choice_label",
        prediction,
        {"count": count, "choice_index": index, "prediction": prediction},
    )


def _ocr_measurement(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    image = np.asarray(inputs[0].value)
    rows = _run_local_ocr(image)
    min_confidence = float(params.get("min_confidence", 0.35))
    pattern = re.compile(str(params.get("numeric_pattern") or r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"))
    unit_hint = str(params.get("unit_hint") or "").strip().lower()
    unit_pattern = re.compile(
        r"(?i)\b(mm|cm|m|ft|in|meters?|metres?|centimeters?|centimetres?|millimeters?|feet|foot|inches?|inch)\b"
    )
    candidates = []
    for text, confidence, box in rows:
        match = pattern.search(text.replace(",", "."))
        if not match or float(confidence) < min_confidence:
            continue
        unit_match = unit_pattern.search(text)
        unit = unit_match.group(1).lower() if unit_match else unit_hint
        candidates.append(
            {
                "value": float(match.group(0)),
                "unit": unit,
                "text": text,
                "confidence": float(confidence),
                "box": box,
                "has_unit": bool(unit_match),
            }
        )
    if not candidates:
        raise ParserOpError(
            "no numeric measurement was recovered by local OCR",
            code="ocr_measurement_missing",
            diagnostics={"ocr_rows": len(rows), "min_confidence": min_confidence},
        )
    candidates.sort(key=lambda row: (row["has_unit"], row["confidence"]), reverse=True)
    best = candidates[0]
    prediction = {"value": best["value"], "unit": best["unit"]}
    return ParserValue(
        "scalar_measurement",
        prediction,
        {
            "prediction": prediction,
            "ocr_text": best["text"],
            "ocr_confidence": best["confidence"],
            "candidate_count": len(candidates),
        },
    )


def _validate_dimension_between(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    measurement, first, second, line = [value.value for value in inputs]
    _validate_dimension_components(first, line, params)
    _validate_dimension_components(second, line, params)
    if first.get("index") == second.get("index") and first == second:
        raise ParserOpError("dimension anchors are not distinct", code="invalid_dimension_anchors")
    return ParserValue(
        "scalar_measurement",
        measurement,
        {
            "prediction": measurement,
            "anchor_areas": [int(first["area"]), int(second["area"])],
            "line_elongation": _component_elongation(line),
            "grounded": True,
        },
    )


def _validate_dimension_extent(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    measurement, anchor, line = [value.value for value in inputs]
    _validate_dimension_components(anchor, line, params)
    return ParserValue(
        "scalar_measurement",
        measurement,
        {
            "prediction": measurement,
            "anchor_area": int(anchor["area"]),
            "line_elongation": _component_elongation(line),
            "grounded": True,
        },
    )


def _clip_match_choices(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    choices = list(context.item.get("choices") or context.item.get("options") or [])
    if len(choices) < 2:
        raise ParserOpError("CLIP choice matching requires at least two choices", code="choices_unavailable")
    texts = [str(choice.get("text") if isinstance(choice, Mapping) else choice) for choice in choices]
    model_name = _resolve_clip_model_name(params)
    model = _load_clip_model(model_name)
    image_rgb = cv2.cvtColor(np.asarray(inputs[0].value), cv2.COLOR_BGR2RGB)
    values = [Image.fromarray(image_rgb), *texts]
    try:
        embeddings = np.asarray(model.encode(values, normalize_embeddings=True))
    except Exception as exc:
        raise ParserOpError(
            f"CLIP encoding failed: {type(exc).__name__}: {exc}",
            code="clip_encoding_failed",
        ) from exc
    scores = embeddings[1:] @ embeddings[0]
    order = np.argsort(scores)[::-1]
    best_index = int(order[0])
    best_score = float(scores[best_index])
    second_score = float(scores[int(order[1])])
    margin = best_score - second_score
    if best_score < float(params.get("min_score", 0.0)) or margin < float(params.get("min_margin", 0.015)):
        raise ParserOpError(
            "CLIP choice match is ambiguous",
            code="ambiguous_clip_match",
            diagnostics={"scores": scores.tolist(), "best_score": best_score, "margin": margin},
        )
    prediction = _choice_prediction(context.item, choices, best_index)
    return ParserValue(
        "choice_label",
        prediction,
        {
            "prediction": prediction,
            "choice_index": best_index,
            "scores": scores.tolist(),
            "best_score": best_score,
            "margin": margin,
            "model": model_name,
        },
    )


def _components_centroids(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    points = [
        [float(component["centroid"][0]), float(component["centroid"][1])]
        for component in inputs[0].value
    ]
    return ParserValue("points_pixels", points, {"point_count": len(points), "points": points})


def _normalize_points(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    points = list(inputs[0].value)
    image = np.asarray(inputs[1].value)
    height, width = image.shape[:2]
    normalized = [
        [float(point[0]) / max(1, width - 1), float(point[1]) / max(1, height - 1)]
        for point in points
    ]
    return ParserValue(
        "normalized_points",
        normalized,
        {"prediction": normalized, "point_count": len(normalized), "width": width, "height": height},
    )


def _normalize_bbox(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    x1, y1, x2, y2 = [float(value) for value in inputs[0].value]
    image = np.asarray(inputs[1].value)
    height, width = image.shape[:2]
    normalized = [
        x1 / max(1, width - 1),
        y1 / max(1, height - 1),
        x2 / max(1, width - 1),
        y2 / max(1, height - 1),
    ]
    return ParserValue("normalized_bbox", normalized, {"prediction": normalized})


def _mask_bbox(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    mask = np.asarray(inputs[0].value, dtype=np.uint8)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ParserOpError("cannot recover a box from an empty mask", code="empty_mask")
    bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    return ParserValue("pixel_bbox", bbox, {"bbox": bbox, "pixel_count": int(len(xs))})


def _skeletonize_mask(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    mask = np.where(np.asarray(inputs[0].value) > 0, 255, 0).astype(np.uint8)
    skeleton = np.zeros_like(mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    current = mask.copy()
    iterations = 0
    max_iterations = int(params.get("max_iterations", 2048))
    while cv2.countNonZero(current) and iterations < max_iterations:
        eroded = cv2.erode(current, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(current, opened))
        current = eroded
        iterations += 1
    if iterations >= max_iterations and cv2.countNonZero(current):
        raise ParserOpError(
            "skeletonization exceeded max_iterations",
            code="skeletonization_limit",
            diagnostics={"max_iterations": max_iterations},
        )
    return ParserValue(
        "binary_mask",
        skeleton,
        {"pixel_count": int(np.count_nonzero(skeleton)), "iterations": iterations},
    )


def _mask_endpoints(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    mask = (np.asarray(inputs[0].value) > 0).astype(np.uint8)
    neighbor_count = cv2.filter2D(mask, cv2.CV_16S, np.ones((3, 3), dtype=np.int16)) - mask
    ys, xs = np.nonzero((mask > 0) & (neighbor_count == 1))
    candidates = np.column_stack((xs, ys)).astype(float)
    if len(candidates) < 2:
        ys, xs = np.nonzero(mask)
        candidates = np.column_stack((xs, ys)).astype(float)
    if len(candidates) < 2:
        raise ParserOpError("path mask has fewer than two pixels", code="path_endpoints_missing")
    differences = candidates[:, None, :] - candidates[None, :, :]
    distances = np.sum(differences * differences, axis=2)
    first, second = np.unravel_index(int(np.argmax(distances)), distances.shape)
    points = [candidates[first].tolist(), candidates[second].tolist()]
    return ParserValue(
        "points_pixels",
        points,
        {"points": points, "endpoint_candidate_count": int(len(candidates))},
    )


def _normalize_polyline(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    points = list(inputs[0].value)
    image = np.asarray(inputs[1].value)
    height, width = image.shape[:2]
    normalized = [
        [float(point[0]) / max(1, width - 1), float(point[1]) / max(1, height - 1)]
        for point in points
    ]
    return ParserValue(
        "normalized_polyline",
        normalized,
        {"prediction": normalized, "point_count": len(normalized)},
    )


def _hough_lines(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    mask = np.asarray(inputs[0].value, dtype=np.uint8)
    lines = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 180.0,
        threshold=int(params.get("threshold", 30)),
        minLineLength=int(params.get("min_line_length", 20)),
        maxLineGap=int(params.get("max_line_gap", 10)),
    )
    values = []
    if lines is None:
        rows = []
    else:
        rows_array = np.asarray(lines)
        if rows_array.ndim == 3 and rows_array.shape[1:] == (1, 4):
            rows = rows_array[:, 0, :]
        elif rows_array.ndim == 2 and rows_array.shape[1] == 4:
            rows = rows_array
        else:
            raise ParserOpError(
                "Hough line detector returned an unexpected shape",
                code="line_detector_contract_error",
                diagnostics={"shape": list(rows_array.shape)},
            )
    for row in rows:
        x1, y1, x2, y2 = [float(value) for value in row]
        values.append(
            {
                "points": [x1, y1, x2, y2],
                "length": float(np.hypot(x2 - x1, y2 - y1)),
                "angle": float(np.degrees(np.arctan2(y2 - y1, x2 - x1))),
            }
        )
    return ParserValue("lines_pixels", values, {"line_count": len(values)})


def _select_longest_line(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    lines = sorted(inputs[0].value, key=lambda row: float(row["length"]), reverse=True)
    if not lines:
        raise ParserOpError("no line segment found", code="line_missing")
    return ParserValue(
        "line_pixels",
        lines[0],
        {"line_count": len(lines), "length": lines[0]["length"], "angle": lines[0]["angle"]},
    )


def _line_endpoints(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    x1, y1, x2, y2 = [float(value) for value in inputs[0].value["points"]]
    points = [[x1, y1], [x2, y2]]
    return ParserValue("points_pixels", points, {"points": points})


def _line_angle(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    angle = float(inputs[0].value["angle"])
    return ParserValue("scalar", angle, {"prediction": angle, "unit": "degrees"})


def _mask_area_ratio(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    mask = np.asarray(inputs[0].value)
    ratio = float(np.count_nonzero(mask) / max(1, mask.size))
    return ParserValue("scalar", ratio, {"prediction": ratio, "pixel_count": int(np.count_nonzero(mask))})


def _color_presence(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    pixels = int(np.count_nonzero(inputs[0].value))
    minimum = int(params.get("min_pixels", 50))
    prediction = pixels >= minimum
    return ParserValue(
        "boolean",
        prediction,
        {"prediction": prediction, "pixel_count": pixels, "min_pixels": minimum},
    )


def _clip_match_source_images(
    context: ParserContext,
    inputs: Sequence[ParserValue],
    params: Mapping[str, Any],
) -> ParserValue:
    start = int(params.get("candidate_start_index", 1))
    candidate_paths = list(context.source_paths[start:])
    if len(candidate_paths) < 2:
        raise ParserOpError(
            "CLIP source-image matching requires at least two candidate images",
            code="candidate_images_unavailable",
            diagnostics={"source_count": len(context.source_paths), "candidate_start_index": start},
        )
    model_name = _resolve_clip_model_name(params)
    model = _load_clip_model(model_name)
    generated_rgb = cv2.cvtColor(np.asarray(inputs[0].value), cv2.COLOR_BGR2RGB)
    images = [Image.fromarray(generated_rgb)]
    for path in candidate_paths:
        try:
            with Image.open(path) as candidate:
                images.append(candidate.convert("RGB").copy())
        except OSError as exc:
            raise ParserOpError(
                f"candidate image unreadable: {path}",
                code="candidate_image_unreadable",
            ) from exc
    embeddings = np.asarray(model.encode(images, normalize_embeddings=True))
    scores = embeddings[1:] @ embeddings[0]
    order = np.argsort(scores)[::-1]
    best_index = int(order[0])
    best_score = float(scores[best_index])
    second_score = float(scores[int(order[1])])
    margin = best_score - second_score
    if best_score < float(params.get("min_score", 0.0)) or margin < float(params.get("min_margin", 0.015)):
        raise ParserOpError(
            "CLIP source-image match is ambiguous",
            code="ambiguous_clip_match",
            diagnostics={"scores": scores.tolist(), "best_score": best_score, "margin": margin},
        )
    choices = list(context.item.get("choices") or context.item.get("options") or [])
    prediction = _choice_prediction(context.item, choices, best_index) if choices else best_index
    return ParserValue(
        "choice_label",
        prediction,
        {
            "prediction": prediction,
            "candidate_index": best_index,
            "candidate_path": candidate_paths[best_index],
            "scores": scores.tolist(),
            "margin": margin,
            "model": model_name,
        },
    )


def _component_box(component: Mapping[str, Any]) -> tuple[float, float, float, float]:
    x1, y1 = float(component["x"]), float(component["y"])
    return x1, y1, x1 + float(component["width"]), y1 + float(component["height"])


def _box_intersection(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _component_elongation(component: Mapping[str, Any]) -> float:
    width, height = max(1.0, float(component["width"])), max(1.0, float(component["height"]))
    return max(width / height, height / width)


def _validate_dimension_components(
    anchor: Mapping[str, Any],
    line: Mapping[str, Any],
    params: Mapping[str, Any],
) -> None:
    if float(anchor["area"]) < float(params.get("min_anchor_area", 30)):
        raise ParserOpError("dimension anchor is too small", code="invalid_dimension_anchor")
    elongation = _component_elongation(line)
    if elongation < float(params.get("min_line_elongation", 2.5)):
        raise ParserOpError(
            "dimension line is not sufficiently elongated",
            code="invalid_dimension_line",
            diagnostics={"line_elongation": elongation},
        )


def _choice_prediction(item: Mapping[str, Any], choices: Sequence[Any], index: int) -> Any:
    answer = item.get("answer")
    choice = choices[index]
    if isinstance(answer, int) and not isinstance(answer, bool):
        return index
    if isinstance(answer, str) and re.fullmatch(r"(?i)[A-Z]", answer.strip()):
        return chr(ord("A") + index)
    if isinstance(choice, Mapping):
        return choice.get("label") or choice.get("text") or index
    return choice


@lru_cache(maxsize=2)
def _load_clip_model(model_name: str):
    if concise_output_enabled():
        try:
            from huggingface_hub.utils import disable_progress_bars

            disable_progress_bars()
        except (ImportError, AttributeError):
            pass
        try:
            from transformers.utils.logging import disable_progress_bar

            disable_progress_bar()
        except (ImportError, AttributeError):
            pass
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ParserOpError(
            "sentence-transformers is required for clip_match_choices",
            code="dependency_missing",
        ) from exc
    device = os.getenv("PROVISE_CLIP_DEVICE", "cpu")
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        raise ParserOpError(
            f"could not load CLIP model {model_name!r}: {type(exc).__name__}: {exc}",
            code="clip_model_unavailable",
        ) from exc


def _resolve_clip_model_name(params: Mapping[str, Any]) -> str:
    return str(
        params.get("model")
        or os.getenv("PROVISE_CLIP_MODEL")
        or "sentence-transformers/clip-ViT-B-32"
    )


@lru_cache(maxsize=1)
def _load_ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR

        return ("rapidocr_onnxruntime", RapidOCR())
    except ImportError:
        try:
            from rapidocr import RapidOCR

            return ("rapidocr", RapidOCR())
        except ImportError as exc:
            raise ParserOpError(
                "A RapidOCR backend is required for ocr_measurement. Install the 'ocr' extra.",
                code="dependency_missing",
            ) from exc


def _run_local_ocr(image: np.ndarray) -> list[tuple[str, float, Any]]:
    backend, engine = _load_ocr_engine()
    try:
        result = engine(image)
    except Exception as exc:
        raise ParserOpError(
            f"{backend} failed: {type(exc).__name__}: {exc}",
            code="ocr_failed",
        ) from exc
    payload = result[0] if isinstance(result, tuple) else result
    if payload is None:
        return []
    rows = []
    for row in payload:
        if isinstance(row, Mapping):
            text = str(row.get("txt") or row.get("text") or "")
            confidence = float(row.get("score") or row.get("confidence") or 0.0)
            box = row.get("box") or row.get("points") or []
        elif isinstance(row, (list, tuple)) and len(row) >= 3:
            box, text, confidence = row[0], str(row[1]), float(row[2])
        else:
            continue
        rows.append((text, confidence, box))
    return rows


def _validate_hsv_params(params: Mapping[str, Any]) -> None:
    lower = _hsv_triplet(params.get("lower"), "lower")
    upper = _hsv_triplet(params.get("upper"), "upper")
    if any(lo > hi for lo, hi in zip(lower, upper)):
        raise ValueError("HSV lower bounds must not exceed upper bounds")


def _validate_load_source_params(params: Mapping[str, Any]) -> None:
    index = int(params.get("index", 0))
    if index < 0:
        raise ValueError("source index must be non-negative")


def _validate_connected_components_params(params: Mapping[str, Any]) -> None:
    connectivity = int(params.get("connectivity", 8))
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")


def _validate_morphology_params(params: Mapping[str, Any]) -> None:
    operation = str(params.get("operation") or "").strip().lower()
    if operation not in {"open", "close", "dilate", "erode"}:
        raise ValueError("operation must be open, close, dilate, or erode")
    kernel_size = int(params.get("kernel_size", 3))
    if kernel_size < 1 or kernel_size > 31 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer between 1 and 31")
    iterations = int(params.get("iterations", 1))
    if iterations < 1 or iterations > 10:
        raise ValueError("iterations must be between 1 and 10")


def _validate_filter_params(params: Mapping[str, Any]) -> None:
    min_area = _non_negative(params.get("min_area", 0), "min_area")
    max_area = _non_negative(params.get("max_area", float("inf")), "max_area")
    if max_area < min_area:
        raise ValueError("max_area must be greater than or equal to min_area")
    _unit_interval(params.get("min_fill_ratio", 0), "min_fill_ratio")
    _unit_interval(params.get("min_compactness", 0), "min_compactness")


def _validate_count_components_params(params: Mapping[str, Any]) -> None:
    minimum = _non_negative(params.get("min_relative_area", 0.25), "min_relative_area")
    maximum = _non_negative(params.get("max_relative_area", 3.0), "max_relative_area")
    if maximum < minimum:
        raise ValueError("max_relative_area must be greater than or equal to min_relative_area")


def _validate_select_params(params: Mapping[str, Any]) -> None:
    _unit_interval(params.get("ambiguity_ratio", 0.85), "ambiguity_ratio")


def _validate_alignment_params(params: Mapping[str, Any]) -> None:
    motion = str(params.get("motion", "euclidean")).strip().lower()
    if motion not in {"translation", "euclidean", "affine"}:
        raise ValueError("motion must be translation, euclidean, or affine")
    iterations = int(params.get("iterations", 100))
    if iterations < 1 or iterations > 1000:
        raise ValueError("iterations must be between 1 and 1000")
    epsilon = float(params.get("epsilon", 1e-5))
    if epsilon <= 0 or epsilon > 0.1:
        raise ValueError("epsilon must be in (0, 0.1]")


def _validate_difference_params(params: Mapping[str, Any]) -> None:
    threshold = int(params.get("threshold", 28))
    if threshold < 0 or threshold > 255:
        raise ValueError("threshold must be between 0 and 255")
    kernel = int(params.get("blur_kernel", 3))
    if kernel < 1 or kernel > 31 or kernel % 2 == 0:
        raise ValueError("blur_kernel must be an odd integer between 1 and 31")


def _validate_crop_params(params: Mapping[str, Any]) -> None:
    for name in ("pad_x", "pad_y"):
        value = float(params.get(name, 0.5 if name == "pad_x" else 2.0))
        if value < 0 or value > 10:
            raise ValueError(f"{name} must be between 0 and 10")
    for name in ("min_width", "min_height"):
        value = int(params.get(name, 0))
        if value < 0 or value > 4096:
            raise ValueError(f"{name} must be between 0 and 4096")


def _validate_dimension_crop_params(params: Mapping[str, Any]) -> None:
    for name, default in (("minor_pad", 4.0), ("major_pad", 0.75)):
        value = float(params.get(name, default))
        if value < 0 or value > 10:
            raise ValueError(f"{name} must be between 0 and 10")
    for name, default in (("min_minor", 512), ("min_major", 192)):
        value = int(params.get(name, default))
        if value < 1 or value > 4096:
            raise ValueError(f"{name} must be between 1 and 4096")


def _validate_relation_params(params: Mapping[str, Any]) -> None:
    mode = str(params.get("mode", "dominant_axis")).strip().lower()
    if mode not in {"dominant_axis", "horizontal", "vertical", "distance"}:
        raise ValueError("mode must be dominant_axis, horizontal, vertical, or distance")
    _unit_interval(params.get("overlap_threshold", 0.25), "overlap_threshold")
    _unit_interval(params.get("near_threshold", 0.2), "near_threshold")


def _validate_ocr_params(params: Mapping[str, Any]) -> None:
    _unit_interval(params.get("min_confidence", 0.35), "min_confidence")
    pattern = str(params.get("numeric_pattern") or r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"numeric_pattern is invalid: {exc}") from exc


def _validate_dimension_params(params: Mapping[str, Any]) -> None:
    elongation = float(params.get("min_line_elongation", 2.5))
    if elongation < 1 or elongation > 100:
        raise ValueError("min_line_elongation must be between 1 and 100")
    _non_negative(params.get("min_anchor_area", 30), "min_anchor_area")


def _validate_clip_params(params: Mapping[str, Any]) -> None:
    score = float(params.get("min_score", 0.0))
    if score < -1 or score > 1:
        raise ValueError("min_score must be between -1 and 1")
    margin = float(params.get("min_margin", 0.015))
    if margin < 0 or margin > 2:
        raise ValueError("min_margin must be between 0 and 2")


def _validate_skeleton_params(params: Mapping[str, Any]) -> None:
    maximum = int(params.get("max_iterations", 2048))
    if maximum < 1 or maximum > 10000:
        raise ValueError("max_iterations must be between 1 and 10000")


def _validate_hough_params(params: Mapping[str, Any]) -> None:
    for name, default in (
        ("threshold", 30),
        ("min_line_length", 20),
        ("max_line_gap", 10),
    ):
        value = int(params.get(name, default))
        if value < 0 or value > 10000:
            raise ValueError(f"{name} must be between 0 and 10000")


def _validate_presence_params(params: Mapping[str, Any]) -> None:
    value = int(params.get("min_pixels", 50))
    if value < 0:
        raise ValueError("min_pixels must be non-negative")


def _validate_point_in_mask_params(params: Mapping[str, Any]) -> None:
    radius = int(params.get("radius", 24))
    if radius < 0 or radius > 512:
        raise ValueError("radius must be between 0 and 512")
    _unit_interval(params.get("min_fraction", 0.03), "min_fraction")
    _non_negative(params.get("min_mask_pixels", 200), "min_mask_pixels")


def _validate_clip_source_params(params: Mapping[str, Any]) -> None:
    _validate_clip_params(params)
    index = int(params.get("candidate_start_index", 1))
    if index < 0 or index > 1000:
        raise ValueError("candidate_start_index must be between 0 and 1000")


def _hsv_triplet(value: Any, name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three HSV values")
    result = [int(item) for item in value]
    limits = [179, 255, 255]
    if any(item < 0 or item > limit for item, limit in zip(result, limits)):
        raise ValueError(f"{name} contains an HSV value outside OpenCV bounds")
    return result


def _non_negative(value: Any, name: str) -> float:
    number = float(value)
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _unit_interval(value: Any, name: str) -> float:
    number = float(value)
    if number < 0 or number > 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return number
