from __future__ import annotations

from typing import Any, Dict, Tuple

from .base import BaseProtocol


PROTOCOL_SPECS: Dict[str, Tuple[str, str]] = {
    "label_code": (".label_code", "LabelCodeProtocol"),
    "instance_marker_count": (".instance_marker", "InstanceMarkerCountProtocol"),
    "direction_grid": (".direction", "DirectionGridProtocol"),
    "binary_color_presence": (".binary_color", "BinaryColorPresenceProtocol"),
    "dense_depth_ab": (".depth", "DenseDepthABProtocol"),
    "region_mask": (".region_mask", "RegionMaskProtocol"),
    "trajectory": (".trajectory", "TrajectoryProtocol"),
    "state_similarity": (".state_similarity", "StateSimilarityProtocol"),
    "agentic_point_marker": (".agentic", "AgenticPointMarkerProtocol"),
    "agentic_vlm_protocol": (".agentic_vlm", "AgenticVLMProtocol"),
    "agentic_parser_ops_protocol": (".parser_ops_protocol", "AgenticParserOpsProtocol"),
    "generic_vlm_fallback": (".fallback", "GenericVLMFallbackProtocol"),
}


def create_protocol(name: str, config: Dict[str, Any] | None = None) -> BaseProtocol:
    if name not in PROTOCOL_SPECS:
        raise ValueError(f"Unknown protocol: {name}. Available: {sorted(PROTOCOL_SPECS)}")
    module_name, class_name = PROTOCOL_SPECS[name]
    from importlib import import_module

    module = import_module(module_name, package=__package__)
    cls = getattr(module, class_name)
    return cls(config or {})


def list_protocols() -> list[str]:
    return sorted(PROTOCOL_SPECS)
