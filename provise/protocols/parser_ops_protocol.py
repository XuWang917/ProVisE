from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from ..evaluation.metrics import score_prediction
from ..parser_ops import DEFAULT_REGISTRY, ParserContext
from .base import BaseProtocol, ParseResult, ScoreResult


class AgenticParserOpsProtocol(BaseProtocol):
    """Generic protocol compiled from a whitelisted deterministic Parser Ops plan."""

    name = "agentic_parser_ops_protocol"

    def parse(self, generated_path: str, item: Dict[str, Any], benchmark_root: str) -> ParseResult:
        pipeline = self.config.get("parser_pipeline")
        if not isinstance(pipeline, dict):
            return ParseResult(
                None,
                False,
                {"error": "parser_pipeline is missing", "error_type": "invalid_pipeline"},
            )
        try:
            source_paths = tuple(self.input_paths(item, benchmark_root))
        except (KeyError, TypeError, ValueError):
            source_paths = ()
        result = DEFAULT_REGISTRY.execute(
            pipeline,
            ParserContext(
                generated_path=generated_path,
                item=item,
                benchmark_root=str(benchmark_root),
                source_paths=source_paths,
                protocol_config=self.config,
            ),
        )
        if not result.success or result.output is None:
            return ParseResult(
                None,
                False,
                {
                    "error": result.error,
                    "error_type": result.error_type,
                    "parser": "parser_ops",
                    "parser_backend": "deterministic",
                    "parser_ops": result.diagnostics,
                },
            )

        prediction = result.prediction
        output_kind = result.output.kind
        if isinstance(prediction, np.ndarray):
            parsed_dir = Path(generated_path).parent / "parsed"
            parsed_dir.mkdir(parents=True, exist_ok=True)
            parsed_path = parsed_dir / f"{Path(generated_path).stem}_{output_kind}.png"
            cv2.imwrite(str(parsed_path), prediction)
            prediction = str(parsed_path)
        return ParseResult(
            prediction,
            True,
            {
                "prediction": prediction,
                "parser": "parser_ops",
                "parser_backend": "deterministic",
                "parser_output_kind": output_kind,
                "parser_ops": result.diagnostics,
                "spatial_evidence_valid": True,
            },
        )

    def score(self, parsed: ParseResult, item: Dict[str, Any], benchmark_root: str) -> ScoreResult:
        metric = str(self.config.get("metric") or (item.get("evaluation") or {}).get("metric") or "unverified")
        metric_config = dict((item.get("evaluation") or {}).get("metric_config") or {})
        metric_config.update(dict(self.config.get("metric_config") or {}))
        result = score_prediction(metric, parsed.prediction, item, benchmark_root, metric_config)
        return ScoreResult(result.score, result.is_correct, dict(result.extra))
