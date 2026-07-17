from __future__ import annotations

from typing import Any, Dict

from ..evaluation.metrics import score_prediction
from ..parser_ops import DEFAULT_REGISTRY, ParserContext, green_instance_marker_count_pipeline
from .base import BaseProtocol, ParseResult, ScoreResult


class InstanceMarkerCountProtocol(BaseProtocol):
    """Count vivid green instance markers in the generated image."""

    name = "instance_marker_count"

    def parse(self, generated_path: str, item: Dict[str, Any], benchmark_root: str) -> ParseResult:
        pipeline = self.config.get("parser_pipeline") or green_instance_marker_count_pipeline()
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
        steps = result.diagnostics.get("steps") or {}
        count_diagnostics = steps.get("count") or {}
        if not result.success:
            return ParseResult(
                0,
                False,
                {
                    "error": result.error,
                    "error_type": result.error_type,
                    "parser": "parser_ops",
                    "parser_ops": result.diagnostics,
                },
            )
        return ParseResult(
            int(result.prediction),
            True,
            {
                "num_components": int(count_diagnostics.get("component_count") or 0),
                "median_area": float(count_diagnostics.get("median_area") or 0.0),
                "parser": "parser_ops",
                "parser_ops": result.diagnostics,
            },
        )

    def score(self, parsed: ParseResult, item: Dict[str, Any], benchmark_root: str) -> ScoreResult:
        result = score_prediction(
            str(self.config.get("metric") or "exact_count"),
            parsed.prediction,
            item,
            benchmark_root,
            self.config.get("metric_config") or {},
        )
        return ScoreResult(
            result.score,
            result.is_correct,
            result.extra,
        )

    def aggregate(self, details: list[Dict[str, Any]], task: str) -> Dict[str, Any]:
        summary = super().aggregate(details, task)
        smape_values = [
            float(row["smape_percent"])
            for row in details
            if row.get("smape_percent") is not None
        ]
        if smape_values:
            summary["official_smape_percent"] = sum(smape_values) / len(smape_values)
        return summary
