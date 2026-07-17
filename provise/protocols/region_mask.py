from __future__ import annotations

import os
from typing import Any, Dict, List

import cv2
import numpy as np

from ..benchmark.media import resolve_path

from .base import BaseProtocol, ParseResult, ScoreResult


class RegionMaskProtocol(BaseProtocol):
    """Compare a generated binary mask with a benchmark GT mask."""

    name = "region_mask"

    def parse(self, generated_path: str, item: Dict[str, Any], benchmark_root: str) -> ParseResult:
        gt_path = self._gt_mask_path(item, benchmark_root)
        metrics = calc_mask_metrics(generated_path, gt_path)
        ok = metrics.get("gt_exists", False) and metrics.get("pred_exists", False)
        return ParseResult(metrics, ok, metrics)

    def score(self, parsed: ParseResult, item: Dict[str, Any], benchmark_root: str) -> ScoreResult:
        precision = float(parsed.extra.get("precision", 0.0))
        iou = float(parsed.extra.get("iou", 0.0))
        return ScoreResult(
            precision,
            precision > float(self.config.get("success_precision", 0.5)),
            {"prediction": precision, "ground_truth": self._gt_mask_path(item, benchmark_root), "precision": precision, "iou": iou},
        )

    def aggregate(self, details: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
        base = super().aggregate(details, task)
        precisions = [float(d.get("precision", 0.0)) for d in details]
        ious = [float(d.get("iou", 0.0)) for d in details]
        base["mean_precision"] = float(np.mean(precisions)) * 100 if precisions else 0.0
        base["mean_iou"] = float(np.mean(ious)) * 100 if ious else 0.0
        base["accuracy"] = base["mean_precision"]
        return base

    def _gt_mask_path(self, item: Dict[str, Any], benchmark_root: str) -> str:
        mask_file = (
            item.get("evaluation", {}).get("mask_path")
            or item.get("target", {}).get("mask_path")
            or item.get("metadata", {}).get("mask", "")
        )
        mask_file = str(mask_file).replace("/mask/", "/masks/")
        return resolve_path(benchmark_root, mask_file)


def calc_mask_metrics(pred_path: str, gt_path: str) -> Dict[str, float | bool]:
    pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    if pred_img is None or gt_img is None:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "iou": 0.0,
            "pred_exists": pred_img is not None,
            "gt_exists": gt_img is not None,
        }

    height, width = gt_img.shape
    if pred_img.shape != gt_img.shape:
        pred_img = cv2.resize(pred_img, (width, height), interpolation=cv2.INTER_NEAREST)

    pred = pred_img > 127
    gt = gt_img > 127
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_area = pred.sum()
    gt_area = gt.sum()
    return {
        "precision": float(inter / pred_area) if pred_area else 0.0,
        "recall": float(inter / gt_area) if gt_area else 0.0,
        "iou": float(inter / union) if union else 0.0,
        "pred_exists": True,
        "gt_exists": os.path.exists(gt_path),
    }
