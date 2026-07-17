from __future__ import annotations

import ast
from collections import deque
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ..benchmark.media import resolve_primary_media_path

from .base import BaseProtocol, ParseResult, ScoreResult


class TrajectoryProtocol(BaseProtocol):
    """Parse a red trajectory line and evaluate with normalized DFD."""

    name = "trajectory"

    def variables(self, item: Dict[str, Any], benchmark_root: str) -> Dict[str, Any]:
        values = super().variables(item, benchmark_root)
        values["start_point"] = normalized_start_point(item, benchmark_root)
        return values

    def parse(self, generated_path: str, item: Dict[str, Any], benchmark_root: str) -> ParseResult:
        source_path = resolve_source_image_path(item, benchmark_root)
        start = normalized_start_xy(item, benchmark_root)
        points, extra = extract_red_trajectory(
            generated_path,
            source_path=source_path,
            start_point=start,
            n_samples=int(self.config.get("n_samples", 20)),
        )
        extra["n_pred"] = len(points)
        return ParseResult(points, bool(points), extra)

    def score(self, parsed: ParseResult, item: Dict[str, Any], benchmark_root: str) -> ScoreResult:
        image_path = resolve_source_image_path(item, benchmark_root)
        src = cv2.imread(image_path)
        src_h, src_w = src.shape[:2] if src is not None else (1, 1)
        gt_raw = parse_trajectory(str(item.get("answer", "")))
        gt = [[x / src_w, y / src_h] for x, y in gt_raw]
        dfd = trajectory_distance(parsed.prediction, gt)
        threshold = float(self.config.get("success_dfd", 0.4))
        return ScoreResult(
            max(0.0, 1.0 - dfd),
            dfd < threshold,
            {"prediction": parsed.prediction, "ground_truth": gt, "dfd": dfd, "threshold": threshold},
        )

    def aggregate(self, details: List[Dict[str, Any]], task: str) -> Dict[str, Any]:
        base = super().aggregate(details, task)
        dfds = [float(d.get("dfd", 1.0)) for d in details]
        base["mean_dfd"] = float(np.mean(dfds)) if dfds else 1.0
        return base


def parse_trajectory(text: str) -> List[List[float]]:
    try:
        pts = ast.literal_eval(text)
        return [[float(x), float(y)] for x, y in pts]
    except Exception:
        return []


def resolve_source_image_path(item: Dict[str, Any], benchmark_root: str) -> str:
    path = resolve_primary_media_path(item, benchmark_root)
    if path:
        return path
    raise KeyError("image_path")


def normalized_start_xy(item: Dict[str, Any], benchmark_root: str) -> Tuple[float, float] | None:
    points = parse_trajectory(str(item.get("answer", "")))
    if not points:
        return None
    image = cv2.imread(resolve_source_image_path(item, benchmark_root))
    if image is None:
        return None
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return None
    return float(points[0][0] / width), float(points[0][1] / height)


def normalized_start_point(item: Dict[str, Any], benchmark_root: str) -> str:
    start = normalized_start_xy(item, benchmark_root)
    if start is None:
        return ""
    return f"({start[0]:.3f}, {start[1]:.3f})"


def extract_red_trajectory(
    image_path: str,
    source_path: str | None = None,
    start_point: Tuple[float, float] | None = None,
    n_samples: int = 20,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    image = cv2.imread(image_path)
    if image is None:
        return [], {"error": "generated image unreadable"}

    source = cv2.imread(source_path) if source_path else None
    if source is not None and source.shape[:2] != image.shape[:2]:
        source = cv2.resize(source, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    mask, diagnostics = build_trajectory_mask(image, source)
    if cv2.countNonZero(mask) == 0:
        diagnostics["error"] = "no trajectory candidates found"
        return [], diagnostics

    component, component_info = select_primary_component(mask, start_point)
    diagnostics.update(component_info)
    skeleton = skeletonize_mask(component)
    path_pixels = ordered_path_pixels(skeleton, start_point)
    if len(path_pixels) < 2:
        path_pixels = ordered_path_pixels(component, start_point)
    if len(path_pixels) < 2:
        diagnostics["error"] = "trajectory path ordering failed"
        return [], diagnostics

    height, width = image.shape[:2]
    points = sample_path(path_pixels, width, height, n_samples)
    diagnostics["path_points"] = len(path_pixels)
    return points, diagnostics


def build_trajectory_mask(image: np.ndarray, source: np.ndarray | None) -> Tuple[np.ndarray, Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {}
    img16 = image.astype(np.int16)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = image_hsv[:, :, 0]
    sat = image_hsv[:, :, 1]
    val = image_hsv[:, :, 2]
    red_hue = (((hue <= 15) | (hue >= 170)) & (sat >= 50) & (val >= 60))
    red_dominance = img16[:, :, 2] - np.maximum(img16[:, :, 1], img16[:, :, 0])
    red_like = red_hue | (red_dominance >= 20)

    if source is None:
        candidate = red_like.astype(np.uint8) * 255
        diagnostics["mask_mode"] = "red_only"
    else:
        diff = cv2.absdiff(image, source)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        src16 = source.astype(np.int16)
        red_gain = img16[:, :, 2] - src16[:, :, 2]
        a_shift = (
            cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.int16)
            - cv2.cvtColor(source, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.int16)
        )
        changed = diff_gray >= 18
        candidate = (
            changed
            & red_like
            & ((red_gain >= 12) | (a_shift >= 8) | (red_dominance >= 35))
        ).astype(np.uint8) * 255
        diagnostics["mask_mode"] = "source_diff_red"
        diagnostics["changed_pixels"] = int(np.count_nonzero(changed))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel, iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel, iterations=2)
    candidate = cv2.dilate(candidate, kernel, iterations=1)
    diagnostics["candidate_pixels"] = int(cv2.countNonZero(candidate))
    return candidate, diagnostics


def select_primary_component(
    mask: np.ndarray,
    start_point: Tuple[float, float] | None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask, {"selected_component_area": int(cv2.countNonZero(mask))}

    height, width = mask.shape[:2]
    if start_point is not None:
        sx = int(round(start_point[0] * (width - 1)))
        sy = int(round(start_point[1] * (height - 1)))
    else:
        sx = sy = None

    best_label = None
    best_key = None
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 6:
            continue
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue
        if sx is None or sy is None:
            key = (0, -area)
        else:
            dist2 = int(np.min((xs - sx) ** 2 + (ys - sy) ** 2))
            near_start = dist2 <= max(16, min(width, height) // 20) ** 2
            key = (0 if near_start else 1, dist2, -area)
        if best_key is None or key < best_key:
            best_key = key
            best_label = label

    if best_label is None:
        return mask, {"selected_component_area": int(cv2.countNonZero(mask))}

    component = np.zeros_like(mask)
    component[labels == best_label] = 255
    info = {
        "selected_component_area": int(cv2.countNonZero(component)),
        "component_count": int(num_labels - 1),
    }
    return component, info


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    work = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(work)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(work, opened)
        eroded = cv2.erode(work, element)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break
    return skel


def ordered_path_pixels(mask: np.ndarray, start_point: Tuple[float, float] | None) -> List[Tuple[int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []

    coords = np.stack([xs, ys], axis=1)
    width = mask.shape[1]
    height = mask.shape[0]
    if start_point is None:
        start_idx = int(np.argmin(coords[:, 0] + coords[:, 1]))
    else:
        sx = start_point[0] * (width - 1)
        sy = start_point[1] * (height - 1)
        start_idx = int(np.argmin((coords[:, 0] - sx) ** 2 + (coords[:, 1] - sy) ** 2))
    start = (int(coords[start_idx, 0]), int(coords[start_idx, 1]))

    pixel_set = {tuple(map(int, pair)) for pair in coords}
    queue = deque([start])
    parent = {start: None}
    distance = {start: 0}
    while queue:
        x, y = queue.popleft()
        for nx in range(x - 1, x + 2):
            for ny in range(y - 1, y + 2):
                if nx == x and ny == y:
                    continue
                neighbor = (nx, ny)
                if neighbor in pixel_set and neighbor not in parent:
                    parent[neighbor] = (x, y)
                    distance[neighbor] = distance[(x, y)] + 1
                    queue.append(neighbor)

    if not distance:
        return []

    end = max(distance, key=distance.get)
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def sample_path(
    path_pixels: List[Tuple[int, int]],
    width: int,
    height: int,
    n_samples: int,
) -> List[List[float]]:
    if not path_pixels:
        return []
    if len(path_pixels) == 1:
        x, y = path_pixels[0]
        return [[float(x) / width, float(y) / height]]
    count = max(2, n_samples)
    idxs = np.linspace(0, len(path_pixels) - 1, count).astype(int)
    return [
        [float(path_pixels[i][0]) / width, float(path_pixels[i][1]) / height]
        for i in idxs
    ]


def trajectory_distance(pred_pts: List, gt_pts: List) -> float:
    if not pred_pts or not gt_pts:
        return 1.0
    pred = np.array(pred_pts, dtype=float)
    gt = np.array(gt_pts, dtype=float)
    dist = np.sqrt(((pred[:, None, :] - gt[None, :, :]) ** 2).sum(axis=2))
    n, m = dist.shape
    ca = np.full((n, m), np.inf)
    ca[0, 0] = dist[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], dist[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], dist[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(dist[i, j], min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]))
    return float(ca[-1, -1])
