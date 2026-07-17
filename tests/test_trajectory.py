from pathlib import Path

import cv2
import numpy as np

from provise.protocols.trajectory import extract_red_trajectory, resolve_source_image_path


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


def test_extract_red_trajectory_uses_source_diff_to_ignore_existing_colored_regions(tmp_path):
    source = np.full((120, 200, 3), 235, dtype=np.uint8)
    source[15:75, 140:190] = (180, 60, 200)
    source[80:110, 20:120] = (40, 120, 220)

    generated = source.copy()
    cv2.line(generated, (30, 90), (120, 30), (0, 0, 255), thickness=5)

    source_path = tmp_path / "source.png"
    generated_path = tmp_path / "generated.png"
    _write_image(source_path, source)
    _write_image(generated_path, generated)

    points, extra = extract_red_trajectory(
        str(generated_path),
        source_path=str(source_path),
        start_point=(30 / 200, 90 / 120),
        n_samples=12,
    )

    assert points
    assert abs(points[0][0] - 0.15) < 0.08
    assert abs(points[0][1] - 0.75) < 0.08
    assert abs(points[-1][0] - 0.60) < 0.08
    assert abs(points[-1][1] - 0.25) < 0.08
    assert extra["candidate_pixels"] > 0


def test_extract_red_trajectory_returns_empty_without_new_stroke(tmp_path):
    source = np.full((80, 120, 3), 220, dtype=np.uint8)
    source[10:50, 60:100] = (180, 60, 200)

    source_path = tmp_path / "source.png"
    generated_path = tmp_path / "generated.png"
    _write_image(source_path, source)
    _write_image(generated_path, source.copy())

    points, extra = extract_red_trajectory(
        str(generated_path),
        source_path=str(source_path),
        start_point=(0.2, 0.7),
        n_samples=12,
    )

    assert points == []
    assert extra["candidate_pixels"] == 0


def test_resolve_source_image_path_supports_unified_nested_media(tmp_path):
    image_path = tmp_path / "interaction" / "trajectory" / "1005_frame_0.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(image_path), np.full((8, 8, 3), 255, dtype=np.uint8))

    item = {
        "input": {
            "media": [
                {
                    "role": "primary",
                    "media": {"type": "image", "path": "interaction/trajectory/1005_frame_0.png"},
                }
            ]
        }
    }

    resolved = resolve_source_image_path(item, str(tmp_path))

    assert resolved == str(image_path.resolve())
