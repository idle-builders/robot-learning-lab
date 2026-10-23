"""Pixel and capture-pose features for the bounded synthetic RGB correction."""

from __future__ import annotations

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.virtual_rgb_goal import orange_bbox

FEATURE_NAMES = [
    "coarse_world_x_m", "coarse_world_y_m",
    "centroid_u_norm", "centroid_v_norm",
    "bbox_x_norm", "bbox_y_norm", "bbox_w_norm", "bbox_h_norm",
    "orange_pixels_norm", "bbox_occupancy",
    "mask_std_u_norm", "mask_std_v_norm", "mask_skew_u", "mask_skew_v",
    "camera_world_x_m", "camera_world_y_m", "camera_world_z_m",
    *[f"camera_rotation_{i}" for i in range(9)],
]


def features(rgb: np.ndarray, coarse_world_xyz: list[float] | np.ndarray,
             camera_world_xyz: np.ndarray, camera_rotation: np.ndarray) -> np.ndarray:
    bbox = orange_bbox(rgb)
    if bbox is None:
        raise ValueError("A valid orange mask is required for features")
    red = rgb[:, :, 0].astype(int)
    green = rgb[:, :, 1].astype(int)
    blue = rgb[:, :, 2].astype(int)
    yy, xx = np.nonzero((red > 100) & (red > green * 1.25) & (green > blue * 1.25))
    x, y, width, height = bbox["xywh_px"]
    u, v = bbox["centroid_uv_px"]
    std_x = max(float(xx.std()), 1e-6)
    std_y = max(float(yy.std()), 1e-6)
    skew_x = float(np.mean(((xx - xx.mean()) / std_x) ** 3))
    skew_y = float(np.mean(((yy - yy.mean()) / std_y) ** 3))
    row = np.asarray([
        coarse_world_xyz[0], coarse_world_xyz[1],
        u / 320, v / 240,
        x / 320, y / 240, width / 320, height / 240,
        len(xx) / 100, len(xx) / (width * height),
        std_x / 320, std_y / 240, skew_x, skew_y,
        *camera_world_xyz, *np.asarray(camera_rotation).reshape(-1),
    ], dtype=np.float64)
    if len(row) != len(FEATURE_NAMES) or not np.isfinite(row).all():
        raise ValueError("Nonfinite or incompatible RGB calibration feature")
    return row
