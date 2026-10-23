"""Conservative RGB-D position from an accepted image box under depth holes.

This is an actor-only local estimate, not an object mask. It refuses sparse
or mixed-depth center patches. The caller must still attach pose/box age and
uncertainty before using the point for a motion or offer-zone decision.
"""

from __future__ import annotations

import math

import numpy as np


def locate_box_center(box: list[float] | None,
                      camera_pose: tuple[np.ndarray, np.ndarray],
                      depth: np.ndarray, *, fovy_deg: float,
                      min_valid: int = 12, max_mad_m: float = .05) -> dict:
    if box is None:
        return {"world_xyz_m": None, "valid_pixels": 0,
                "patch_pixels": 0, "median_absolute_deviation_m": None,
                "status": "no_box"}
    height, width = depth.shape
    u = (box[0] + box[2]) / 2
    v = (box[1] + box[3]) / 2
    x, y = int(np.clip(u, 0, width - 1)), int(np.clip(v, 0, height - 1))
    patch = depth[max(0, y - 2):min(height, y + 3),
                  max(0, x - 2):min(width, x + 3)]
    valid = patch[np.isfinite(patch) & (patch > .08) & (patch < 2.)]
    base = {"valid_pixels": int(len(valid)), "patch_pixels": int(patch.size)}
    if len(valid) < min_valid:
        return {**base, "world_xyz_m": None,
                "median_absolute_deviation_m": None,
                "status": "insufficient_depth"}
    distance = float(np.median(valid))
    mad = float(np.median(np.abs(valid - distance)))
    if mad > max_mad_m:
        return {**base, "world_xyz_m": None,
                "median_absolute_deviation_m": round(mad, 5),
                "status": "mixed_depth"}
    focal = height / (2 * math.tan(math.radians(fovy_deg) / 2))
    ray = np.asarray(((u - width / 2) / focal,
                      -(v - height / 2) / focal, -1.))
    origin, rotation = camera_pose
    world = origin + rotation @ (distance * ray)
    return {**base, "world_xyz_m": world.tolist(),
            "median_absolute_deviation_m": round(mad, 5),
            "status": "located"}
