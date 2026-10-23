"""Convert a virtual chest depth frame into a conservative local obstacle set."""

from __future__ import annotations

import math

import numpy as np


def obstacle_points(depth: np.ndarray, camera_pose: tuple[np.ndarray, np.ndarray],
                    base_estimate: np.ndarray, *, fovy_deg: float = 67.,
                    stride: int = 6,
                    self_exclusion_radius_m: float = .12) -> np.ndarray:
    height, width = depth.shape
    focal = height / (2 * math.tan(math.radians(fovy_deg) / 2))
    rows, columns = np.mgrid[stride // 2:height:stride,
                             stride // 2:width:stride]
    d = depth[rows, columns]
    valid = np.isfinite(d) & (d > .08) & (d < 1.8)
    rays = np.stack(((columns - width / 2) / focal,
                     -(rows - height / 2) / focal,
                     -np.ones_like(d)), axis=-1)
    origin, rotation = camera_pose
    world = origin + (rays[valid] * d[valid, None]) @ rotation.T
    radial = np.linalg.norm(world[:, :2] - base_estimate[:2], axis=1)
    occupied = ((world[:, 2] > .035) & (world[:, 2] < .55) &
                (radial > self_exclusion_radius_m) & (radial < 1.2))
    return world[occupied, :2]


def clear_depth_heading(desired: float, points: np.ndarray,
                        pose_estimate: np.ndarray, target_xy: np.ndarray,
                        *, footprint_margin_m: float = .22,
                        max_sector_rad: float = .66,
                        search_offset_rad: float = .6) -> tuple[float, float, float] | None:
    """Choose one camera-observed short step; unknown side sectors are blocked."""
    if len(points) < 10:
        return None
    options = []
    for offset in np.linspace(-search_offset_rad, search_offset_rad, 25):
        heading = desired + float(offset)
        relative = math.atan2(math.sin(heading - pose_estimate[2]),
                              math.cos(heading - pose_estimate[2]))
        if abs(relative) > max_sector_rad:
            continue
        step = .08 * np.array([math.cos(heading), math.sin(heading)])
        clearance = min(float(np.min(np.linalg.norm(
            points - (pose_estimate[:2] + fraction * step), axis=1)))
            for fraction in (.5, 1.0))
        if clearance < footprint_margin_m:
            continue
        remaining = float(np.linalg.norm(target_xy - (pose_estimate[:2] + step)))
        score = remaining + .025 * abs(float(offset)) - .01 * min(clearance, 1.0)
        options.append((score, abs(float(offset)), heading, float(offset), clearance))
    if not options:
        return None
    _, _, heading, offset, clearance = min(options)
    return heading, offset, clearance
