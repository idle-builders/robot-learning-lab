"""Project head depth into observed upper-volume route occupancy.

No simulator labels or future object positions are used. Visible surfaces may
veto an unqualified route, but the depth image cannot certify unseen space.
"""

from __future__ import annotations

import numpy as np


def observed_upper_points(depth_m: np.ndarray, estimated_xy_m: np.ndarray,
                          camera_z_m: float, camera_rotation: np.ndarray,
                          fovy_deg: float) -> np.ndarray:
    height, width = depth_m.shape
    rows, columns = np.indices(depth_m.shape)
    usable = np.isfinite(depth_m) & (depth_m > .10) & (depth_m < 1.0)
    focal = height / (2. * np.tan(np.deg2rad(fovy_deg) / 2.))
    values = depth_m[usable]
    camera_points = np.column_stack((
        (columns[usable] + .5 - width / 2.) * values / focal,
        (height / 2. - rows[usable] - .5) * values / focal,
        -values))
    world = camera_points @ camera_rotation.T + np.r_[estimated_xy_m,
                                                      camera_z_m]
    near_upper = ((world[:, 2] >= .46) & (world[:, 2] <= .64) &
                  (np.linalg.norm(world[:, :2] - estimated_xy_m, axis=1) < 1.2))
    return world[near_upper]


def route_upper_gap(points_xyz_m: np.ndarray,
                    path_xy_m: list[list[float]]) -> dict:
    path = np.asarray(path_xy_m, dtype=float)
    if len(points_xyz_m) == 0 or len(path) < 2:
        return {"upper_points": len(points_xyz_m),
                "minimum_observed_gap_m": None,
                "near_margin_point_count": 0,
                "nearest_observed_xyz_m": None}
    points = np.asarray(points_xyz_m, dtype=float)
    nearest_xy = np.full(len(points), np.inf)
    for start, end in zip(path, path[1:]):
        delta = end - start
        length_sq = float(delta @ delta)
        if length_sq < 1.e-12:
            continue
        fraction = np.clip(((points[:, :2] - start) @ delta) / length_sq,
                           0., 1.)
        on_path = start + fraction[:, None] * delta
        nearest_xy = np.minimum(nearest_xy,
                                np.linalg.norm(points[:, :2] - on_path,
                                               axis=1))
    radial = np.maximum(0., nearest_xy - .17)
    vertical = np.maximum(0., points[:, 2] - .50)
    gap = np.hypot(radial, vertical)
    index = int(np.argmin(gap))
    return {"upper_points": len(points),
            "minimum_observed_gap_m": round(float(gap[index]), 5),
            "near_margin_point_count": int(np.count_nonzero(gap < .05)),
            "nearest_observed_xyz_m": points[index].round(5).tolist()}
