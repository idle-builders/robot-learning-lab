"""10 Hz actor-side near-body scan-change alarm during a translation skill.

This uses only rays, odometry and a fixed neutral-body scan-plane envelope.
It does not see simulator contacts, obstacle names or a future prop trajectory.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from collections.abc import Callable

from dynamic_range import point_clusters, scan_points


def scan_plane_gap(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """XY distance to the neutral collision silhouette near torso-ray height."""
    if len(points) == 0:
        return np.empty(0)
    delta = points - pose[:2]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    forward = np.abs(c * delta[:, 0] + s * delta[:, 1])
    side = np.abs(-s * delta[:, 0] + c * delta[:, 1])
    return np.hypot(np.maximum(forward - .07, 0.),
                    np.maximum(side - .17, 0.))


class FastRangeGuard:
    """Flag a coherent changed surface inside the 4 cm scan-plane band."""

    def __init__(self, *, band_m: float = .04,
                 near_cluster_min_points: int | None = None,
                 single_point_critical_band_m: float | None = None) -> None:
        self.previous_points: np.ndarray | None = None
        self.previous_time_s: float | None = None
        self.band_m = band_m
        self.near_cluster_min_points = near_cluster_min_points
        self.single_point_critical_band_m = single_point_critical_band_m

    def update(self, ranges: list[float | None], pose: np.ndarray,
               time_s: float, *, point_gap: Callable[[np.ndarray], np.ndarray] | None = None) -> dict:
        points = scan_points(ranges, pose)
        gap = point_gap if point_gap is not None else lambda xy: scan_plane_gap(xy, pose)
        gaps = gap(points)
        near = points[gaps < self.band_m]
        if (self.single_point_critical_band_m is not None and len(gaps)
                and float(gaps.min()) < self.single_point_critical_band_m):
            index = int(np.argmin(gaps))
            group = points[index:index + 1]
            self.previous_points = points
            self.previous_time_s = time_s
            return {"status": "critical_body_point", "near_point_count": len(near),
                    "changed_near_point_count": 0, "coherent_point_count": 1,
                    "coherent_points_xy_m": np.round(group, 4).tolist(),
                    "center_xy_m": np.round(group[0], 4).tolist(),
                    "velocity_xy_m_s": [0., 0.],
                    "scan_plane_gap_m": float(gaps[index])}
        if self.near_cluster_min_points is not None:
            near_groups = [group for group in point_clusters(near, link_m=.06)
                           if len(group) >= self.near_cluster_min_points]
            if near_groups:
                group = min(near_groups, key=lambda item: float(gap(item).min()))
                self.previous_points = points
                self.previous_time_s = time_s
                return {"status": "near_body_cluster", "near_point_count": len(near),
                        "changed_near_point_count": 0,
                        "coherent_point_count": len(group),
                        "coherent_points_xy_m": np.round(group, 4).tolist(),
                        "center_xy_m": np.round(group.mean(axis=0), 4).tolist(),
                        "velocity_xy_m_s": [0., 0.],
                        "scan_plane_gap_m": float(gap(group).min())}
        if self.previous_points is None or len(self.previous_points) < 10:
            self.previous_points = points
            self.previous_time_s = time_s
            return {"status": "initialized", "near_point_count": len(near),
                    "changed_near_point_count": 0,
                    "scan_plane_gap_m": float(gaps.min()) if len(gaps) else None}
        if len(near):
            distance, _ = cKDTree(self.previous_points).query(near)
            changed = near[distance > .01]
        else:
            changed = np.empty((0, 2))
        groups = [group for group in point_clusters(changed, link_m=.06)
                  if len(group) >= 3]
        self.previous_points = points
        self.previous_time_s = time_s
        if not groups:
            return {"status": "no_hazard", "near_point_count": len(near),
                    "changed_near_point_count": len(changed),
                    "scan_plane_gap_m": float(gaps.min()) if len(gaps) else None}
        group = min(groups, key=lambda item: float(gap(item).min()))
        return {"status": "near_body_change", "near_point_count": len(near),
                "changed_near_point_count": len(changed),
                "coherent_point_count": len(group),
                "coherent_points_xy_m": np.round(group, 4).tolist(),
                "center_xy_m": np.round(group.mean(axis=0), 4).tolist(),
                "velocity_xy_m_s": [0., 0.],
                "scan_plane_gap_m": float(gap(group).min())}
