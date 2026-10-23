"""Independent near-body change alarm from consecutive 72-beam scans.

This monitor does not identify an object or trust the main moving-cluster track.
It tests whether newly observed range surfaces enter the nominal body footprint
buffer, then asks the skill layer to leave that region. It has no sim truth.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from dynamic_range import point_clusters, scan_points
from footprint_route import HALF_LENGTH_M, HALF_WIDTH_M


def footprint_distances(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0)
    delta = points - pose[:2]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    longitudinal = np.abs(c * delta[:, 0] + s * delta[:, 1])
    lateral = np.abs(-s * delta[:, 0] + c * delta[:, 1])
    return np.hypot(np.maximum(longitudinal - HALF_LENGTH_M, 0.),
                    np.maximum(lateral - HALF_WIDTH_M, 0.))


class RawProximityMonitor:
    """Near-footprint surfaces missing from the previous posed scan."""

    def __init__(self) -> None:
        self.previous_points: np.ndarray | None = None
        self.previous_time_s: float | None = None

    def update(self, ranges: list[float | None], pose: np.ndarray,
               time_s: float) -> dict:
        points = scan_points(ranges, pose)
        gaps = footprint_distances(points, pose)
        near = points[gaps < .085]
        if self.previous_points is None or len(self.previous_points) < 10:
            self.previous_points = points
            self.previous_time_s = time_s
            return {"status": "initialized", "near_point_count": len(near),
                    "changed_near_point_count": 0, "center_xy_m": None,
                    "velocity_xy_m_s": None,
                    "nominal_footprint_gap_m": float(gaps.min()) if len(gaps) else None}
        delta, _ = cKDTree(self.previous_points).query(near) if len(near) else (
            np.empty(0), np.empty(0))
        changed = near[delta > .055]
        clusters = [group for group in point_clusters(changed, link_m=.13)
                    if len(group) >= 3]
        self.previous_points = points
        self.previous_time_s = time_s
        if not clusters:
            return {"status": "no_hazard", "near_point_count": len(near),
                    "changed_near_point_count": len(changed),
                    "center_xy_m": None, "velocity_xy_m_s": None,
                    "nominal_footprint_gap_m": float(gaps.min()) if len(gaps) else None}
        group = min(clusters, key=lambda item: float(
            footprint_distances(item, pose).min()))
        center = group.mean(axis=0)
        gap = float(footprint_distances(group, pose).min())
        return {"status": "near_body_change", "near_point_count": len(near),
                "changed_near_point_count": len(changed),
                "coherent_point_count": len(group),
                "center_xy_m": np.round(center, 4).tolist(),
                "velocity_xy_m_s": [0., 0.],
                "nominal_footprint_gap_m": gap}


def raw_proximity_hazard(observation: dict) -> bool:
    return (observation.get("status") == "near_body_change" and
            observation.get("nominal_footprint_gap_m") is not None and
            observation["nominal_footprint_gap_m"] < .07)
