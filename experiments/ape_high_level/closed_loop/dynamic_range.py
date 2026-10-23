"""Actor-side diagnostic for motion in successive torso range scans."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def scan_points(ranges: list[float | None], pose: np.ndarray) -> np.ndarray:
    angles = np.linspace(-math.pi, math.pi, len(ranges), endpoint=False)
    return np.asarray([pose[:2] + distance * np.array([
        math.cos(pose[2] + angle), math.sin(pose[2] + angle)])
        for angle, distance in zip(angles, ranges)
        if distance is not None and distance < 1.3]).reshape(-1, 2)


def point_clusters(points: np.ndarray, *, link_m: float = .13) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    neighbors = cKDTree(points).query_ball_point(points, link_m)
    remaining = set(range(len(points)))
    components = []
    while remaining:
        frontier = [remaining.pop()]
        component = []
        while frontier:
            index = frontier.pop()
            component.append(index)
            for neighbor in neighbors[index]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    frontier.append(neighbor)
        if len(component) >= 2:
            components.append(points[component])
    return components


class DynamicRangeTracker:
    """Compare current returns with a first scan; keep one nearby moving cluster."""

    def __init__(self) -> None:
        self.initial_points: np.ndarray | None = None
        self.previous_center: np.ndarray | None = None
        self.previous_time_s: float | None = None
        self.previous_velocity = np.zeros(2)

    def update(self, ranges: list[float | None], pose: np.ndarray,
               time_s: float) -> dict:
        points = scan_points(ranges, pose)
        if self.initial_points is None:
            self.initial_points = points.copy()
            return {"status": "initialized", "dynamic_point_count": 0,
                    "center_xy_m": None, "velocity_xy_m_s": None,
                    "nearest_dynamic_surface_m": None,
                    "projected_surface_distance_0p4s_m": None}
        if len(points) == 0 or len(self.initial_points) == 0:
            return {"status": "no_returns", "dynamic_point_count": 0,
                    "center_xy_m": None, "velocity_xy_m_s": None,
                    "nearest_dynamic_surface_m": None,
                    "projected_surface_distance_0p4s_m": None}
        reference_dist, _ = cKDTree(self.initial_points).query(points)
        changed = points[(reference_dist > .075) &
                         (np.linalg.norm(points - pose[:2], axis=1) < .9)]
        groups = point_clusters(changed)
        predicted = None
        if self.previous_center is not None and self.previous_time_s is not None:
            predicted = (self.previous_center +
                         (time_s - self.previous_time_s) * self.previous_velocity)
            # The moving prop can revisit its first-scan position, where a
            # background-difference test suppresses all of its returns.
            near_prediction = points[np.linalg.norm(points - predicted, axis=1) < .17]
            groups = point_clusters(near_prediction) + groups
        if not groups:
            return {"status": "no_coherent_change", "dynamic_point_count": 0,
                    "center_xy_m": None, "velocity_xy_m_s": None,
                    "nearest_dynamic_surface_m": None,
                    "projected_surface_distance_0p4s_m": None}
        if predicted is None:
            selected = min(groups, key=lambda group: float(np.linalg.norm(
                group.mean(axis=0) - pose[:2])) - .01 * len(group))
        else:
            selected = min(groups, key=lambda group: float(np.linalg.norm(
                group.mean(axis=0) - predicted)) - .005 * len(group))
            if np.linalg.norm(selected.mean(axis=0) - predicted) > .20:
                return {"status": "track_occluded", "dynamic_point_count": 0,
                        "center_xy_m": np.round(predicted, 4).tolist(),
                        "velocity_xy_m_s": np.round(self.previous_velocity, 4).tolist(),
                        "nearest_dynamic_surface_m": None,
                        "projected_surface_distance_0p4s_m": None}
        center = selected.mean(axis=0)
        velocity = None
        if self.previous_center is not None and self.previous_time_s is not None:
            dt = time_s - self.previous_time_s
            if .05 <= dt <= 2.0:
                raw = (center - self.previous_center) / dt
                if np.linalg.norm(raw) <= .6:
                    velocity = raw
                    self.previous_velocity = velocity
        self.previous_center = center
        self.previous_time_s = time_s
        nearest = float(np.min(np.linalg.norm(selected - pose[:2], axis=1)))
        projected = (float(np.min(np.linalg.norm(
            selected + .4 * velocity - pose[:2], axis=1)))
            if velocity is not None else None)
        return {"status": "tracked" if velocity is not None else "candidate",
                "dynamic_point_count": len(selected),
                "center_xy_m": np.round(center, 4).tolist(),
                "velocity_xy_m_s": (np.round(velocity, 4).tolist()
                                    if velocity is not None else None),
                "nearest_dynamic_surface_m": nearest,
                "projected_surface_distance_0p4s_m": projected}


def incoming_hazard(track: dict, pose: np.ndarray) -> bool:
    """Conservative 0.4 s motion flag from actor-side range differences."""
    if (track.get("status") != "tracked" or
            track.get("velocity_xy_m_s") is None or
            track.get("center_xy_m") is None or
            track.get("projected_surface_distance_0p4s_m") is None):
        return False
    velocity = np.asarray(track["velocity_xy_m_s"])
    relative = np.asarray(track["center_xy_m"]) - pose[:2]
    closing = -float(np.dot(velocity, relative)) / (float(np.linalg.norm(relative)) + 1e-9)
    return (float(np.linalg.norm(velocity)) > .10 and closing > .06 and
            track["projected_surface_distance_0p4s_m"] < .24)
