"""Track a newly observed compact obstacle against a sensor-derived prior.

Inputs are range endpoints, capture-time estimated pose, and prior free-space
cells. The model is deliberately small: one unambiguous circular footprint.
Ambiguous shape or stale observations stay unknown to the planner. No
simulator pose, object identity or evaluator clearance enters this module.
"""

from __future__ import annotations

from collections import deque
import math

import numpy as np

from dynamic_range import point_clusters


class NovelRangeObject:
    def __init__(self, xs: np.ndarray, ys: np.ndarray,
                 prior_free: np.ndarray, prior_occupied: np.ndarray):
        self.xs, self.ys = xs, ys
        self.prior_free, self.prior_occupied = prior_free, prior_occupied
        self.recent_points = deque(maxlen=2)
        self.center_history = deque(maxlen=3)
        self.last_result = None

    def _novel_points(self, ranges, pose_xy):
        points = []
        for angle, distance in zip(np.linspace(-math.pi, math.pi, len(ranges), endpoint=False), ranges):
            if distance is None:
                continue
            point = pose_xy + distance * np.array([math.cos(angle), math.sin(angle)])
            if not (self.xs[0] <= point[0] <= self.xs[-1] and
                    self.ys[0] <= point[1] <= self.ys[-1]):
                continue
            col = int(np.argmin(np.abs(self.xs - point[0])))
            row = int(np.argmin(np.abs(self.ys - point[1])))
            if self.prior_free[row, col] and not self.prior_occupied[row, col]:
                points.append(point)
        return np.asarray(points).reshape(-1, 2)

    @staticmethod
    def _fit_circle(points):
        design = np.c_[2 * points[:, 0], 2 * points[:, 1], np.ones(len(points))]
        cx, cy, constant = np.linalg.lstsq(design, np.sum(points ** 2, axis=1), rcond=None)[0]
        radius = math.sqrt(max(0., constant + cx * cx + cy * cy))
        residual = np.abs(np.linalg.norm(points - np.array([cx, cy]), axis=1) - radius)
        return np.array([cx, cy]), float(radius), float(np.median(residual))

    def update(self, ranges, pose_xy: np.ndarray, capture_time_s: float) -> dict:
        raw = self._novel_points(ranges, pose_xy)
        groups = sorted(point_clusters(raw, link_m=.20), key=len, reverse=True)
        if len(groups) > 1 and len(groups[1]) >= 3:
            result = {"status": "ambiguous_changes", "novel_count": len(raw),
                      "cluster_count": len(groups), "capture_time_s": capture_time_s,
                      "center_xy_m": None, "radius_m": None, "velocity_xy_m_s": None}
            self.recent_points.clear()
            self.center_history.clear()
        elif not groups or len(groups[0]) < 3:
            result = {"status": "no_coherent_change", "novel_count": len(raw),
                      "cluster_count": len(groups), "capture_time_s": capture_time_s,
                      "center_xy_m": None, "radius_m": None, "velocity_xy_m_s": None}
            self.recent_points.clear()
            self.center_history.clear()
        else:
            self.recent_points.append(groups[0])
            points = np.concatenate(tuple(self.recent_points))
            if len(self.recent_points) < 2 or len(points) < 6:
                result = {"status": "candidate", "novel_count": len(raw),
                          "cluster_count": len(groups), "capture_time_s": capture_time_s,
                          "center_xy_m": None, "radius_m": None, "velocity_xy_m_s": None}
            else:
                center, radius, residual = self._fit_circle(points)
                if not (.12 <= radius <= .45 and residual <= .03):
                    result = {"status": "shape_uncertain", "novel_count": len(raw),
                              "cluster_count": len(groups), "capture_time_s": capture_time_s,
                              "center_xy_m": None, "radius_m": None,
                              "fit_radius_m": radius, "fit_median_residual_m": residual,
                              "velocity_xy_m_s": None}
                    self.center_history.clear()
                else:
                    self.center_history.append((capture_time_s, center))
                    velocity = None
                    if len(self.center_history) >= 3:
                        earlier_t, earlier_center = self.center_history[0]
                        dt = capture_time_s - earlier_t
                        if .4 <= dt <= 2.0:
                            velocity = (center - earlier_center) / dt
                    result = {"status": "tracked" if velocity is not None else "shape_confirmed",
                              "novel_count": len(raw), "cluster_count": len(groups),
                              "capture_time_s": capture_time_s,
                              "center_xy_m": center.tolist(), "radius_m": radius,
                              "fit_median_residual_m": residual,
                              "velocity_xy_m_s": velocity.tolist() if velocity is not None else None}
        self.last_result = result
        return result
