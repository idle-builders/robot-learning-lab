"""Local 2-D scan registration against a sensor-built room map.

The first healthy scan anchors a map at the estimated starting pose. Later
scans correct encoder/command odometry; simulator pose is never an input.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


class ScanLocalizer:
    def __init__(self) -> None:
        self.points: np.ndarray | None = None
        self.last: dict = {"status": "uninitialized"}

    @staticmethod
    def endpoints(scan: list[float | None], pose: np.ndarray) -> np.ndarray:
        angles = np.linspace(-math.pi, math.pi, len(scan), endpoint=False)
        return np.array([
            pose[:2] + distance * np.array([
                math.cos(pose[2] + angle), math.sin(pose[2] + angle)])
            for angle, distance in zip(angles, scan)
            if distance is not None and .14 < distance < 2.2
        ], dtype=float).reshape(-1, 2)

    def initialize(self, scan: list[float | None], pose: np.ndarray) -> dict:
        self.points = self.endpoints(scan, pose)
        self.last = {"status": "mapped" if len(self.points) >= 20 else "weak_initial_scan",
                     "map_points": len(self.points)}
        return self.last

    def correct(self, scan: list[float | None], predicted: np.ndarray) -> tuple[np.ndarray, dict]:
        if self.points is None:
            return predicted, self.initialize(scan, predicted)
        if len(self.points) < 20:
            return predicted, {"status": "weak_map"}
        local = self.endpoints(scan, np.zeros(3))
        if len(local) < 20:
            return predicted, {"status": "weak_scan", "scan_points": len(local)}
        tree = cKDTree(self.points)
        rotation = np.array([[math.cos(predicted[2]), -math.sin(predicted[2])],
                             [math.sin(predicted[2]), math.cos(predicted[2])]])
        rotated = local @ rotation.T

        # The IMU supplies heading; register only translation. Trim bad returns
        # from moved furniture/people and unseen surfaces. A weak odometry prior
        # prevents large jumps across the repeated wall geometry.
        def residual(offset: np.ndarray) -> float:
            distances, _ = tree.query(rotated + predicted[:2] + offset)
            nearest = np.sort(distances)[:max(16, int(.7 * len(distances)))]
            return float(np.mean(np.minimum(nearest, .12) ** 2) +
                         .08 * np.sum(offset ** 2))

        answer = minimize(residual, np.zeros(2), method="Nelder-Mead",
                          options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 100})
        offset = np.asarray(answer.x, dtype=float)
        distances, _ = tree.query(rotated + predicted[:2] + offset)
        inliers = distances < .10
        quality = {"scan_points": len(local), "inlier_fraction": round(float(np.mean(inliers)), 3),
                   "median_residual_m": round(float(np.median(distances)), 4),
                   "correction_m": round(float(np.linalg.norm(offset)), 4)}
        if np.linalg.norm(offset) > .13 or np.mean(inliers) < .48:
            self.last = {"status": "registration_rejected", **quality}
            return predicted, self.last
        corrected = predicted.copy()
        corrected[:2] += offset
        self.last = {"status": "registered", **quality}
        return corrected, self.last
