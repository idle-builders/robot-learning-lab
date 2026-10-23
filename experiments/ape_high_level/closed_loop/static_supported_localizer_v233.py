"""V233 candidate: exclude scan returns unsupported by the origin map.

This is a development ablation, not a qualified lidar or a permission to
relax the actor's separate motion-consistency gate.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from localization import ScanLocalizer


STATIC_SUPPORT_M = .08
MIN_STATIC_POINTS = 20


class StaticSupportedScanLocalizer(ScanLocalizer):
    def correct(self, scan: list[float | None], predicted: np.ndarray
                ) -> tuple[np.ndarray, dict]:
        if self.points is None:
            return predicted, self.initialize(scan, predicted)
        if len(self.points) < MIN_STATIC_POINTS:
            return predicted, {"status": "weak_map"}
        local = self.endpoints(scan, np.zeros(3))
        if len(local) < MIN_STATIC_POINTS:
            return predicted, {"status": "weak_scan", "scan_points": len(local)}
        tree = cKDTree(self.points)
        rotation = np.array([[math.cos(predicted[2]), -math.sin(predicted[2])],
                             [math.sin(predicted[2]), math.cos(predicted[2])]])
        rotated = local @ rotation.T
        prior_distance, _ = tree.query(rotated + predicted[:2])
        supported = prior_distance <= STATIC_SUPPORT_M
        count = int(supported.sum())
        base_quality = {"scan_points": len(local),
                        "static_support_points": count,
                        "static_support_fraction": round(float(np.mean(supported)), 3),
                        "static_support_radius_m": STATIC_SUPPORT_M}
        if count < MIN_STATIC_POINTS or np.mean(supported) < .48:
            self.last = {"status": "weak_static_scan", **base_quality}
            return predicted, self.last
        stable = rotated[supported]

        def residual(offset: np.ndarray) -> float:
            distances, _ = tree.query(stable + predicted[:2] + offset)
            nearest = np.sort(distances)[:max(16, int(.7 * len(distances)))]
            return float(np.mean(np.minimum(nearest, .12) ** 2) +
                         .08 * np.sum(offset ** 2))

        answer = minimize(residual, np.zeros(2), method="Nelder-Mead",
                          options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 100})
        offset = np.asarray(answer.x, dtype=float)
        distances, _ = tree.query(stable + predicted[:2] + offset)
        inliers = distances < .10
        quality = {**base_quality,
                   "inlier_fraction": round(float(np.mean(inliers)), 3),
                   "median_residual_m": round(float(np.median(distances)), 4),
                   "correction_m": round(float(np.linalg.norm(offset)), 4)}
        if np.linalg.norm(offset) > .13 or np.mean(inliers) < .48:
            self.last = {"status": "registration_rejected", **quality}
            return predicted, self.last
        corrected = predicted.copy()
        corrected[:2] += offset
        self.last = {"status": "registered", **quality}
        return corrected, self.last
