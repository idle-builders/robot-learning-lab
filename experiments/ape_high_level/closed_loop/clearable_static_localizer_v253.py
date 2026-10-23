"""V253: remove promoted surfaces contradicted by repeated clear rays.

Only promoted points can be removed. The origin map remains an anchor. A ray
must align within half a scan beam and terminate at least 12 cm beyond the
point on two pose-safe registered scans. Closer hits are occlusion, not free.
"""

from __future__ import annotations

import math

import numpy as np

from persistent_static_localizer_v251 import (
    MOTION_SAFE_CORRECTION_M, PersistentStaticScanLocalizer)


ANGULAR_TOLERANCE_RAD = math.radians(2.5)
FREE_BEYOND_POINT_M = .12
CONTRADICTIONS_TO_CLEAR = 2


class ClearableStaticScanLocalizer(PersistentStaticScanLocalizer):
    def __init__(self) -> None:
        super().__init__()
        self.origin_count = 0
        self.promoted_contradictions: list[int] = []
        self.evicted_total = 0

    def initialize(self, scan: list[float | None], pose: np.ndarray) -> dict:
        quality = super().initialize(scan, pose)
        self.origin_count = 0 if self.points is None else len(self.points)
        self.promoted_contradictions = []
        self.evicted_total = 0
        return quality

    def correct(self, scan: list[float | None], predicted: np.ndarray
                ) -> tuple[np.ndarray, dict]:
        previous_promoted = len(self.promoted_contradictions)
        corrected, quality = super().correct(scan, predicted)
        if self.points is None:
            return corrected, quality
        newly_promoted = len(self.points) - self.origin_count - previous_promoted
        if newly_promoted < 0:
            raise AssertionError("promoted-point tracking lost alignment")
        self.promoted_contradictions.extend([0] * newly_promoted)
        if (quality["status"] != "registered" or
                np.linalg.norm(corrected[:2] - predicted[:2]) >
                MOTION_SAFE_CORRECTION_M):
            return corrected, {**quality,
                               "evicted_total": self.evicted_total,
                               "tracked_promoted_points": len(
                                   self.promoted_contradictions)}
        angles = np.linspace(-math.pi, math.pi, len(scan), endpoint=False)
        old_points = self.points[self.origin_count:
                                 self.origin_count + previous_promoted]
        evict = []
        contradicted = 0
        for index, point in enumerate(old_points):
            ray = point - corrected[:2]
            expected_range = float(np.linalg.norm(ray))
            bearing = math.atan2(ray[1], ray[0]) - corrected[2]
            differences = np.arctan2(np.sin(angles - bearing),
                                     np.cos(angles - bearing))
            beam = int(np.argmin(np.abs(differences)))
            if abs(differences[beam]) > ANGULAR_TOLERANCE_RAD:
                continue
            observed = scan[beam]
            if observed is None:
                continue  # no finite free-space endpoint
            if observed > expected_range + FREE_BEYOND_POINT_M:
                self.promoted_contradictions[index] += 1
                contradicted += 1
            elif abs(observed - expected_range) < .08:
                self.promoted_contradictions[index] = 0
            if self.promoted_contradictions[index] >= CONTRADICTIONS_TO_CLEAR:
                evict.append(index)
        if evict:
            keep = np.ones(len(self.points), dtype=bool)
            for index in evict:
                keep[self.origin_count + index] = False
            self.points = self.points[keep]
            for index in reversed(evict):
                del self.promoted_contradictions[index]
            self.evicted_total += len(evict)
        quality = {**quality,
                   "static_map_points": len(self.points),
                   "contradicted_promoted_this_scan": contradicted,
                   "evicted_this_scan": len(evict),
                   "evicted_total": self.evicted_total,
                   "tracked_promoted_points": len(
                       self.promoted_contradictions)}
        self.last = quality
        return corrected, quality
