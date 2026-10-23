"""V257: keep promoted scan anchors short-lived unless reobserved.

The first map remains an origin anchor. New surfaces need V254's repeated
viewpoint evidence but are temporary: after three scans without a compatible
return they leave the registration map. This is an occupancy-memory candidate,
not proof that a lingering object is static while it is present.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from persistent_static_localizer_v251 import MOTION_SAFE_CORRECTION_M
from viewpoint_static_localizer_v254 import ViewpointStaticScanLocalizer


MAX_UNSEEN_SCANS = 3
REOBSERVATION_RADIUS_M = .06


class ExpiringViewpointScanLocalizer(ViewpointStaticScanLocalizer):
    def __init__(self) -> None:
        super().__init__()
        self.origin_count = 0
        self.promoted_last_seen: list[int] = []
        self.expired_total = 0

    def initialize(self, scan: list[float | None], pose: np.ndarray) -> dict:
        quality = super().initialize(scan, pose)
        self.origin_count = 0 if self.points is None else len(self.points)
        self.promoted_last_seen = []
        self.expired_total = 0
        return quality

    def correct(self, scan: list[float | None], predicted: np.ndarray
                ) -> tuple[np.ndarray, dict]:
        expired = 0
        if self.points is not None:
            if len(self.points) - self.origin_count != len(self.promoted_last_seen):
                raise AssertionError("temporary-map metadata lost alignment")
            keep_learned = np.array([
                self.scan_index - last < MAX_UNSEEN_SCANS
                for last in self.promoted_last_seen], dtype=bool)
            expired = int((~keep_learned).sum())
            if expired:
                keep = np.r_[np.ones(self.origin_count, dtype=bool),
                             keep_learned]
                self.points = self.points[keep]
                self.promoted_last_seen = [last for last, retained in zip(
                    self.promoted_last_seen, keep_learned) if retained]
                self.expired_total += expired
        old_count = len(self.promoted_last_seen)
        corrected, quality = super().correct(scan, predicted)
        if self.points is None:
            return corrected, quality
        added = len(self.points) - self.origin_count - old_count
        if added < 0:
            raise AssertionError("temporary-map promotion lost alignment")
        self.promoted_last_seen.extend([self.scan_index] * added)
        if (quality["status"] == "registered" and
                np.linalg.norm(corrected[:2] - predicted[:2]) <=
                MOTION_SAFE_CORRECTION_M and
                self.promoted_last_seen):
            local = self.endpoints(scan, np.zeros(3))
            rotation = np.array([[math.cos(corrected[2]), -math.sin(corrected[2])],
                                 [math.sin(corrected[2]), math.cos(corrected[2])]])
            world = local @ rotation.T + corrected[:2]
            if len(world):
                distances, _ = cKDTree(world).query(
                    self.points[self.origin_count:])
                for index, distance in enumerate(distances):
                    if distance <= REOBSERVATION_RADIUS_M:
                        self.promoted_last_seen[index] = self.scan_index
        quality = {**quality,
                   "static_map_points": len(self.points),
                   "temporary_promoted_points": len(self.promoted_last_seen),
                   "expired_this_scan": expired,
                   "expired_total": self.expired_total,
                   "max_unseen_scans": MAX_UNSEEN_SCANS}
        self.last = quality
        return corrected, quality
