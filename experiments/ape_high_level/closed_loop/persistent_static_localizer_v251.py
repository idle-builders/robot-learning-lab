"""V251: grow the static scan map only from repeated, pose-safe observations.

Candidate new surfaces need three distinct registered scans. Map insertion is
restricted to corrections whose norm alone guarantees the controller's
separate 1 cm lateral / +5 mm forward motion gate will accept the pose.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from static_supported_localizer_v233 import StaticSupportedScanLocalizer


MOTION_SAFE_CORRECTION_M = .0045
NOVEL_FROM_MAP_M = .08
PERSISTENCE_RADIUS_M = .04
PERSISTENCE_SCANS = 3
MAX_MISSED_SCANS = 6


class PersistentStaticScanLocalizer(StaticSupportedScanLocalizer):
    def __init__(self) -> None:
        super().__init__()
        self.scan_index = 0
        self.pending: list[dict] = []
        self.promoted_total = 0

    def initialize(self, scan: list[float | None], pose: np.ndarray) -> dict:
        self.scan_index = 0
        self.pending = []
        self.promoted_total = 0
        return super().initialize(scan, pose)

    def correct(self, scan: list[float | None], predicted: np.ndarray
                ) -> tuple[np.ndarray, dict]:
        corrected, quality = super().correct(scan, predicted)
        self.scan_index += 1
        quality = {**quality, "static_map_points": 0 if self.points is None else
                   len(self.points), "pending_new_surfaces": len(self.pending),
                   "promoted_total": self.promoted_total}
        if (quality["status"] != "registered" or self.points is None or
                np.linalg.norm(corrected[:2] - predicted[:2]) >
                MOTION_SAFE_CORRECTION_M):
            self.last = quality
            return corrected, quality

        local = self.endpoints(scan, np.zeros(3))
        rotation = np.array([[math.cos(corrected[2]), -math.sin(corrected[2])],
                             [math.sin(corrected[2]), math.cos(corrected[2])]])
        world = local @ rotation.T + corrected[:2]
        distances, _ = cKDTree(self.points).query(world)
        novel = world[distances > NOVEL_FROM_MAP_M]
        self.pending = [candidate for candidate in self.pending
                        if self.scan_index - candidate["last"] <= MAX_MISSED_SCANS]
        for point in novel:
            if self.pending:
                deltas = np.array([np.linalg.norm(point - candidate["point"])
                                   for candidate in self.pending])
                index = int(np.argmin(deltas))
                if deltas[index] <= PERSISTENCE_RADIUS_M:
                    candidate = self.pending[index]
                    if candidate["last"] != self.scan_index:
                        candidate["point"] = .7 * candidate["point"] + .3 * point
                        candidate["seen"] += 1
                        candidate["last"] = self.scan_index
                    continue
            self.pending.append({"point": point.copy(), "seen": 1,
                                 "last": self.scan_index})
        promoted = []
        retained = []
        for candidate in self.pending:
            if candidate["seen"] >= PERSISTENCE_SCANS:
                nearest, _ = cKDTree(self.points).query(candidate["point"])
                if nearest > PERSISTENCE_RADIUS_M:
                    promoted.append(candidate["point"])
            else:
                retained.append(candidate)
        if promoted:
            self.points = np.vstack((self.points, np.asarray(promoted)))
            self.promoted_total += len(promoted)
        self.pending = retained
        quality = {**quality,
                   "static_map_points": len(self.points),
                   "pending_new_surfaces": len(self.pending),
                   "promoted_this_scan": len(promoted),
                   "promoted_total": self.promoted_total,
                   "map_insertion_motion_guarantee_m":
                       MOTION_SAFE_CORRECTION_M}
        self.last = quality
        return corrected, quality
