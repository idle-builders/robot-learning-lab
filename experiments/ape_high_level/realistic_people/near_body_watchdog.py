"""Small, independent low-range alarm for the eye-proxy experiment.

This consumes raw range and command timing only. It never reads an object ID,
owner choice, visual odometry, map, or evaluator geometry. A return that is
missing or invalid is unknown, not a distant surface.
"""

from __future__ import annotations

from collections import deque
import math
from statistics import median


class NearBodyWatchdog:
    def __init__(self) -> None:
        self.history: deque[tuple[float, float]] = deque(maxlen=3)
        self.latched = False

    def observe(self, low_ranges_m: list[float | None], time_s: float,
                preceding_translation_m: float) -> dict:
        valid = [float(value) if value is not None and math.isfinite(value)
                 and value > 0 else None for value in low_ranges_m]
        # Five adjacent 5-degree rays must agree on a near surface. The
        # nearest individual rays are too sensitive to one corrupted return.
        patches = [median(present) for start in range(len(valid) - 4)
                   if len(present := [value for value in valid[start:start + 5]
                                      if value is not None]) >= 4]
        if not patches:
            self.history.clear()
            return {"status": "unknown", "alarm": self.latched,
                    "reason": "no_coherent_low_patch", "patch_range_m": None}

        nearest_patch = min(patches)
        # A commanded displacement changes range even when the obstacle is
        # stationary. Require a fresh stationary history for the trend cue.
        if abs(preceding_translation_m) > 1e-4 or (
                self.history and time_s <= self.history[-1][0]):
            self.history.clear()
        self.history.append((time_s, nearest_patch))

        reason = None
        if nearest_patch <= .27:
            reason = "critical_near_return"
        elif nearest_patch <= .31 and len(self.history) == 3:
            (t0, r0), (t1, r1), (t2, r2) = self.history
            if t2 - t0 <= .8 and t1 - t0 <= .55 and t2 - t1 <= .55 and (
                    r0 - r1 >= .005 and r1 - r2 >= .005 and r0 - r2 >= .035):
                reason = "closing_while_stationary"
        if reason:
            self.latched = True
        return {"status": "alarm" if self.latched else "clear",
                "alarm": self.latched, "reason": reason,
                "patch_range_m": round(nearest_patch, 5),
                "stationary_samples": len(self.history),
                "time_s": time_s}

    def reset_after_recovery(self) -> None:
        self.latched = False
        self.history.clear()

    def reset_motion_history(self) -> None:
        self.history.clear()
