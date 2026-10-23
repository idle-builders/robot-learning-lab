"""Actor-only, abstaining side cue from elevated RGB-D obstacle evidence.

This nominates a side for an information-gathering view; it never clears a
base step. The motion planner and independent range monitor retain that job.
"""

from __future__ import annotations

import math

import numpy as np

from depth_voxel import MAX_AGE_S, VOXEL_M, DepthVoxelMap


def choose_side(voxel: DepthVoxelMap, pose: np.ndarray,
                target_xy: np.ndarray, time_s: float) -> dict:
    forward = np.asarray(target_xy, dtype=float) - pose[:2]
    target_distance = float(np.linalg.norm(forward))
    if target_distance < .15:
        return {"side": None, "reason": "target_too_close"}
    forward /= target_distance
    left = np.array([-forward[1], forward[0]])
    xy_cells = {(ix, iy) for (ix, iy, iz), hit_s in voxel.occupied.items()
                if hit_s >= time_s - min(MAX_AGE_S, 1.6)
                and hit_s >= voxel.free.get((ix, iy, iz), -math.inf)
                and .30 <= (iz + .5) * VOXEL_M <= .43}
    if not xy_cells:
        return {"side": None, "reason": "no_fresh_high_surface",
                "high_cells": 0}
    points = np.asarray([[(ix + .5) * VOXEL_M,
                          (iy + .5) * VOXEL_M]
                         for ix, iy in sorted(xy_cells)], dtype=float)
    offsets = points - pose[:2]
    along = offsets @ forward
    lateral = offsets @ left
    selected = lateral[(along >= .12) & (along <= min(.85,
                    target_distance + .30)) & (np.abs(lateral) <= .65)]
    positive = int(np.count_nonzero(selected > .05))
    negative = int(np.count_nonzero(selected < -.05))
    strong = positive + negative
    dominance = max(positive, negative) / strong if strong else 0.
    median = float(np.median(selected)) if len(selected) else None
    side = None
    reason = "ambiguous_high_surface"
    if (strong >= 8 and dominance >= .75 and median is not None and
            abs(median) >= .08):
        side = "right" if positive > negative else "left"
        reason = "elevated_blocker_opposite_side"
    return {"side": side, "reason": reason,
            "target_distance_m": round(target_distance, 4),
            "high_cells": len(xy_cells),
            "forward_high_cells": len(selected),
            "high_cells_left": positive,
            "high_cells_right": negative,
            "dominant_fraction": round(dominance, 3),
            "median_lateral_m": (None if median is None
                                 else round(median, 4))}
