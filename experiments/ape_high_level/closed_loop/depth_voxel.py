"""Small actor-only time-stamped 3-D ray map for active head-depth experiments.

Only a depth image, its calibrated camera pose and capture time enter this
module. Unknown space remains unknown, including behind a measured surface.
This does not infer an object's interior from a visible point cloud.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


VOXEL_M = .04
MAX_AGE_S = 4.0


class DepthVoxelMap:
    def __init__(self) -> None:
        self.free: dict[tuple[int, int, int], float] = {}
        self.occupied: dict[tuple[int, int, int], float] = {}
        self.frames = 0
        self.last_time_s: float | None = None

    @staticmethod
    def cell(xyz: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.floor(np.asarray(xyz) / VOXEL_M).astype(int))

    def observe(self, depth: np.ndarray,
                camera_pose: tuple[np.ndarray, np.ndarray],
                time_s: float, *, fovy_deg: float, stride: int = 8,
                max_depth_m: float = 1.4,
                self_hit: Callable[[np.ndarray], bool] | None = None,
                self_masked: bool = False) -> dict:
        height, width = depth.shape
        focal = height / (2 * math.tan(math.radians(fovy_deg) / 2))
        rows, cols = np.mgrid[stride // 2:height:stride,
                              stride // 2:width:stride]
        d = depth[rows, cols]
        valid = np.isfinite(d) & (d > .08) & (d < max_depth_m)
        rays = np.stack(((cols - width / 2) / focal,
                         -(rows - height / 2) / focal,
                         -np.ones_like(d)), axis=-1)[valid]
        origin, rotation = camera_pose
        endpoints = origin + (rays * d[valid, None]) @ rotation.T
        used = 0
        for endpoint in endpoints:
            displacement = endpoint - origin
            length = float(np.linalg.norm(displacement))
            if length < (.08 if self_masked else .12):
                continue
            # The selected body's near shell can appear in the head frame.
            # Do not pretend a self-hit clears space behind it.
            if (not self_masked and
                    float(np.linalg.norm(endpoint[:2] - origin[:2])) < .12):
                continue
            if self_hit is not None and self_hit(endpoint):
                continue
            for distance in np.arange(.04, max(.04, length - .06), VOXEL_M):
                cell = self.cell(origin + (distance / length) * displacement)
                self.free[cell] = time_s
            cell = self.cell(endpoint)
            self.occupied[cell] = time_s
            used += 1
        self.frames += 1
        self.last_time_s = time_s
        self.free = {cell: stamp for cell, stamp in self.free.items()
                     if time_s - stamp <= MAX_AGE_S}
        self.occupied = {cell: stamp for cell, stamp in self.occupied.items()
                         if time_s - stamp <= MAX_AGE_S}
        return {"depth_rays_used": used, "frames": self.frames,
                "free_voxels": len(self.free),
                "occupied_voxels": len(self.occupied)}

    def state(self, xyz: np.ndarray, time_s: float) -> str:
        cell = self.cell(xyz)
        free_s = self.free.get(cell, -math.inf)
        hit_s = self.occupied.get(cell, -math.inf)
        if hit_s >= time_s - MAX_AGE_S and hit_s >= free_s:
            return "occupied"
        if free_s >= time_s - MAX_AGE_S:
            return "free"
        return "unknown"

    def upper_occupied_xy(self, time_s: float, *, max_age_s: float = 1.6,
                          z_min_m: float = .18,
                          z_max_m: float = .43) -> np.ndarray:
        """Fresh upper obstacle marks, projected into local 2-D route space.

        Returns obstacle *evidence*, not known-free space. A current range
        guard still owns motion stop. Multiple occupied heights collapse to
        one 4 cm XY cell so route search is bounded.
        """
        cells = {(x, y) for (x, y, z), stamp in self.occupied.items()
                 if stamp >= time_s - min(MAX_AGE_S, max_age_s)
                 and stamp >= self.free.get((x, y, z), -math.inf)
                 and z_min_m <= (z + .5) * VOXEL_M <= z_max_m}
        return np.asarray([[(x + .5) * VOXEL_M,
                            (y + .5) * VOXEL_M] for x, y in sorted(cells)],
                          dtype=float).reshape(-1, 2)

    def buffered_upper_coverage(self, pose: np.ndarray,
                                time_s: float) -> dict:
        """Conservative neutral upper-body coverage for a proposed base pose.

        This is an observability query, not an articulated collision checker.
        The sample volume includes about 5 cm around the selected shell's
        upper arm/torso reach. A caller must not turn "unknown" into "free".
        """
        c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
        counts = {"free": 0, "occupied": 0, "unknown": 0}
        for z in (.18, .22, .26, .30, .34, .38):
            for dx in (-.06, 0., .06):
                for dy in (-.20, -.16, -.10, 0., .10, .16, .20):
                    xyz = np.array([pose[0] + c * dx - s * dy,
                                    pose[1] + s * dx + c * dy, z])
                    counts[self.state(xyz, time_s)] += 1
        return {"status": ("occupied" if counts["occupied"] else
                           "unknown" if counts["unknown"] else "free"),
                **counts, "samples": sum(counts.values()),
                "age_s": (None if self.last_time_s is None else
                          round(time_s - self.last_time_s, 3))}
