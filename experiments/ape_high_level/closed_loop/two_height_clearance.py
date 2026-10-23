"""Actor-only short swept-body clearance from two range planes and head depth.

This remains a point-distance hypothesis, not a proof of unseen free volume.
The caller must retain independent motion monitoring and outcome checks.
"""

from __future__ import annotations

import math

import numpy as np

from depth_voxel import VOXEL_M, DepthVoxelMap
from footprint_route import scan_points
from robot_geometry import RobotGeometry


POINT_MARGIN_M = .065
MAX_UPPER_AGE_S = 6.
MAX_LOW_DEPTH_AGE_S = 1.6


def points_from_scan(ranges: list[float | None], pose: np.ndarray,
                     height_m: float, current_solids: tuple) -> np.ndarray:
    xy = scan_points(ranges, pose)
    xyz = np.column_stack((xy, np.full(len(xy), height_m)))
    return np.asarray([
        point for point in xyz
        if RobotGeometry.solid_distance(point, current_solids) >= -.005
    ], dtype=float).reshape(-1, 3)


class TwoHeightClearance:
    def __init__(self, robot: RobotGeometry, actor_qpos: np.ndarray,
                 pose: np.ndarray, range_11: list[float | None],
                 range_4: list[float | None], voxel: DepthVoxelMap,
                 time_s: float, *, range_8: list[float | None] | None = None,
                 head_low: bool = False) -> None:
        self.robot = robot
        self.actor_qpos = actor_qpos.copy()
        self.pose = pose.copy()
        robot.set_state(robot.pose_qpos(actor_qpos, pose))
        current = robot.primitives()
        self.sources = {
            "range_11": points_from_scan(range_11, pose, .11, current),
            "range_4": points_from_scan(range_4, pose, .04, current),
            "head_upper": np.asarray([
                [(ix + .5) * VOXEL_M, (iy + .5) * VOXEL_M,
                 (iz + .5) * VOXEL_M]
                for (ix, iy, iz), hit_s in voxel.occupied.items()
                if (hit_s >= voxel.free.get((ix, iy, iz), -math.inf)
                    and time_s - hit_s <= MAX_UPPER_AGE_S
                    and .18 <= (iz + .5) * VOXEL_M <= .43)
            ], dtype=float).reshape(-1, 3)}
        if range_8 is not None:
            self.sources["range_8"] = points_from_scan(range_8, pose, .08,
                                                        current)
        if head_low:
            self.sources["head_low"] = np.asarray([
                [(ix + .5) * VOXEL_M, (iy + .5) * VOXEL_M,
                 (iz + .5) * VOXEL_M]
                for (ix, iy, iz), hit_s in voxel.occupied.items()
                if (hit_s >= voxel.free.get((ix, iy, iz), -math.inf)
                    and time_s - hit_s <= MAX_LOW_DEPTH_AGE_S
                    and .04 <= (iz + .5) * VOXEL_M <= .16)
            ], dtype=float).reshape(-1, 3)
        self.all_points = np.concatenate(tuple(self.sources.values()))

    def edge_gap(self, start_xy: np.ndarray, end_xy: np.ndarray,
                 yaw: float) -> float:
        gap = math.inf
        delta = np.asarray(end_xy) - np.asarray(start_xy)
        for fraction in (.25, .5, .75, 1.):
            future = self.pose.copy()
            future[:2] = np.asarray(start_xy) + fraction * delta
            future[2] = yaw
            self.robot.set_state(self.robot.pose_qpos(self.actor_qpos, future))
            solids = self.robot.primitives()
            if len(self.all_points):
                gap = min(gap, float(np.min(RobotGeometry.solid_distances(
                    self.all_points, solids))))
        return gap

    def source_counts(self) -> dict[str, int]:
        return {name: len(points) for name, points in self.sources.items()}
