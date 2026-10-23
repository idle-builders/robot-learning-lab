"""Actor-only swept-route decision for a provisional owner meeting pose.

The map contains only time-stamped camera/range evidence. An unknown cell is
never converted into permission to move. This module does not know evaluator
obstacles, owner future motion or selected-body action dynamics.
"""

from __future__ import annotations

import math

import numpy as np

from depth_voxel import DepthVoxelMap
from robot_geometry import RobotGeometry
from run_pre_show_reserve_screen_v100 import COMMAND_STEP_M, DIRECTIONS


POSE_OFFSETS = {
    "estimated": (0., 0., 0.),
    "left_1cm": (0., .01, 0.),
    "right_1cm": (0., -.01, 0.),
    "yaw_left_2deg": (0., 0., math.radians(2.)),
    "yaw_right_2deg": (0., 0., -math.radians(2.)),
}
ROUTE_FREE_AGE_S = .4


def freshness_filtered(voxel_map: DepthVoxelMap, time_s: float,
                       free_age_s: float) -> DepthVoxelMap:
    """Retain old occupied evidence; require recent free rays for motion.

    No source has established that the owner, hand block or low prop is static.
    Free evidence older than the route horizon is therefore context only.
    """
    recent = DepthVoxelMap()
    recent.free = {cell: stamp for cell, stamp in voxel_map.free.items()
                   if stamp >= time_s - free_age_s}
    recent.occupied = voxel_map.occupied.copy()
    recent.last_time_s = voxel_map.last_time_s
    return recent


def swept_step(robot: RobotGeometry, voxel_map: DepthVoxelMap,
               actor_qpos: np.ndarray, pose: np.ndarray, command: str,
               time_s: float) -> dict:
    future = pose.copy()
    future[:2] += COMMAND_STEP_M * np.asarray(DIRECTIONS[command])
    return {
        "command": command,
        "upper": robot.swept_exterior_coverage(
            voxel_map, actor_qpos, pose, future, time_s,
            margin_m=.05, z_min_m=.18, z_max_m=.43),
        "lower": robot.swept_exterior_coverage(
            voxel_map, actor_qpos, pose, future, time_s,
            margin_m=.05, z_min_m=.02, z_max_m=.18),
    }


def route_screen(robot: RobotGeometry, voxel_map: DepthVoxelMap,
                 actor_qpos: np.ndarray, estimated_pose: np.ndarray,
                 sequences: dict[str, tuple[str, ...]], time_s: float,
                 *, pose_offsets: dict[str, tuple[float, float, float]]
                 = POSE_OFFSETS) -> dict:
    """Require every sampled swept shell to be observed free.

    Pose offsets are diagnostic samples, not a calibrated covariance bound.
    The result is only a map-coverage screen; it cannot authorize a motion
    without a bound on pose error, dynamic occupancy and skill response.
    """
    result = {}
    for name, commands in sequences.items():
        variants = {}
        for label, offset in pose_offsets.items():
            current = estimated_pose + np.asarray(offset)
            checked = []
            for command in commands:
                step = swept_step(robot, voxel_map, actor_qpos,
                                  current, command, time_s)
                checked.append(step)
                if (step["upper"]["status"] != "free" or
                        step["lower"]["status"] != "free"):
                    break
                current = current.copy()
                current[:2] += COMMAND_STEP_M * np.asarray(DIRECTIONS[command])
            variants[label] = {
                "status": ("observed_free" if len(checked) == len(commands)
                           and all(step[height]["status"] == "free"
                                   for step in checked
                                   for height in ("upper", "lower"))
                           else "occupied" if any(
                               step[height]["status"] == "occupied"
                               for step in checked
                               for height in ("upper", "lower"))
                           else "unknown"),
                "checked_steps": checked,
            }
        result[name] = {
            "coverage_status": (
                "observed_free" if all(v["status"] == "observed_free"
                                       for v in variants.values())
                else "occupied" if any(v["status"] == "occupied"
                                       for v in variants.values())
                else "unknown"),
            "pose_samples": variants,
        }
    return result
