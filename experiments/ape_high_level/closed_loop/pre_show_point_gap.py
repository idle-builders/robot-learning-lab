"""Continuous body-to-range-endpoint diagnostic for candidate ape steps."""

from __future__ import annotations

import numpy as np

from robot_geometry import RobotGeometry
from run_pre_show_reserve_screen_v100 import COMMAND_STEP_M, DIRECTIONS


def endpoints(rays: list[dict]) -> np.ndarray:
    return np.asarray([
        np.asarray(ray["actor_origin_m"], dtype=float) +
        float(ray["measured_range_m"]) *
        np.asarray(ray["actor_direction"], dtype=float)
        for ray in rays if ray["endpoint_hit"]], dtype=float).reshape(-1, 3)


def minimum_swept_point_gap(robot: RobotGeometry, actor_qpos: np.ndarray,
                            start_pose: np.ndarray, commands: tuple[str, ...],
                            points: np.ndarray) -> dict:
    """Check measured endpoints, not unobserved free space or real solids.

    All commanded displacement is assumed to execute; the historical V100
    evaluator applies 0.55 of that displacement. This diagnostic is a veto
    only. Point samples cannot certify an empty swept volume.
    """
    if not len(points):
        return {"minimum_gap_m": None, "closest_step": None,
                "closest_fraction": None, "point_count": 0}
    current = np.asarray(start_pose, dtype=float).copy()
    minimum = (float("inf"), None, None)
    for step_index, command in enumerate(commands):
        future = current.copy()
        future[:2] += COMMAND_STEP_M * np.asarray(DIRECTIONS[command])
        for fraction in (0., .25, .5, .75, 1.):
            pose = current + fraction * (future - current)
            robot.set_state(robot.pose_qpos(actor_qpos, pose))
            gap = float(np.min(robot.solid_distances(
                points, robot.primitives())))
            if gap < minimum[0]:
                minimum = (gap, step_index, fraction)
        current = future
    return {"minimum_gap_m": round(minimum[0], 5),
            "closest_step": minimum[1],
            "closest_fraction": minimum[2],
            "point_count": len(points)}
