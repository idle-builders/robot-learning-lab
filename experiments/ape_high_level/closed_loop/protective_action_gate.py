"""Actor-only bounded low-point forecast for hold versus shoulder response.

The side alarm supplies a motion direction. This compares the selected
robot's own articulated collision primitives against observed range points
advanced at a conservative approach-speed bound. It never reads scene geoms.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from audit_protective_postures_v327 import self_clearance
from robot_geometry import RobotGeometry


MAX_APPROACH_M_S = .25
SHOULDER_TARGET_RAD = -.8
ACTION_DURATION_S = .4
FORECAST_S = .6
MIN_CLEARANCE_GAIN_M = .01


def nearby_low_points(points: np.ndarray, pose: np.ndarray,
                      side: int) -> np.ndarray:
    delta = points[:, :2] - pose[:2]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    x = c * delta[:, 0] + s * delta[:, 1]
    y = side * (-s * delta[:, 0] + c * delta[:, 1])
    return points[(np.abs(x) < .25) & (y > .15) & (y < .50) &
                  (points[:, 2] > .015) & (points[:, 2] < .16)]


def compare(points_now: np.ndarray, pose: np.ndarray,
            actor_qpos: np.ndarray, side_name: str,
            robot: RobotGeometry) -> dict:
    side = 1 if side_name == "left" else -1
    points = nearby_low_points(points_now, pose, side)
    if len(points) < 3:
        return {"status": "insufficient_3d_returns", "side": side_name,
                "side_points": len(points), "chosen": "hold"}
    lateral = np.array((-np.sin(pose[2]), np.cos(pose[2]), 0.)) * side
    base_qpos = robot.pose_qpos(actor_qpos, pose)
    address = int(robot.model.joint(f"{side_name}_shoulder_pitch").qposadr[0])
    gaps = {}
    self_gaps = {}
    for policy in ("hold", "shoulder_pitch"):
        smallest = float("inf")
        self_smallest = float("inf")
        for elapsed in np.linspace(0., FORECAST_S, 31):
            future_points = points - MAX_APPROACH_M_S * elapsed * lateral
            qpos = base_qpos.copy()
            if policy == "shoulder_pitch":
                qpos[address] = SHOULDER_TARGET_RAD * np.clip(
                    elapsed / ACTION_DURATION_S, 0., 1.)
            robot.set_state(qpos)
            smallest = min(smallest, float(np.min(
                robot.solid_distances(future_points, robot.primitives()))))
            self_smallest = min(self_smallest,
                                self_clearance(SimpleNamespace(
                                    model=robot.model, data=robot.data),
                                    side_name)[0])
        gaps[policy] = round(smallest, 5)
        self_gaps[policy] = round(self_smallest, 5)
    chosen = ("shoulder_pitch" if
              gaps["shoulder_pitch"] >= gaps["hold"] + MIN_CLEARANCE_GAIN_M
              and self_gaps["shoulder_pitch"] >= 0. else "hold")
    return {"status": "compared", "side": side_name,
            "side_points": len(points),
            "max_approach_m_s": MAX_APPROACH_M_S,
            "forecast_s": FORECAST_S,
            "minimum_gain_m": MIN_CLEARANCE_GAIN_M,
            "predicted_point_gap_m": gaps,
            "predicted_self_gap_m": self_gaps,
            "chosen": chosen}
