"""Conservative actor-side handoff invitation zone on the selected ape.

Only own joint/pose state and public robot-relative invitation points enter.
The point reserve is a development assumption, not a person-arm clearance or
permission to receive physical contact. Environment free space is separate.
"""

from __future__ import annotations

import numpy as np

from articulated_person_scene import FAR_HAND_XYZ_M, NEAR_HAND_XYZ_M
from robot_geometry import RobotGeometry


INVITATION_BODY_RESERVE_M = .20
ZONE_POINTS = {"near": NEAR_HAND_XYZ_M, "far": FAR_HAND_XYZ_M}


def choose_invitation_zone(robot: RobotGeometry,
                           actor_qpos: np.ndarray,
                           estimated_pose: np.ndarray) -> dict:
    robot.set_state(RobotGeometry.pose_qpos(actor_qpos, estimated_pose))
    body = robot.primitives()
    scores = {name: round(RobotGeometry.solid_distance(point, body), 5)
              for name, point in ZONE_POINTS.items()}
    viable = [name for name, gap in scores.items()
              if gap >= INVITATION_BODY_RESERVE_M]
    chosen = max(viable, key=lambda name: scores[name]) if viable else None
    return {"chosen_zone": chosen,
            "actor_zone_point_m": ZONE_POINTS[chosen].tolist()
            if chosen is not None else None,
            "actor_endpoint_body_gaps_m": scores,
            "required_endpoint_body_reserve_m": INVITATION_BODY_RESERVE_M,
            "scene_free_space_verified": False,
            "person_arm_sweep_verified": False,
            "physical_contact_authorized": False,
            "social_action": ("request_far_offer_and_hold" if chosen == "far"
                              else "request_alternate_offer_or_reposition")}
