"""Observed sparse-view abstention for the negative X/Y camera quadrant."""

from __future__ import annotations

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.selective_template_rgb_goal_v42b import (
    SelectiveTemplateRgbGoal as ReacquiringSelector,
)


NEGATIVE_CAMERA_X_BOUND_M = -0.05


class SelectiveTemplateRgbGoal(ReacquiringSelector):
    """Use calibrated RGB when sparse template shifts are viewed from negative X."""

    def capture(self, native_qpos: np.ndarray, time_s: float,
                *, pose_healthy: bool = True) -> tuple[dict, np.ndarray]:
        event, rgb = super().capture(native_qpos, time_s, pose_healthy=pose_healthy)
        abstain = bool(event["used_for_control"] and event["template_used_for_control"] and
                       24 <= event["orange_pixels"] < 64 and
                       event["camera_world_xyz_m"][0] < NEGATIVE_CAMERA_X_BOUND_M and
                       event["camera_world_xyz_m"][1] < 0.0)
        event["negative_camera_sparse_abstention"] = abstain
        event["negative_camera_sparse_rule"] = (
            "24 <= observed orange pixels < 64, capture camera world X < -0.05 m and Y < 0 m; "
            "retain calibrated RGB instead of sparse rendered-template correction")
        if abstain:
            calibrated = np.asarray(event["pre_template_goal_world_xyz_m"], dtype=float)
            event["estimated_goal_world_xyz_m"] = calibrated.tolist()
            event["template_used_for_control"] = False
            self.last_goal_world = calibrated
        return event, rgb
