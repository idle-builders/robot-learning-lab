"""Opened diagnostic: blend positive-camera sparse template correction."""

from __future__ import annotations

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.selective_template_rgb_goal_v45 import (
    SelectiveTemplateRgbGoal as V45Selector,
)


class SelectiveTemplateRgbGoal(V45Selector):
    sparse_positive_blend: float = 1.0

    def capture(self, native_qpos: np.ndarray, time_s: float,
                *, pose_healthy: bool = True) -> tuple[dict, np.ndarray]:
        event, rgb = super().capture(native_qpos, time_s, pose_healthy=pose_healthy)
        blend = self.sparse_positive_blend
        if not 0 <= blend <= 1:
            raise ValueError("Sparse positive-camera blend must be in [0, 1]")
        event["sparse_positive_blend_diagnostic"] = blend
        if not (event["used_for_control"] and event["template_used_for_control"] and
                24 <= event["orange_pixels"] < 64 and
                event["camera_world_xyz_m"][0] >= 0):
            return event, rgb
        calibrated = np.asarray(event["pre_template_goal_world_xyz_m"], dtype=float)
        template = np.asarray(event["estimated_goal_world_xyz_m"], dtype=float)
        selected = calibrated + blend * (template - calibrated)
        event["estimated_goal_world_xyz_m"] = selected.tolist()
        event["template_used_for_control"] = blend > 0
        event["sparse_positive_blend_rule"] = (
            "Opened diagnostic only: 24-63 toy pixels from nonnegative camera X; "
            "interpolate calibrated RGB and V45 template goals")
        self.last_goal_world = selected
        return event, rgb
