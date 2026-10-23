"""Opened diagnostic: partially restore a negative-camera sparse correction."""

from __future__ import annotations

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.selective_template_rgb_goal_v47 import (
    SelectiveTemplateRgbGoal as V47Selector,
)


class SelectiveTemplateRgbGoal(V47Selector):
    negative_sparse_blend: float = 0.0

    def capture(self, native_qpos: np.ndarray, time_s: float,
                *, pose_healthy: bool = True) -> tuple[dict, np.ndarray]:
        event, rgb = super().capture(native_qpos, time_s, pose_healthy=pose_healthy)
        blend = self.negative_sparse_blend
        if not 0 <= blend <= 1:
            raise ValueError("Negative sparse diagnostic blend must be in [0, 1]")
        event["negative_sparse_blend_diagnostic"] = blend
        if not event.get("negative_camera_sparse_abstention"):
            return event, rgb
        calibrated = np.asarray(event["pre_template_goal_world_xyz_m"], dtype=float)
        correction_mm = np.asarray(event["template_correction_xy_mm"], dtype=float)
        selected = calibrated.copy()
        selected[:2] += blend * correction_mm / 1000
        event["estimated_goal_world_xyz_m"] = selected.tolist()
        event["template_used_for_control"] = blend > 0
        event["negative_sparse_blend_rule"] = (
            "Opened diagnostic only: interpolate calibrated RGB and the rendered-template "
            "correction in the V47 negative-camera sparse-abstention region")
        self.last_goal_world = selected
        return event, rgb
