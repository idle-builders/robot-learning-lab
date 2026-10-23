"""Use rendered-template correction only for sparse, materially shifted RGB goals.

Both alternatives see the same image and camera/robot pose. The threshold was
chosen from opened V31 failures before any V32 native evaluation.
"""

from __future__ import annotations

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.template_virtual_rgb_goal_v30 import (
    TemplateMatchedVirtualRgbGoal,
)


class SelectiveTemplateRgbGoal(TemplateMatchedVirtualRgbGoal):
    """Keep the calibrated goal unless the RGB evidence supports correction."""

    def capture(self, native_qpos: np.ndarray, time_s: float,
                *, pose_healthy: bool = True) -> tuple[dict, np.ndarray]:
        event, rgb = super().capture(native_qpos, time_s, pose_healthy=pose_healthy)
        event["template_used_for_control"] = False
        if not event["template_applied"]:
            return event, rgb
        correction = np.asarray(event["template_correction_xy_mm"], dtype=float)
        use_template = (event["orange_pixels"] < 40 and
                        float(np.linalg.norm(correction)) >= 3.0)
        event["template_used_for_control"] = use_template
        event["template_selection_rule"] = "orange_pixels < 40 and correction_norm_mm >= 3"
        if not use_template:
            calibrated_goal = np.asarray(event["pre_template_goal_world_xyz_m"], dtype=float)
            event["estimated_goal_world_xyz_m"] = calibrated_goal.tolist()
            if event["used_for_control"]:
                self.last_goal_world = calibrated_goal
        return event, rgb
