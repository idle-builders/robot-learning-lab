"""Opened candidate: expand visible-template correction to the 40-63 px band.

The V32 rule left a 50-pixel entry with a 4.24 mm image-derived correction
on calibrated RGB. This candidate uses only observed pixels and template
disagreement; native retention still determines whether it can advance.
"""

from __future__ import annotations

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.template_virtual_rgb_goal_v30 import (
    TemplateMatchedVirtualRgbGoal,
)


class SelectiveTemplateRgbGoal(TemplateMatchedVirtualRgbGoal):
    """Keep calibrated RGB outside the bounded sparse-image band."""

    def capture(self, native_qpos: np.ndarray, time_s: float,
                *, pose_healthy: bool = True) -> tuple[dict, np.ndarray]:
        event, rgb = super().capture(native_qpos, time_s, pose_healthy=pose_healthy)
        event["template_used_for_control"] = False
        if not event["template_applied"]:
            return event, rgb
        correction = np.asarray(event["template_correction_xy_mm"], dtype=float)
        use_template = (event["orange_pixels"] < 64 and
                        float(np.linalg.norm(correction)) >= 3.0)
        event["template_used_for_control"] = use_template
        event["template_selection_rule"] = "orange_pixels < 64 and correction_norm_mm >= 3"
        if not use_template:
            calibrated_goal = np.asarray(event["pre_template_goal_world_xyz_m"], dtype=float)
            event["estimated_goal_world_xyz_m"] = calibrated_goal.tolist()
            if event["used_for_control"]:
                self.last_goal_world = calibrated_goal
        return event, rgb
