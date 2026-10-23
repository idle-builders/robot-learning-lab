"""Abstain from image-template control when the toy mask is too small.

The opened 10546 view has 20 orange pixels: its template reduced scored
position error but changed a 1.96 s calibrated-RGB contact pass into a miss.
The earlier varied-view evidence includes useful template corrections with
25 pixels, so 24 is a conservative visible-area floor for control. The
rendered estimate is still measured and recorded when this rule abstains.
"""

from __future__ import annotations

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.template_virtual_rgb_goal_v30 import (
    TemplateMatchedVirtualRgbGoal,
)


MIN_CONTROL_ORANGE_PIXELS = 24


class SelectiveTemplateRgbGoal(TemplateMatchedVirtualRgbGoal):
    """Choose a template goal from visible pixels and camera pose only."""

    def capture(self, native_qpos: np.ndarray, time_s: float,
                *, pose_healthy: bool = True) -> tuple[dict, np.ndarray]:
        event, rgb = super().capture(native_qpos, time_s, pose_healthy=pose_healthy)
        event["template_used_for_control"] = False
        event["template_abstained_low_visibility"] = False
        if not event["template_applied"]:
            return event, rgb
        correction = np.asarray(event["template_correction_xy_mm"], dtype=float)
        correction_norm = float(np.linalg.norm(correction))
        camera_x, camera_y = event["camera_world_xyz_m"][:2]
        sparse = event["orange_pixels"] < 64 and correction_norm >= 3.0
        corner = (0.03 <= camera_x < 0.08 and camera_y < -0.08 and
                  correction_norm >= 2.0 and event["template_pixels_disagree"] <= 10)
        use_template = (event["orange_pixels"] >= MIN_CONTROL_ORANGE_PIXELS and
                        (sparse or corner))
        event["template_used_for_control"] = use_template
        event["template_abstained_low_visibility"] = (
            event["orange_pixels"] < MIN_CONTROL_ORANGE_PIXELS and (sparse or corner))
        event["template_selection_rule"] = (
            "orange_pixels >= 24 and ((orange_pixels < 64 and "
            "correction_norm_mm >= 3) or (0.03 <= camera_x_m < 0.08 and "
            "camera_y_m < -0.08 and correction_norm_mm >= 2 and "
            "template_pixels_disagree <= 10))")
        if not use_template:
            calibrated_goal = np.asarray(event["pre_template_goal_world_xyz_m"], dtype=float)
            event["estimated_goal_world_xyz_m"] = calibrated_goal.tolist()
            if event["used_for_control"]:
                self.last_goal_world = calibrated_goal
        return event, rgb
