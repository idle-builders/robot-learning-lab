"""Opened candidate: use image-derived correction in the negative-X camera corner.

V38's 10508 entry has 124 orange pixels and a 2.24 mm image-derived
correction that lowers scored goal error from 2.56 to 0.35 mm. This rule
also retains the V35 sparse-image branch. Native tests decide contact.
"""

from __future__ import annotations

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.template_virtual_rgb_goal_v30 import (
    TemplateMatchedVirtualRgbGoal,
)


class SelectiveTemplateRgbGoal(TemplateMatchedVirtualRgbGoal):
    """Select templates from bounded observed image and camera features."""

    def capture(self, native_qpos: np.ndarray, time_s: float,
                *, pose_healthy: bool = True) -> tuple[dict, np.ndarray]:
        event, rgb = super().capture(native_qpos, time_s, pose_healthy=pose_healthy)
        event["template_used_for_control"] = False
        if not event["template_applied"]:
            return event, rgb
        correction = np.asarray(event["template_correction_xy_mm"], dtype=float)
        correction_norm = float(np.linalg.norm(correction))
        camera_x, camera_y = event["camera_world_xyz_m"][:2]
        sparse = event["orange_pixels"] < 64 and correction_norm >= 3.0
        corner = (0.03 <= camera_x < 0.08 and camera_y < -0.08 and
                  correction_norm >= 2.0 and event["template_pixels_disagree"] <= 10)
        use_template = sparse or corner
        event["template_used_for_control"] = use_template
        event["template_selection_rule"] = (
            "(orange_pixels < 64 and correction_norm_mm >= 3) or "
            "(0.03 <= camera_x_m < 0.08 and camera_y_m < -0.08 and "
            "correction_norm_mm >= 2 and template_pixels_disagree <= 10)")
        if not use_template:
            calibrated_goal = np.asarray(event["pre_template_goal_world_xyz_m"], dtype=float)
            event["estimated_goal_world_xyz_m"] = calibrated_goal.tolist()
            if event["used_for_control"]:
                self.last_goal_world = calibrated_goal
        return event, rgb
