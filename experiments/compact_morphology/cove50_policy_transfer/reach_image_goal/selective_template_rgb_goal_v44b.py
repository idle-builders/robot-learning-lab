"""Opened diagnostic of fractional template correction in sparse RGB views."""

from __future__ import annotations

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.selective_template_rgb_goal_v41 import (
    SelectiveTemplateRgbGoal as V41Selector,
)


class SelectiveTemplateRgbGoal(V41Selector):
    sparse_template_blend: float = 1.0

    def capture(self, native_qpos: np.ndarray, time_s: float,
                *, pose_healthy: bool = True) -> tuple[dict, np.ndarray]:
        event, rgb = super().capture(native_qpos, time_s, pose_healthy=pose_healthy)
        blend = self.sparse_template_blend
        if not 0 <= blend <= 1:
            raise ValueError("Sparse template blend must be in [0, 1]")
        event["sparse_template_blend"] = blend
        if (not event["used_for_control"] or not event["template_used_for_control"] or
                not 24 <= event["orange_pixels"] < 64):
            return event, rgb
        calibrated = np.asarray(event["pre_template_goal_world_xyz_m"], dtype=float)
        template = np.asarray(event["estimated_goal_world_xyz_m"], dtype=float)
        selected = calibrated + blend * (template - calibrated)
        event["estimated_goal_world_xyz_m"] = selected.tolist()
        event["template_used_for_control"] = blend > 0
        event["template_blend_rule"] = "Only 24–63 observed toy pixels; blend the previously selected template correction with calibrated RGB"
        self.last_goal_world = selected
        return event, rgb
