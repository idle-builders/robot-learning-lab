"""Candidate sparse-template abstention from observed camera view and correction."""

from __future__ import annotations

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.selective_template_rgb_goal_v45 import (
    SelectiveTemplateRgbGoal as V45Selector,
)


POSITIVE_CAMERA_Y_MAX_M = -0.08
LARGE_SPARSE_CORRECTION_MM = 6.0


class SelectiveTemplateRgbGoal(V45Selector):
    """Keep calibrated RGB for a large sparse correction from the side view."""

    def capture(self, native_qpos: np.ndarray, time_s: float,
                *, pose_healthy: bool = True) -> tuple[dict, np.ndarray]:
        event, rgb = super().capture(native_qpos, time_s, pose_healthy=pose_healthy)
        correction = event.get("template_correction_xy_mm") or (0.0, 0.0)
        magnitude = float(np.linalg.norm(correction))
        camera_x, camera_y = event["camera_world_xyz_m"][:2]
        abstain = bool(event["used_for_control"] and event["template_used_for_control"] and
                       24 <= event["orange_pixels"] < 64 and camera_x >= 0 and
                       camera_y < POSITIVE_CAMERA_Y_MAX_M and
                       magnitude >= LARGE_SPARSE_CORRECTION_MM)
        event["positive_camera_large_sparse_abstention"] = abstain
        event["positive_camera_sparse_correction_norm_mm"] = round(magnitude, 4)
        event["positive_camera_large_sparse_rule"] = (
            "Observed 24-63 toy pixels, capture camera world X >= 0 m and Y < -0.08 m, "
            "rendered-template correction >= 6 mm; retain calibrated RGB")
        if abstain:
            calibrated = np.asarray(event["pre_template_goal_world_xyz_m"], dtype=float)
            event["estimated_goal_world_xyz_m"] = calibrated.tolist()
            event["template_used_for_control"] = False
            self.last_goal_world = calibrated
        return event, rgb
