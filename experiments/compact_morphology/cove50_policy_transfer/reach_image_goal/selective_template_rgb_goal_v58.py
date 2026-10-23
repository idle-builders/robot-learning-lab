"""Candidate small observed correction in high-error negative-camera sparse views."""

from __future__ import annotations

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.selective_template_rgb_goal_v47 import (
    SelectiveTemplateRgbGoal as V47Selector,
)


MIN_NEGATIVE_SPARSE_CORRECTION_MM = 9.0
NEGATIVE_SPARSE_FRACTION = .15


class SelectiveTemplateRgbGoal(V47Selector):
    """Keep V47 except for a bounded observed correction in sparse negative views."""

    def capture(self, native_qpos: np.ndarray, time_s: float,
                *, pose_healthy: bool = True) -> tuple[dict, np.ndarray]:
        event, rgb = super().capture(native_qpos, time_s, pose_healthy=pose_healthy)
        correction = np.asarray(event.get("template_correction_xy_mm") or (0.0, 0.0),
                                dtype=float)
        norm_mm = float(np.linalg.norm(correction))
        apply = bool(event.get("negative_camera_sparse_abstention") and
                     norm_mm >= MIN_NEGATIVE_SPARSE_CORRECTION_MM)
        event["negative_camera_sparse_partial_correction"] = apply
        event["negative_camera_sparse_correction_norm_mm"] = round(norm_mm, 4)
        event["negative_camera_sparse_partial_rule"] = (
            "Observed negative-camera 24-63-pixel sparse region and rendered-template "
            "correction >= 9 mm: use 15% of that correction; otherwise preserve V47")
        if apply:
            calibrated = np.asarray(event["pre_template_goal_world_xyz_m"], dtype=float)
            selected = calibrated.copy()
            selected[:2] += NEGATIVE_SPARSE_FRACTION * correction / 1000
            event["estimated_goal_world_xyz_m"] = selected.tolist()
            event["template_used_for_control"] = True
            self.last_goal_world = selected
        return event, rgb
