"""Image-only corrected goal; simulator truth is absent from inference."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.rgb_calibration_model import (
    RgbCalibration,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.rgb_goal_features import (
    features,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.virtual_rgb_goal import (
    VirtualRgbGoal,
)


class CalibratedVirtualRgbGoal(VirtualRgbGoal):
    def __init__(self, *, calibration_model: Path, selected_body_sha256: str,
                 max_age_s: float, update_through_s: float | None = None) -> None:
        super().__init__(selected_body_sha256=selected_body_sha256,
                         max_age_s=max_age_s, update_through_s=update_through_s)
        self.calibration = RgbCalibration(calibration_model)
        self.calibration_model_sha256 = hashlib.sha256(calibration_model.read_bytes()).hexdigest()

    def capture(self, native_qpos: np.ndarray, time_s: float,
                *, pose_healthy: bool = True) -> tuple[dict, np.ndarray]:
        event, rgb = super().capture(native_qpos, time_s, pose_healthy=pose_healthy)
        event["coarse_goal_world_xyz_m"] = event["estimated_goal_world_xyz_m"]
        event["calibration_model_sha256"] = self.calibration_model_sha256
        event["calibration_correction_xy_mm"] = None
        if not event["accepted"]:
            return event, rgb
        coarse = np.asarray(event["estimated_goal_world_xyz_m"], dtype=np.float64)
        row = features(rgb, coarse, self.data.cam_xpos[self.camera],
                       self.data.cam_xmat[self.camera])
        correction_mm = self.calibration.correction_mm(row)
        corrected = coarse.copy()
        corrected[:2] += correction_mm / 1000.0
        event["estimated_goal_world_xyz_m"] = corrected.tolist()
        event["calibration_correction_xy_mm"] = np.round(correction_mm, 4).tolist()
        if event["used_for_control"]:
            self.last_goal_world = corrected
        return event, rgb
