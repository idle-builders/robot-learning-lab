"""Joint-action frame for a screened Cove single-foot pose."""

from __future__ import annotations

import re


def configure_residual_actions(cfg, preload_joints: dict[str, float],
                               *, leg_scale: float = 0.2) -> None:
    """Make a zero action command the measured gravity-preloaded target."""
    action = cfg.actions["joint_pos"]
    action.use_default_offset = False
    action.offset = {f"^{re.escape(name)}$": float(value)
                     for name, value in preload_joints.items()}
    action.scale = {
        **{f"^{side}_{joint}$": leg_scale
           for side in ("left", "right")
           for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")},
        **{f"^{name}$": 0.0 for name in preload_joints
           if not name.startswith(("left_hip_", "left_knee", "left_ankle_",
                                   "right_hip_", "right_knee", "right_ankle_"))},
    }
