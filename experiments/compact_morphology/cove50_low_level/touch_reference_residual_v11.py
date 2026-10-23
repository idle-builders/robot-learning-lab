"""Arm-only residual whose initial authority follows the RGB teacher ramp.

The V10 actor perturbed all 21 joints at full gain on the first step. This
variant keeps the same 79-value actor observation and saved teacher, but gives
the learner authority only over the five touching-arm channels. It is a new
experimental action interface, not a reinterpretation of V10's score.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from touch_reference_residual_v9 import TouchReferenceResidual, TouchReferenceResidualCfg
from touch_reference_residual_v10 import residual_touch_env_cfg as v10_env_cfg


@dataclass(kw_only=True)
class ArmRampResidualCfg(TouchReferenceResidualCfg):
    residual_gain: float = .04

    def build(self, env) -> ArmRampResidual:
        return ArmRampResidual(self, env)


class ArmRampResidual(TouchReferenceResidual):
    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != self.action_dim:
            raise ValueError("V11 expects all 21 source action channels")
        bounded = actions.clamp(-1, 1)
        correction = torch.zeros_like(bounded)
        ramp = ((self._env.episode_length_buf.float() + 1) / 75).clamp(max=.975)
        correction[:, self._arm_indices] = bounded[:, self._arm_indices] * ramp[:, None]
        super().process_actions(correction)


def residual_touch_env_cfg(*, cube_xy=None, cube_xy_range=None):
    cfg = v10_env_cfg(cube_xy=cube_xy, cube_xy_range=cube_xy_range,
                      residual_gain=.04)
    old = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = ArmRampResidualCfg(
        entity_name=old.entity_name, actuator_names=old.actuator_names,
        scale=old.scale, offset=old.offset, clip=old.clip,
        preserve_order=old.preserve_order,
        use_default_offset=old.use_default_offset,
        residual_gain=.04,
    )
    return cfg
