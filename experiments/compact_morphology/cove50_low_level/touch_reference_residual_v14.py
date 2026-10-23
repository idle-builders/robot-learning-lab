"""Bounded contact-phase residual around the exact observed-RGB V10 reference.

The actor has no authority during the first 50 approach steps. Only five
right-arm channels gain authority smoothly over the following 15 steps, with
a hard 0.005 native-action residual bound. This is a changed staged-learning
candidate, not a claim that contact has been learned.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from touch_reference_residual_v10 import residual_touch_env_cfg as _v10_cfg
from touch_reference_residual_v9 import TouchReferenceResidual, TouchReferenceResidualCfg


@dataclass(kw_only=True)
class ContactPhaseResidualCfg(TouchReferenceResidualCfg):
    residual_gain: float = 0.005
    warmup_steps: int = 50
    ramp_steps: int = 15

    def build(self, env):
        return ContactPhaseResidual(self, env)


class ContactPhaseResidual(TouchReferenceResidual):
    cfg: ContactPhaseResidualCfg

    def __init__(self, cfg: ContactPhaseResidualCfg, env):
        if not (0 < cfg.residual_gain <= 0.005 and cfg.warmup_steps >= 0
                and cfg.ramp_steps >= 1):
            raise ValueError("V14 stage authority bounds differ")
        super().__init__(cfg, env)
        self.last_policy_gain = torch.zeros(self.num_envs, device=self.device)
        self.last_policy_residual = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor) -> None:
        stage = ((self._env.episode_length_buf.float() + 1 - self.cfg.warmup_steps)
                 / self.cfg.ramp_steps).clamp(0.0, 1.0)
        self.last_policy_gain[:] = self.cfg.residual_gain * stage
        bounded = torch.zeros_like(actions)
        bounded[:, self._arm_indices] = actions[:, self._arm_indices].clamp(-1.0, 1.0)
        self.last_policy_residual[:] = bounded * stage[:, None]
        super().process_actions(self.last_policy_residual)


def residual_touch_env_cfg(*, cube_xy=None, cube_xy_range=None):
    cfg = _v10_cfg(cube_xy=cube_xy, cube_xy_range=cube_xy_range,
                  residual_gain=0.005)
    original = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = ContactPhaseResidualCfg(
        entity_name=original.entity_name, actuator_names=original.actuator_names,
        scale=original.scale, offset=original.offset, clip=original.clip,
        preserve_order=original.preserve_order,
        use_default_offset=original.use_default_offset,
    )
    return cfg
