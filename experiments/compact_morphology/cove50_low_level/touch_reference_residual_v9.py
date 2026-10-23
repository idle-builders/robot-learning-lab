"""Observed-RGB teacher reference plus bounded learned action residual.

The policy action is a correction around a deployable image-guided teacher.
Object pose and contact truth do not enter the reference or actor input.
This is a separate V9 experiment, not a continuation of V1 scratch PPO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationTermCfg

from ape_walk.physics.frame50 import require_direct_frame
from collect_free_object_touch import BODY, adapted_target
from free_object_touch_env import touch_env_cfg
from probe_observed_free_object_touch import entry_rgb_calibration


def teacher_reference_state(env) -> torch.Tensor:
    """Actor-visible next reference and prior RGB coordinates for cue timing."""
    return env.action_manager.get_term("joint_pos").next_reference_state()


@dataclass(kw_only=True)
class TouchReferenceResidualCfg(JointPositionActionCfg):
    residual_gain: float = .08

    def build(self, env) -> TouchReferenceResidual:
        return TouchReferenceResidual(self, env)


class TouchReferenceResidual(JointPositionAction):
    cfg: TouchReferenceResidualCfg

    def __init__(self, cfg: TouchReferenceResidualCfg, env):
        super().__init__(cfg, env)
        self._model = require_direct_frame(BODY)
        self._coefficients, self.calibration = entry_rgb_calibration()
        self._teacher_target = self._offset.clone()
        self._last_rgb = torch.zeros((self.num_envs, 6), device=self.device)
        self._held_alpha = torch.full((self.num_envs,), float("nan"), device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.teacher_raw = torch.zeros_like(self._raw_actions)
        self._arm_indices = [self._target_names.index(name) for name in (
            "right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch",
            "right_elbow_pitch", "right_gripper")]
        self.rgb_entry_count = 0
        self.rgb_cue_count = 0

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._initialized[env_ids] = False
        self._held_alpha[env_ids] = float("nan")
        self._teacher_target[env_ids] = self._offset[env_ids]
        self._last_rgb[env_ids] = 0
        self.teacher_raw[env_ids] = 0

    def next_reference_state(self) -> torch.Tensor:
        ramp = ((self._env.episode_length_buf.float() + 1) / 75).clamp(max=.975)
        held = ~torch.isnan(self._held_alpha)
        alpha = torch.where(held, self._held_alpha, ramp)
        difference = self._teacher_target - self._offset
        raw = torch.where(self._scale.abs() > 1e-8,
                          difference / self._scale.clamp_min(1e-8),
                          torch.zeros_like(difference))
        next_raw = (alpha[:, None] * raw).clamp(-1, 1)
        next_raw[~self._initialized] = 0
        return torch.cat((self._initialized.float()[:, None], held.float()[:, None],
                          alpha[:, None], next_raw[:, self._arm_indices],
                          self._last_rgb[:, :2]), dim=1)

    def _initialize_from_rgb(self, ids: torch.Tensor, rgb: torch.Tensor) -> None:
        pixels = rgb[ids].detach().cpu().numpy()
        if not np.all((pixels[:, 3] == 1) & (pixels[:, 5] == 0)):
            raise RuntimeError("V9 teacher reset has missing/stale RGB target")
        estimates = np.column_stack((np.ones(len(pixels)), pixels[:, :3])) @ self._coefficients
        for env_id, xy in zip(ids.detach().cpu().tolist(), estimates):
            target = adapted_target(self._model, (float(xy[0]), float(xy[1])))
            self._teacher_target[env_id] = self._offset[env_id]
            for name, value in target.items():
                self._teacher_target[env_id, self._target_names.index(name)] = value
        self._last_rgb[ids] = rgb[ids]
        self._initialized[ids] = True
        self.rgb_entry_count += len(ids)

    def process_actions(self, actions: torch.Tensor) -> None:
        rgb = self._env.obs_buf["actor"][:, -6:]
        new = (~self._initialized).nonzero(as_tuple=False).squeeze(-1)
        if len(new):
            self._initialize_from_rgb(new, rgb)
        valid = (rgb[:, 3] == 1) & (rgb[:, 5] == 0)
        delta = rgb[:, :2] - self._last_rgb[:, :2]
        cue = (self._initialized & torch.isnan(self._held_alpha) &
               (self._env.episode_length_buf > 0) & valid &
               (delta[:, 0] <= -.0015) & (delta[:, 1] <= -.0015))
        previous_alpha = (self._env.episode_length_buf.float() / 75).clamp(max=.975)
        self._held_alpha[cue] = previous_alpha[cue]
        self.rgb_cue_count += int(cue.sum().item())
        self._last_rgb[:] = rgb
        ramp = ((self._env.episode_length_buf.float() + 1) / 75).clamp(max=.975)
        alpha = torch.where(torch.isnan(self._held_alpha), ramp, self._held_alpha)
        difference = self._teacher_target - self._offset
        raw = torch.where(self._scale.abs() > 1e-8,
                          difference / self._scale.clamp_min(1e-8),
                          torch.zeros_like(difference))
        self.teacher_raw[:] = (alpha[:, None] * raw).clamp(-1, 1)
        combined = (self.teacher_raw + self.cfg.residual_gain * actions).clamp(-1, 1)
        super().process_actions(combined)


def residual_touch_env_cfg(*, cube_xy=None, cube_xy_range=None,
                           residual_gain: float = .08):
    cfg = touch_env_cfg(BODY, entry_mode="stand-tabletop", stand_cube_xy=cube_xy,
                        stand_cube_xy_range=cube_xy_range)
    original = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = TouchReferenceResidualCfg(
        entity_name=original.entity_name, actuator_names=original.actuator_names,
        scale=original.scale, offset=original.offset, clip=original.clip,
        preserve_order=original.preserve_order,
        use_default_offset=original.use_default_offset,
        residual_gain=residual_gain,
    )
    rgb = cfg.observations["actor"].terms.pop("rgb_target")
    cfg.observations["actor"].terms["teacher_reference"] = ObservationTermCfg(func=teacher_reference_state)
    cfg.observations["actor"].terms["rgb_target"] = rgb
    return cfg
