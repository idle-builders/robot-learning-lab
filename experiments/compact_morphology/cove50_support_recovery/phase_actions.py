"""A scheduled stand → one-foot → stand target for residual PPO training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointPositionAction


def phase_blend(env, lift_s: float, hold_s: float, lower_s: float,
                stage: str = "all"):
    """Smooth stand → target → stand blend shared by actions and rewards."""
    t = env.episode_length_buf.to(torch.float32) * env.step_dt
    if stage == "all":
        up = torch.clamp(t / lift_s, 0.0, 1.0)
        down = torch.clamp((lift_s + hold_s + lower_s - t) / lower_s, 0.0, 1.0)
    elif stage == "stance":
        up = torch.clamp(t / (lift_s / 2.0), 0.0, 1.0)
        down = torch.clamp((lift_s + hold_s + lower_s - t) /
                           (lower_s / 2.0), 0.0, 1.0)
    elif stage == "swing":
        up = torch.clamp((t - lift_s / 2.0) / (lift_s / 2.0), 0.0, 1.0)
        down = torch.clamp((lift_s + hold_s + lower_s / 2.0 - t) /
                           (lower_s / 2.0), 0.0, 1.0)
    else:
        raise ValueError(f"Unknown transition stage: {stage}")
    phase = torch.minimum(up, down)
    return phase * phase * (3.0 - 2.0 * phase)


@dataclass(kw_only=True)
class PhasePositionActionCfg(JointPositionActionCfg):
    target_joints: tuple[float, ...]
    lift_s: float = 2.0
    hold_s: float = 3.0
    lower_s: float = 2.0
    freeze_swing_side: str | None = None
    arm_clearance_bump_rad: float = 0.0
    staged: bool = False
    stance_side: str = "right"

    def build(self, env):
        return PhasePositionAction(self, env)


class PhasePositionAction(JointPositionAction):
    def __init__(self, cfg: PhasePositionActionCfg, env):
        super().__init__(cfg, env)
        if len(cfg.target_joints) != self.action_dim:
            raise ValueError("Phase target must match all selected actuator joints")
        self._home = self._entity.data.default_joint_pos[:, self._target_ids].clone()
        self._target = torch.tensor(cfg.target_joints, device=self.device,
                                    dtype=self._home.dtype)[None, :]
        self._frozen_indices = [
            i for i, name in enumerate(self._target_names)
            if cfg.freeze_swing_side and name.startswith(f"{cfg.freeze_swing_side}_")
            and any(name.endswith(axis) for axis in
                    ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch"))
        ]
        if cfg.stance_side not in ("left", "right"):
            raise ValueError(f"Unsupported stance side: {cfg.stance_side}")
        self._swing_side = "left" if cfg.stance_side == "right" else "right"
        self._arm_clearance_index = self._target_names.index(
            f"{self._swing_side}_shoulder_roll")
        self._swing_indices = [i for i, name in enumerate(self._target_names)
                               if name.startswith(f"{self._swing_side}_") and any(
                                   name.endswith(axis) for axis in
                                   ("hip_yaw", "hip_roll", "hip_pitch", "knee",
                                    "ankle_pitch"))]

    def process_actions(self, actions: torch.Tensor):
        phase = phase_blend(self._env, self.cfg.lift_s,
                            self.cfg.hold_s, self.cfg.lower_s,
                            "stance" if self.cfg.staged else "all")
        joint_phase = phase[:, None].expand(-1, self.action_dim).clone()
        if self.cfg.staged:
            joint_phase[:, self._swing_indices] = phase_blend(
                self._env, self.cfg.lift_s, self.cfg.hold_s,
                self.cfg.lower_s, "swing")[:, None]
        self._offset = self._home + joint_phase * (self._target - self._home)
        if self.cfg.arm_clearance_bump_rad:
            self._offset[:, self._arm_clearance_index] += (
                self.cfg.arm_clearance_bump_rad * torch.sin(torch.pi * phase))
        if self._frozen_indices:
            actions = actions.clone()
            actions[:, self._frozen_indices] = 0.0
        super().process_actions(actions)
