"""Prospective Stage-A reference action at the native 21-joint interface.

This is an action adapter, not a trained policy. A zero residual must reproduce
the authored load/return target from the same skill clock and latched side.
"""

from dataclasses import dataclass

import numpy as np
import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values


def _smooth_phase(value: float) -> float:
    # Match the saved native teacher's Python/NumPy scalar arithmetic before
    # converting to the float32 target that MJLab receives.
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def reference_tables(step_dt: float, steps: int, shift_roll_rad: float,
                     device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute exact source-teacher target values on the fixed 50 Hz clock."""
    hip, shoulder = [], []
    for step in range(steps):
        t = step * step_dt
        if t < 2.0:
            shift = _smooth_phase(t - 1.0)
        elif t < 5.0:
            shift = 1.0
        else:
            shift = 1.0 - _smooth_phase((t - 5.0) / 2.0)
        hip.append(shift_roll_rad * shift)
        shoulder.append(0.20 * shift)
    return (torch.tensor(hip, dtype=torch.float32, device=device),
            torch.tensor(shoulder, dtype=torch.float32, device=device))


def load_transfer_reference(
    step_index: torch.Tensor,
    latched_side: torch.Tensor,
    target_names: list[str],
    hip_table: torch.Tensor,
    shoulder_table: torch.Tensor,
) -> torch.Tensor:
    """Gather exact V5/V5b zero-based targets for parallel environments."""
    hip = hip_table[step_index] * latched_side
    shoulder = shoulder_table[step_index] * latched_side
    target = torch.zeros((step_index.shape[0], len(target_names)),
                         device=step_index.device, dtype=torch.float32)
    for side in ("left", "right"):
        target[:, target_names.index(f"{side}_hip_roll")] = hip
    target[:, target_names.index("left_shoulder_roll")] = torch.where(
        latched_side > 0, shoulder, 0.0
    )
    target[:, target_names.index("right_shoulder_roll")] = torch.where(
        latched_side < 0, shoulder, 0.0
    )
    return target


@dataclass(kw_only=True)
class SupportReferenceActionCfg(JointPositionActionCfg):
    use_default_offset: bool = False
    shift_roll_rad: float = 0.35
    reference_native_scale: float | dict[str, float] = 1.0
    reference_native_offset: float | dict[str, float] = 0.0
    reference_use_default_offset: bool = True
    reference_table_steps: int = 512
    # None preserves the V10 21D parity adapter. A tuple gives MJLab's action
    # manager a real smaller policy slice, in exactly this named joint order.
    residual_joint_names: tuple[str, ...] | None = None
    raw_residual_clip: float = 1.0

    def build(self, env):
        return SupportReferenceAction(self, env)


class SupportReferenceAction(JointPositionAction):
    """Apply a native position residual around the latched-side reference.

    ``clip`` is the original absolute joint-target limit. It is applied after
    adding the reference, not to the residual in ``process_actions``.
    """

    def __init__(self, cfg: SupportReferenceActionCfg, env):
        super().__init__(cfg, env)
        target_shape = (self.num_envs, len(self._target_names))
        if cfg.residual_joint_names is None:
            self.residual_joint_names = tuple(self._target_names)
        else:
            self.residual_joint_names = tuple(cfg.residual_joint_names)
        if (not self.residual_joint_names or
                len(set(self.residual_joint_names)) != len(self.residual_joint_names) or
                any(name not in self._target_names for name in self.residual_joint_names)):
            raise ValueError("Residual joints must be distinct native target names")
        if not 0.0 < cfg.raw_residual_clip <= 1.0:
            raise ValueError("Raw residual clip must lie in (0, 1]")
        self.residual_target_ids = torch.tensor(
            [self._target_names.index(name) for name in self.residual_joint_names],
            device=self.device, dtype=torch.long,
        )
        # BaseAction initially allocates 21D buffers. Resize the policy-facing
        # buffers only; native scale, offset and clip remain 21D below.
        self._action_dim = len(self.residual_joint_names)
        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self.last_bounded_action = torch.zeros_like(self._raw_actions)

        def resolved(value: float | dict[str, float], default: float) -> torch.Tensor:
            array = torch.full(target_shape, default, device=self.device)
            if isinstance(value, dict):
                indices, _, values = resolve_matching_names_values(value, self._target_names)
                array[:, indices] = torch.tensor(values, device=self.device)
            else:
                array.fill_(float(value))
            return array

        self.reference_native_scale = resolved(cfg.reference_native_scale, 1.0)
        self.reference_native_offset = (
            self._entity.data.default_joint_pos[:, self._target_ids].clone()
            if cfg.reference_use_default_offset else
            resolved(cfg.reference_native_offset, 0.0)
        )
        if abs(env.step_dt - 0.02) > 1e-9:
            raise ValueError("V10 source reference requires a 0.02 s control step")
        if env.cfg.episode_length_s > cfg.reference_table_steps * env.step_dt:
            raise ValueError("Reference table is shorter than the configured episode")
        self.reference_hip_table, self.reference_shoulder_table = reference_tables(
            env.step_dt, cfg.reference_table_steps, cfg.shift_roll_rad, self.device
        )

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self.last_bounded_action = self._raw_actions.clamp(
            -self.cfg.raw_residual_clip, self.cfg.raw_residual_clip
        )
        self._processed_actions = self.last_bounded_action * self._scale + self._offset

    def reference_target(self) -> torch.Tensor:
        """Decode the source teacher's 21 native actions at this control index."""
        command = self._env.command_manager.get_term("twist")
        step_index = self._env.episode_length_buf
        reference = load_transfer_reference(
            step_index, command.stance_side, self._target_names,
            self.reference_hip_table, self.reference_shoulder_table,
        )
        active = self.reference_native_scale.abs() > 1e-8
        safe_scale = torch.where(active, self.reference_native_scale, 1.0)
        native = torch.where(
            active, (reference - self.reference_native_offset) / safe_scale, 0.0
        )
        self.last_reference_native_action = native
        return native * self.reference_native_scale + self.reference_native_offset

    def apply_actions(self):
        # Reproduce the source teacher's target -> native action -> applied
        # target conversion, including its float32 rounding and zero-scale
        # channels, before adding a learned correction.
        target = self.reference_target()
        # The named policy slice is scattered into the full native target.
        target[:, self.residual_target_ids] += self._processed_actions
        self.last_unclipped_target = target.clone()
        if self.cfg.clip is not None:
            target = torch.clamp(target, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        self.last_absolute_target = target
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(target - encoder_bias, joint_ids=self._target_ids)
