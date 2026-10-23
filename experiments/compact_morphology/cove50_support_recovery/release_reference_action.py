"""Side-mapped seven-channel release residual over the two-foot load reference.

The final three policy channels target the swing leg selected by the latched
stance command. The other leg's pitch/knee/ankle target remains at reference.
"""

from dataclasses import dataclass

import torch

from support_reference_action import SupportReferenceAction, SupportReferenceActionCfg


RELEASE_RESIDUAL_JOINTS = (
    "left_hip_roll", "right_hip_roll",
    "left_shoulder_roll", "right_shoulder_roll",
    "left_hip_pitch", "left_knee", "left_ankle_pitch",
)
SWING_JOINT_SUFFIXES = ("hip_pitch", "knee", "ankle_pitch")


@dataclass(kw_only=True)
class ReleaseReferenceActionCfg(SupportReferenceActionCfg):
    residual_joint_names: tuple[str, ...] = RELEASE_RESIDUAL_JOINTS
    release_scales_rad: tuple[float, ...] = (0.05, 0.05, 0.05, 0.05,
                                            0.70, 1.20, 0.33)

    def build(self, env):
        return ReleaseReferenceAction(self, env)


class ReleaseReferenceAction(SupportReferenceAction):
    """Scatter three swing commands by side without exposing side-specific heads."""

    def __init__(self, cfg: ReleaseReferenceActionCfg, env):
        super().__init__(cfg, env)
        if tuple(self.residual_joint_names) != RELEASE_RESIDUAL_JOINTS:
            raise ValueError("Release policy needs the seven named channel contract")
        if len(cfg.release_scales_rad) != 7 or any(x <= 0 for x in cfg.release_scales_rad):
            raise ValueError("Release policy needs seven positive target scales")
        self.release_scales = torch.tensor(cfg.release_scales_rad, device=self.device)
        self.fixed_target_ids = self.residual_target_ids[:4]
        self.swing_target_ids = {
            side: torch.tensor(
                [self._target_names.index(f"{side}_{suffix}")
                 for suffix in SWING_JOINT_SUFFIXES],
                dtype=torch.long, device=self.device,
            )
            for side in ("left", "right")
        }

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self.last_bounded_action = self._raw_actions.clamp(
            -self.cfg.raw_residual_clip, self.cfg.raw_residual_clip
        )
        self._processed_actions = self.last_bounded_action * self.release_scales

    def apply_actions(self):
        target = self.reference_target()
        t = self._env.episode_length_buf * self._env.step_dt
        release_window = ((t >= 2.0) & (t < 5.0)).float()[:, None]
        correction = self._processed_actions * release_window
        self.last_applied_residual = correction
        target[:, self.fixed_target_ids] += correction[:, :4]
        right_stance = self._env.command_manager.get_term("twist").stance_side > 0
        target[:, self.swing_target_ids["left"]] += torch.where(
            right_stance[:, None], correction[:, 4:], 0.0
        )
        target[:, self.swing_target_ids["right"]] += torch.where(
            right_stance[:, None], 0.0, correction[:, 4:]
        )
        self.last_unclipped_target = target.clone()
        if self.cfg.clip is not None:
            target = torch.clamp(target, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        self.last_absolute_target = target
        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(target - encoder_bias, joint_ids=self._target_ids)
