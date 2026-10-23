"""Arena action boundary: 3-D walking command -> Berkeley's released 22-joint ONNX policy."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_JOINTS


class BerkeleyVelocityAction(ActionTerm):
    """Freeze the published walker while a higher-level policy chooses body velocity."""

    cfg: "BerkeleyVelocityActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._joint_ids, names = self._asset.find_joints(HUMANOID_LITE_JOINTS, preserve_order=True)
        if len(names) != 22:
            raise RuntimeError(f"Expected Berkeley's 22 joints, got {names}")
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 22, device=self.device)
        self._last_walker_actions = torch.zeros(self.num_envs, 22, device=self.device)
        self._session = ort.InferenceSession(str(cfg.policy_path), providers=["CPUExecutionProvider"])
        if self._session.get_inputs()[0].shape[-1] != 75:
            raise RuntimeError("Released Berkeley walker observation contract changed")
        self._input_name = self._session.get_inputs()[0].name

    @property
    def action_dim(self):
        return 3

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = torch.clamp(
            actions,
            min=torch.tensor([-0.5, -0.3, -1.0], device=self.device),
            max=torch.tensor([0.5, 0.3, 1.0], device=self.device),
        )
        data = self._asset.data
        observation = torch.cat(
            (
                self._raw_actions,
                data.root_ang_vel_b,
                data.projected_gravity_b,
                data.joint_pos[:, self._joint_ids] - data.default_joint_pos[:, self._joint_ids],
                data.joint_vel[:, self._joint_ids],
                self._last_walker_actions,
            ),
            dim=-1,
        )
        action = self._session.run(None, {self._input_name: observation.detach().cpu().numpy().astype(np.float32)})[0]
        self._last_walker_actions[:] = torch.from_numpy(action).to(self.device)
        self._processed_actions[:] = data.default_joint_pos[:, self._joint_ids] + 0.25 * self._last_walker_actions

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        self._raw_actions[env_ids] = 0
        self._last_walker_actions[env_ids] = 0


@configclass
class BerkeleyVelocityActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = BerkeleyVelocityAction
    asset_name: str = "robot"
    policy_path: str = str(Path(__file__).with_name("policy_humanoid_dynamic.onnx"))


class BerkeleyVelocityLeftArmAction(BerkeleyVelocityAction):
    """Keep the released walker on the legs while learning five left-arm targets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 8, device=self.device)
        self._arm_scale = torch.tensor([1.2, 0.9, 0.9, 1.3, 0.8], device=self.device)
        self.arm_override_enabled = True

    @property
    def action_dim(self):
        return 8

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)
        walking = torch.clamp(
            actions[:, :3],
            min=torch.tensor([-0.5, -0.3, -1.0], device=self.device),
            max=torch.tensor([0.5, 0.3, 1.0], device=self.device),
        )
        data = self._asset.data
        observation = torch.cat((
            walking, data.root_ang_vel_b, data.projected_gravity_b,
            data.joint_pos[:, self._joint_ids] - data.default_joint_pos[:, self._joint_ids],
            data.joint_vel[:, self._joint_ids], self._last_walker_actions,
        ), dim=-1)
        walker_action = self._session.run(None, {
            self._input_name: observation.detach().cpu().numpy().astype(np.float32)
        })[0]
        self._last_walker_actions[:] = torch.from_numpy(walker_action).to(self.device)
        self._processed_actions[:] = data.default_joint_pos[:, self._joint_ids] + 0.25 * self._last_walker_actions
        if self.arm_override_enabled:
            arm_ids = self._joint_ids[:5]
            desired = data.default_joint_pos[:, arm_ids] + self._arm_scale * self._raw_actions[:, 3:]
            limits = data.soft_joint_pos_limits[:, arm_ids]
            self._processed_actions[:, :5] = torch.clamp(desired, min=limits[:, :, 0], max=limits[:, :, 1])


@configclass
class BerkeleyVelocityLeftArmActionCfg(BerkeleyVelocityActionCfg):
    class_type: type[ActionTerm] = BerkeleyVelocityLeftArmAction
