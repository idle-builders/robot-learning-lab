"""Actor-visible RGB target relative to two kinematically projected pads.

The pad projections use measured robot joints and a nominal root transform.
Camera and pads share the root, so their relative projection is invariant to
the unknown global robot pose. Cube pose and contact never enter this feature.
"""

from __future__ import annotations

import mujoco
import numpy as np

from probe_free_object_relative_rgb_v5 import (CAMERA, FIXED_PAD, IMAGE_HEIGHT,
                                               IMAGE_WIDTH, MOVING_PAD, projection)


class PadRelativeRgb:
    def __init__(self, env):
        self.model = env.sim.mj_model
        self.data = mujoco.MjData(self.model)
        self.moving = self.model.geom(MOVING_PAD).id
        self.fixed = self.model.geom(FIXED_PAD).id
        robot = env.scene["robot"]
        self.default_joint_pos = robot.data.default_joint_pos[0].detach().cpu().numpy().copy()
        self.joint_addresses = np.asarray(
            [self.model.joint(f"robot/{name}").qposadr[0] for name in robot.joint_names])
        assert len(self.default_joint_pos) == len(self.joint_addresses) == 21
        assert self.model.camera(CAMERA).id >= 0

    def features(self, env, actor_obs) -> np.ndarray:
        rgb = actor_obs[-6:].detach().cpu().numpy()
        if rgb[3] != 1 or rgb[5] != 0:
            return np.zeros(4, dtype=np.float32)
        self.data.qpos[:] = self.model.qpos0
        # The native actor's first 21 values are noisy joint positions relative
        # to saved defaults. Reconstruct only those measured joints. Simulator
        # root, cube, exact noiseless joints and contact do not enter.
        self.data.qpos[self.joint_addresses] = (
            actor_obs[:21].detach().cpu().numpy() + self.default_joint_pos)
        mujoco.mj_forward(self.model, self.data)
        cube_uv = rgb[:2] * [IMAGE_WIDTH - 1, IMAGE_HEIGHT - 1]
        moving_uv = projection(self.model, self.data, self.data.geom_xpos[self.moving])
        fixed_uv = projection(self.model, self.data, self.data.geom_xpos[self.fixed])
        relative = np.r_[cube_uv - moving_uv, cube_uv - fixed_uv]
        assert np.isfinite(relative).all()
        return (relative / [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]).astype(np.float32)

    def augmented(self, env, actor_obs) -> np.ndarray:
        return np.r_[actor_obs.detach().cpu().numpy(), self.features(env, actor_obs)].astype(np.float32)
