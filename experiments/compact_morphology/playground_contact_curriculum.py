"""Floor-contact PPO with physics-checked contact resets and standing resets."""

import hashlib
import json
from pathlib import Path

from absl import app
import jax
import jax.numpy as jp
from mujoco import mjx

from mujoco_playground import locomotion, registry

from playground_reach import REACH_MODEL
from playground_reach_stable import CompactStableFloorReach, default_config
from learning import train_jax_ppo


BANK = Path(__file__).with_name("contact_reset_bank.json")


class CompactContactCurriculum(CompactStableFloorReach):
    def __init__(self, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)
        bank = json.loads(BANK.read_text())
        if bank["model_sha256"] != hashlib.sha256(REACH_MODEL.read_bytes()).hexdigest():
            raise ValueError("Contact reset bank does not match the reach model")
        self._contact_qpos = jp.asarray([pose["qpos"] for pose in bank["poses"]])
        self._contact_ctrl = jp.asarray([pose["ctrl"] for pose in bank["poses"]])
        self._contact_targets = jp.asarray([pose["target_xyz_m"] for pose in bank["poses"]])

    def reset(self, rng):
        rng, choose_rng, pose_rng, jitter_rng, noise_rng = jax.random.split(rng, 5)
        state = super().reset(rng)
        use_contact_pose = jax.random.bernoulli(choose_rng, p=0.65)
        index = jax.random.randint(pose_rng, (), 0, len(self._contact_targets))
        jitter = jax.random.uniform(jitter_rng, (2,), minval=-0.003, maxval=0.003)
        contact_target = self._contact_targets[index] + jp.array([jitter[0], jitter[1], 0.0])
        info = dict(state.info)
        info["command"] = jp.where(use_contact_pose, contact_target, info["command"])
        data = state.data.replace(
            qpos=jp.where(use_contact_pose, self._contact_qpos[index], state.data.qpos),
            ctrl=jp.where(use_contact_pose, self._contact_ctrl[index], state.data.ctrl),
            mocap_pos=state.data.mocap_pos.at[0].set(info["command"]),
        )
        data = mjx.forward(self.mjx_model, data)
        obs = self._get_obs(data, info, jp.zeros_like(state.obs), noise_rng)
        return state.replace(data=data, obs=obs, info=info)


locomotion.register_environment("CompactContactCurriculum", CompactContactCurriculum, default_config)
registry.ALL_ENVS += ("CompactContactCurriculum",)


if __name__ == "__main__":
    app.run(train_jax_ppo.main)
