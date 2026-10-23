"""PPO contact-hold lesson reset from the approach actor's real handoff states."""

import hashlib
import json
from pathlib import Path

from absl import app
import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np

from mujoco_playground import locomotion, registry

from playground_relative_contact import CompactRelativeLateralContact, default_config
from playground_reach import REACH_MODEL
from learning import train_jax_ppo


DATASET = Path(__file__).with_name("handoff_first_contacts.npz")


class CompactHandoffHold(CompactRelativeLateralContact):
    def __init__(self, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)
        summary = json.loads(DATASET.with_suffix(".json").read_text())
        if summary["model_sha256"] != hashlib.sha256(REACH_MODEL.read_bytes()).hexdigest():
            raise ValueError("Handoff states do not match the reach model")
        with np.load(DATASET) as data:
            valid = data["found"].astype(bool)
            self._handoff_qpos = jp.asarray(data["qpos"][valid])
            self._handoff_qvel = jp.asarray(data["qvel"][valid])
            self._handoff_ctrl = jp.asarray(data["ctrl"][valid])
            self._handoff_targets = jp.asarray(data["target"][valid])
            self._handoff_obs = jp.asarray(data["obs"][valid])
            self._handoff_last_act = jp.asarray(data["last_act"][valid])
            self._handoff_last_last_act = jp.asarray(data["last_last_act"][valid])
        if len(self._handoff_targets) < 2:
            raise ValueError("Need multiple physical contact states for a handoff lesson")

    def reset(self, rng):
        state = super().reset(rng)
        _, pose_rng = jax.random.split(rng)
        index = jax.random.randint(pose_rng, (), 0, len(self._handoff_targets))
        target = self._handoff_targets[index]
        info = dict(state.info)
        info["command"] = target
        info["last_act"] = self._handoff_last_act[index]
        info["last_last_act"] = self._handoff_last_last_act[index]
        info["last_vel"] = self._handoff_qvel[index, 6:]
        info["motor_targets"] = self._handoff_ctrl[index]
        data = mjx.forward(self.mjx_model, state.data.replace(
            qpos=self._handoff_qpos[index],
            qvel=self._handoff_qvel[index],
            ctrl=self._handoff_ctrl[index],
            mocap_pos=state.data.mocap_pos.at[0].set(target),
        ))
        return state.replace(data=data, info=info, obs=self._handoff_obs[index])


locomotion.register_environment("CompactHandoffHold", CompactHandoffHold, default_config)
registry.ALL_ENVS += ("CompactHandoffHold",)


if __name__ == "__main__":
    app.run(train_jax_ppo.main)
