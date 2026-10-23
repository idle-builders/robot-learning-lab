"""First compact locomotion lesson: walk forward with alternating foot support.

Continue the existing balance checkpoint in upstream Playground/Brax PPO. The
robot, action interface, impedance, and OP3 rewards stay the same; this lesson
restricts commands to forward-or-stand and adds Berkeley Lite's biped
single-support reward. The existing OP3 sole-slip cost remains active.
"""

import os

import jax
import jax.numpy as jp

from absl import app
from mujoco_playground import locomotion, registry
from mujoco_playground.config import locomotion_params

from playground_train import CompactJoystick, default_config, compact_ppo_config
from learning import train_jax_ppo


def walk_config():
    cfg = default_config()
    cfg.action_scale = float(os.environ.get("COMPACT_WALK_ACTION_SCALE", "0.3"))
    cfg.reward_config.scales.single_stance = 2.0
    return cfg


class CompactForwardWalk(CompactJoystick):
    def __init__(self, config=None, config_overrides=None):
        super().__init__(config or walk_config(), config_overrides)
        self._sole_contact_adr = jp.array([
            self.mj_model.sensor_adr[self.mj_model.sensor(f"{side}_sole_floor_found").id]
            for side in ("left", "right")
        ])

    def sample_command(self, rng):
        if os.environ.get("COMPACT_WALK_STAGE", "forward") == "balance":
            return jp.zeros(3)
        return jp.where(
            jax.random.bernoulli(rng, 0.1),
            jp.zeros(3),
            jp.array([0.22, 0.0, 0.0]),
        )

    def _contacts(self, data):
        return data.sensordata[self._sole_contact_adr] > 0

    def reset(self, rng):
        state = super().reset(rng)
        contact = self._contacts(state.data)
        info = dict(state.info)
        info["previous_sole_contact"] = contact
        info["sole_stance_s"] = jp.zeros(2)
        info["sole_air_s"] = jp.zeros(2)
        return state.replace(info=info)

    def _get_reward(self, data, action, info, metrics, done):
        rewards = super()._get_reward(data, action, info, metrics, done)
        contact = self._contacts(data)
        same_mode = contact == info["previous_sole_contact"]
        stance_s = jp.where(contact, jp.where(same_mode, info["sole_stance_s"] + self.dt, self.dt), 0.0)
        air_s = jp.where(~contact, jp.where(same_mode, info["sole_air_s"] + self.dt, self.dt), 0.0)
        mode_s = jp.where(contact, stance_s, air_s)
        one_support = jp.sum(contact.astype(jp.int32)) == 1
        moving = jp.linalg.norm(info["command"][:2]) > 0.1
        rewards["single_stance"] = jp.where(one_support & moving, jp.minimum(jp.min(mode_s), 0.5), 0.0)
        return rewards

    def step(self, state, action):
        result = super().step(state, action)
        contact = self._contacts(result.data)
        same_mode = contact == state.info["previous_sole_contact"]
        info = dict(result.info)
        info["sole_stance_s"] = jp.where(contact, jp.where(same_mode, state.info["sole_stance_s"] + self.dt, self.dt), 0.0)
        info["sole_air_s"] = jp.where(~contact, jp.where(same_mode, state.info["sole_air_s"] + self.dt, self.dt), 0.0)
        info["previous_sole_contact"] = contact
        return result.replace(info=info)


locomotion.register_environment("CompactForwardWalk", CompactForwardWalk, walk_config)
registry.ALL_ENVS += ("CompactForwardWalk",)


def walk_ppo_config(env_name, impl=None):
    cfg = compact_ppo_config(env_name, impl)
    # Brax otherwise batches ten reset cycles before each checkpoint and can
    # train far past --num_timesteps after rounding to its native update size.
    cfg.num_resets_per_eval = 1
    return cfg


locomotion_params.brax_ppo_config = walk_ppo_config


if __name__ == "__main__":
    app.run(train_jax_ppo.main)
