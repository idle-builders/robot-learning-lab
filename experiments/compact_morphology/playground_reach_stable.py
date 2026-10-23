"""Sustained-contact continuation of the same native Playground/Brax task."""

from absl import app
import jax.numpy as jp
from ml_collections import config_dict

from mujoco_playground import locomotion, registry

from playground_reach import CompactFloorReach, default_config as reach_config
from learning import train_jax_ppo


def default_config():
    cfg = reach_config()
    # Follow Playground manipulation tasks: survival matters and success must
    # persist. The OP3 step clips negative aggregate reward, so the terminal
    # penalty is applied after that step rather than via this scale table.
    cfg.reward_config.scales = config_dict.create(
        reach=2.0,
        touch=2.0,
        alive=5.0,
        success=500.0,
        orientation=-0.2,
        torques=-0.0001,
        action_rate=-0.002,
        termination=0.0,
    )
    return cfg


class CompactStableFloorReach(CompactFloorReach):
    def __init__(self, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)

    def reset(self, rng):
        state = super().reset(rng)
        info = dict(state.info)
        info["contact_streak"] = jp.array(0, dtype=jp.int32)
        return state.replace(info=info)

    def _get_reward(self, data, action, info, metrics, done):
        reward = super()._get_reward(data, action, info, metrics, done)
        touched = jp.any(data.sensordata[self._hand_contacts] > 0)
        upright = done == 0
        reward["touch"] = (touched & upright).astype(jp.float32)
        reward["alive"] = upright.astype(jp.float32)
        reward["success"] = (
            touched & upright & (info["contact_streak"] == 29)
        ).astype(jp.float32)
        return reward

    def step(self, state, action):
        state = super().step(state, action)
        touched = jp.any(state.data.sensordata[self._hand_contacts] > 0)
        info = dict(state.info)
        info["contact_streak"] = jp.where(
            touched & (state.done == 0), info["contact_streak"] + 1, 0
        )
        return state.replace(info=info, reward=state.reward - state.done)


locomotion.register_environment("CompactStableFloorReach", CompactStableFloorReach, default_config)
registry.ALL_ENVS += ("CompactStableFloorReach",)


if __name__ == "__main__":
    app.run(train_jax_ppo.main)
