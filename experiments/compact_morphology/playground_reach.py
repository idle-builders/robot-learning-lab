"""Floor-touch skill using the upstream OP3/Brax PPO loop and compact bodies.

Targets vary at reset. The policy sees their position in the three command
slots of the OP3 observation and controls every joint; no reach motion is
authored. Initialize from each body's locomotion checkpoint when available.
"""

import os
from pathlib import Path

from absl import app
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground import locomotion, registry
from mujoco_playground._src import mjx_env
from mujoco_playground.config import locomotion_params

import playground_train as base


REACH_MODEL = Path(__file__).with_name("reach_models") / f"{base.SHAPE}.xml"


def default_config():
    cfg = base.default_config()
    # OP3's 0.3-rad locomotion actions cannot reach the floor-touch crouch;
    # the required knee targets are about 1.5-1.75 rad in these bodies.
    cfg.action_scale = 1.8
    cfg.reward_config.scales = config_dict.create(
        reach=2.0,
        touch=5.0,
        orientation=-0.2,
        torques=-0.0001,
        action_rate=-0.002,
        termination=-1.0,
    )
    return cfg


class CompactFloorReach(base.CompactJoystick):
    def __init__(self, config=None, config_overrides=None):
        cfg = config or default_config()
        mjx_env.MjxEnv.__init__(self, cfg, config_overrides)
        self._mj_model = mujoco.MjModel.from_xml_path(str(REACH_MODEL))
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mjx_model = mjx_env.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = str(REACH_MODEL)
        self._post_init()
        self._home_pelvis_z = float(self._init_q[2])
        self._hand_sites = jp.array([
            self._mj_model.site("left_hand_center").id,
            self._mj_model.site("right_hand_center").id,
        ])
        self._hand_contacts = jp.array([
            self._mj_model.sensor_adr[self._mj_model.sensor(f"{side}_hand_target_found").id]
            for side in ("left", "right")
        ])

    def sample_command(self, rng):
        rng_x, rng_y = jax.random.split(rng)
        return jp.array([
            jax.random.uniform(rng_x, minval=0.08, maxval=0.18),
            jax.random.uniform(rng_y, minval=-0.06, maxval=0.06),
            0.018,
        ])

    def reset(self, rng):
        state = super().reset(rng)
        data = state.data.replace(
            mocap_pos=state.data.mocap_pos.at[0].set(state.info["command"])
        )
        return state.replace(data=mjx.forward(self.mjx_model, data))

    def _get_reward(self, data, action, info, metrics, done):
        del metrics
        target = data.mocap_pos[0]
        hand_distance = jp.min(jp.linalg.norm(data.site_xpos[self._hand_sites] - target, axis=1))
        touched = jp.any(data.sensordata[self._hand_contacts] > 0)
        return {
            "reach": 1.0 / (1.0 + 12.0 * hand_distance),
            "touch": touched.astype(jp.float32),
            "orientation": self._cost_orientation(self.get_gravity(data)),
            "torques": self._cost_torques(data.actuator_force),
            "action_rate": self._cost_action_rate(action, info["last_act"], info["last_last_act"]),
            "termination": self._cost_termination(done, info["step"]),
        }


locomotion.register_environment("CompactFloorReach", CompactFloorReach, default_config)
registry.ALL_ENVS += ("CompactFloorReach",)


def reach_ppo_config(env_name, impl=None):
    cfg = base.compact_ppo_config(env_name, impl)
    cfg.num_resets_per_eval = 1
    return cfg


locomotion_params.brax_ppo_config = reach_ppo_config

from learning import train_jax_ppo  # noqa: E402


if __name__ == "__main__":
    app.run(train_jax_ppo.main)
