"""Run upstream MuJoCo Playground OP3 joystick PPO on one compact body.

Set COMPACT_SHAPE to ape_sourced_v1 (or a historical comparison body), then pass
the regular ``learning.train_jax_ppo`` flags. The PPO trainer and locomotion
reward are upstream; this adapter only maps our MJCF's names and fall height.
"""

import os
from pathlib import Path

from absl import app
import jax.numpy as jp
from mujoco import mjx
import mujoco

from mujoco_playground import locomotion, registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.op3 import joystick as op3_joystick
from mujoco_playground._src.locomotion.op3 import op3_constants
from mujoco_playground.config import locomotion_params


SHAPE = os.environ.get("COMPACT_SHAPE", "low_shoulder")
if SHAPE not in ("low_shoulder", "ape_short_torso", "ape_sourced_v1", "waist_pitch"):
    raise ValueError(f"Unknown COMPACT_SHAPE: {SHAPE}")
MODEL_PATH = Path(__file__).with_name("playground_models") / f"{SHAPE}.xml"

# The OP3 reward and observation code uses these names, not any OP3 geometry.
op3_constants.FEET_SITES = ["left_sole_center", "right_sole_center"]
op3_constants.LEFT_FEET_GEOMS = ["left_sole"]
op3_constants.RIGHT_FEET_GEOMS = ["right_sole"]
op3_constants.ROOT_BODY = "pelvis"


def default_config():
    cfg = op3_joystick.default_config()
    cfg.impl = "jax"
    cfg.sim_dt = 0.002
    cfg.episode_length = 300
    cfg.max_foot_height = 0.035
    # OP3's default broad tracking kernel can reward standing at slow speeds.
    # Train on nonzero forward commands with a sharper, shared tracking target.
    cfg.reward_config.tracking_sigma = 0.04
    cfg.lin_vel_x = [0.15, 0.30]
    cfg.lin_vel_y = [-0.08, 0.08]
    cfg.ang_vel_yaw = [-0.50, 0.50]
    return cfg


class CompactJoystick(op3_joystick.Joystick):
    """OP3's native joystick task with our joint/sole names and body model."""

    def __init__(self, config=None, config_overrides=None):
        cfg = config or default_config()
        mjx_env.MjxEnv.__init__(self, cfg, config_overrides)
        self._mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mjx_model = mjx_env.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = str(MODEL_PATH)
        self._post_init()
        self._home_pelvis_z = float(self._init_q[2])

    def _get_obs(self, data, info, obs_history, rng):
        # Upstream OP3 hardcodes a 49-D frame (20 joints). The waist candidate
        # has one extra joint, so allocate the same three-frame history correctly.
        frame_width = 9 + 2 * self.mjx_model.nu
        wanted = self._config.obs_history_size * frame_width
        if obs_history.shape[0] != wanted:
            obs_history = jp.zeros(wanted)
        return super()._get_obs(data, info, obs_history, rng)

    def _get_termination(self, data: mjx.Data):
        pelvis_z = data.xpos[self._torso_body_id, 2]
        fell = (self.get_gravity(data)[2] < 0.65) | (pelvis_z < 0.65 * self._home_pelvis_z)
        q = data.qpos[7:]
        limits = jp.any(q < self._lowers) | jp.any(q > self._uppers)
        return jp.where(self._config.early_termination, fell | limits, limits)


locomotion.register_environment("CompactJoystick", CompactJoystick, default_config)
registry.ALL_ENVS += ("CompactJoystick",)
upstream_ppo_config = locomotion_params.brax_ppo_config


def compact_ppo_config(env_name, impl=None):
    cfg = upstream_ppo_config("Op3Joystick", impl)
    cfg.num_timesteps = 8_000_000
    cfg.num_envs = 512
    cfg.num_evals = 4
    cfg.episode_length = 300
    return cfg


locomotion_params.brax_ppo_config = compact_ppo_config

from learning import train_jax_ppo  # noqa: E402  upstream flags are registered after our env


if __name__ == "__main__":
    app.run(train_jax_ppo.main)
