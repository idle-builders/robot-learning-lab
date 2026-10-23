"""Ape foot-control lessons in the upstream Playground/Brax PPO loop.

This is a curriculum on one 20-action policy contract, not authored motion.
APE_FOOT_LESSON=stand|shift|lift|setdown|step changes the reward and command phase.
The three observed command slots mean [swing_side, target_step_m, phase].
"""

import os

from absl import app
import jax
import jax.numpy as jp
from mujoco import mjx
from ml_collections import config_dict
from mujoco_playground import locomotion, registry
from mujoco_playground.config import locomotion_params

from playground_train import CompactJoystick, compact_ppo_config, default_config as base_config
from learning import train_jax_ppo


STAGE = os.environ.get("APE_FOOT_LESSON", "shift")
if STAGE not in ("stand", "shift", "lift", "setdown", "step"):
    raise ValueError(f"Unknown APE_FOOT_LESSON: {STAGE}")


def default_config():
    cfg = base_config()
    cfg.action_scale = float(os.environ.get(
        "APE_ACTION_SCALE", 0.3 if STAGE in ("stand", "shift") else 0.6
    ))
    cfg.episode_length = 100
    cfg.velocity_kick = [0.0, 0.15]
    cfg.kick_wait_times = [1.0, 2.0]
    cfg.reward_config.scales = config_dict.create(
        upright=1.0, load_shift=1.5, foot_lift=3.0, touchdown=4.0,
        stance_slip=-1.0, hand_floor=-2.0, torques=-0.0002,
        action_rate=-0.002, termination=-1.0,
        hold_xy=1.5 if STAGE == "stand" else (0.2 if STAGE == "shift" else 0.0),
        hold_velocity=1.0 if STAGE == "stand" else (0.2 if STAGE == "shift" else 0.0),
        both_soles=1.0 if STAGE in ("stand", "shift") else 0.0,
    )
    return cfg


class ApeFootLesson(CompactJoystick):
    def __init__(self, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)
        self._feet_site_ids = jp.asarray(self._feet_site_id)
        self._sole_contact = jp.array([
            self.mj_model.sensor_adr[self.mj_model.sensor(f"{side}_sole_floor_found").id]
            for side in ("left", "right")
        ])
        self._sole_load = jp.array([
            self.mj_model.sensor_adr[self.mj_model.sensor(f"{side}_sole_normal_force").id]
            for side in ("left", "right")
        ])
        self._hand_floor = jp.array([
            self.mj_model.sensor_adr[self.mj_model.sensor(f"{side}_hand_floor_found").id]
            for side in ("left", "right")
        ])

    def sample_command(self, rng):
        if STAGE == "stand":
            return jp.zeros(3)
        side_key, length_key = jax.random.split(rng)
        # Stay inside the parent joystick policy's command scale while changing
        # the sign into a left/right swing-foot cue.
        side = jp.where(jax.random.bernoulli(side_key), 0.2, -0.2)
        step_length = jax.random.uniform(
            length_key, minval=0.02 if STAGE == "step" else 0.0,
            maxval=0.05 if STAGE == "step" else 0.001,
        )
        phase = 0.0 if STAGE == "shift" else 1.0
        return jp.array([side, step_length, phase])

    def reset(self, rng):
        rng, pose_key = jax.random.split(rng)
        state = super().reset(rng)
        # Small, explicitly hypothetical start-pose variation. It prevents a
        # policy from relying on one exact neutral configuration; measured
        # actuator/contact randomization is a separate later experiment.
        joint_noise = jax.random.uniform(
            pose_key, (self.mjx_model.nu,), minval=-0.01, maxval=0.01
        )
        qpos = state.data.qpos.at[7:].add(joint_noise)
        data = mjx.forward(self.mjx_model, state.data.replace(qpos=qpos))
        side_index = jp.where(state.info["command"][0] > 0, 0, 1)
        info = dict(state.info)
        info["initial_swing_x"] = data.site_xpos[self._feet_site_ids[side_index], 0]
        info["initial_root_xy"] = data.qpos[:2]
        info["lifted_once"] = jp.array(False)
        rng, obs_key = jax.random.split(rng)
        obs = self._get_obs(data, info, jp.zeros_like(state.obs), obs_key)
        return state.replace(data=data, info=info, obs=obs)

    def _get_reward(self, data, action, info, metrics, done):
        del metrics
        swing = jp.where(info["command"][0] > 0, 0, 1)
        stance = 1 - swing
        contacts = data.sensordata[self._sole_contact] > 0
        loads = jp.maximum(data.sensordata[self._sole_load], 0.0)
        support_ratio = loads[stance] / jp.maximum(jp.sum(loads), 1e-6)
        shifted = jp.clip((support_ratio - 0.50) / 0.30, 0.0, 1.0)
        upright = jp.clip((self.get_gravity(data)[2] - 0.60) / 0.35, 0.0, 1.0)
        swing_xyz = data.site_xpos[self._feet_site_ids[swing]]
        clearance = jp.clip(swing_xyz[2] / 0.025, 0.0, 1.0)
        lifted = contacts[stance] & ~contacts[swing] & (swing_xyz[2] > 0.012)
        phase = info["command"][2]
        wants_lift = phase > 0
        wants_touchdown = phase < 0
        target_x = info["initial_swing_x"] + info["command"][1]
        touchdown_error = jp.abs(swing_xyz[0] - target_x)
        touchdown = (
            contacts[stance] & contacts[swing] & info["lifted_once"]
            & (touchdown_error < 0.018)
        )
        stance_speed = jp.linalg.norm(
            data.sensordata[self._foot_linvel_sensor_adr][stance, :2]
        )
        root_drift = jp.linalg.norm(data.qpos[:2] - info["initial_root_xy"])
        root_speed = jp.linalg.norm(data.qvel[:2])
        standing_height = jp.exp(-jp.abs(data.qpos[2] - self._home_pelvis_z) / 0.025)
        return {
            "upright": upright,
            "load_shift": (
                (STAGE != "stand") * shifted * upright * standing_height
                * contacts[stance]
            ),
            "foot_lift": wants_lift * lifted * clearance * upright,
            "touchdown": wants_touchdown * touchdown * upright,
            "stance_slip": contacts[stance] * stance_speed,
            "hold_xy": jp.exp(-root_drift / 0.03) * upright,
            "hold_velocity": jp.exp(-root_speed / 0.05) * upright,
            "both_soles": contacts[0] * contacts[1] * upright,
            "hand_floor": jp.sum(data.sensordata[self._hand_floor] > 0),
            "torques": self._cost_torques(data.actuator_force),
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info["last_last_act"]
            ),
            "termination": self._cost_termination(done, info["step"]),
        }

    def step(self, state, action):
        info = dict(state.info)
        if STAGE in ("setdown", "step"):
            phase = jp.where(info["step"] < 25, 1.0, -1.0)
            info["command"] = info["command"].at[2].set(phase)
        result = super().step(state.replace(info=info), action)
        swing = jp.where(result.info["command"][0] > 0, 0, 1)
        swing_z = result.data.site_xpos[self._feet_site_ids[swing], 2]
        contact = result.data.sensordata[self._sole_contact[swing]] > 0
        out_info = dict(result.info)
        out_info["lifted_once"] = info["lifted_once"] | ((swing_z > 0.012) & ~contact)
        return result.replace(info=out_info)


locomotion.register_environment("ApeFootLesson", ApeFootLesson, default_config)
registry.ALL_ENVS += ("ApeFootLesson",)


def ppo_config(env_name, impl=None):
    cfg = compact_ppo_config(env_name, impl)
    cfg.episode_length = 100
    cfg.num_resets_per_eval = 1
    return cfg


locomotion_params.brax_ppo_config = ppo_config


if __name__ == "__main__":
    app.run(train_jax_ppo.main)
