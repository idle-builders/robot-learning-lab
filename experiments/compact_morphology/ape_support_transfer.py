"""Reference-style support-transfer lesson for the sourced ape, trained from scratch.

Uses upstream Playground/Brax PPO, with a fixed proprioceptive command contract.
The reward tracks whole-body COM relative to the current double-support geometry;
upright posture and two sole contacts gate that single task reward. No guessed
servo/observation-noise randomizer or arbitrary additive reward stack is hidden
here. This is a nominal-simulation experiment, not a validated hardware twin.
"""

import jax
import jax.numpy as jp
from absl import app
from ml_collections import config_dict

from mujoco_playground import locomotion, registry
from mujoco_playground.config import locomotion_params

from playground_train import (
    CompactJoystick,
    SHAPE,
    compact_ppo_config,
    default_config as base_config,
)
from learning import train_jax_ppo


if SHAPE != "ape_sourced_v1":
    raise ValueError("ApeSupportTransfer requires COMPACT_SHAPE=ape_sourced_v1")


def default_config():
    cfg = base_config()
    # CompactJoystick loads MJCF directly and does not apply OP3's Kp/Kd.
    # Remove these inherited, ineffective settings from the printed config.
    del cfg["Kp"]
    del cfg["Kd"]
    cfg.episode_length = 100
    cfg.action_scale = 0.3  # Upstream OP3 position-target scale, not a servo specification.
    cfg.obs_noise = 0.0  # Sensor-specific noise requires identified sensor data.
    cfg.velocity_kick = [0.0, 0.0]
    cfg.reward_config.scales = config_dict.create(goal=1.0)
    return cfg


class ApeSupportTransfer(CompactJoystick):
    """Track a signed COM target while both soles stay on the floor.

    Command is one number: lateral offset as a fraction of the *initial* sole
    spacing. -0.2, 0, +0.2 span right shift, exact-zero stand, left shift.
    With quasi-static contacts, 0.2 spacing corresponds to about a 70/30 load
    split; load itself remains an independent physical evaluation measurement.
    """

    def __init__(self, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)
        self._sole_contact_adr = jp.asarray([
            self.mj_model.sensor_adr[
                self.mj_model.sensor(f"{side}_sole_floor_found").id
            ]
            for side in ("left", "right")
        ])
        self._sole_load_adr = jp.asarray([
            self.mj_model.sensor_adr[
                self.mj_model.sensor(f"{side}_sole_normal_force").id
            ]
            for side in ("left", "right")
        ])

    def sample_command(self, rng):
        # Microduck starts locomotion with 2% exact-zero commands and increases
        # that bucket only after motion works. The first ape probe used 1/3 and
        # learned to exploit the already-perfect zero-command stand instead.
        draw = jax.random.uniform(rng)
        offset = jp.where(draw < 0.02, 0.0, jp.where(draw < 0.51, -0.2, 0.2))
        return jp.asarray([offset])

    def reset(self, rng):
        state = super().reset(rng)
        sole_y = state.data.site_xpos[self._feet_site_id, 1]
        info = dict(state.info)
        info["initial_sole_mid_y"] = jp.mean(sole_y)
        info["initial_sole_spacing"] = jp.abs(sole_y[0] - sole_y[1])
        return state.replace(info=info)

    def _get_obs(self, data, info, obs_history, rng):
        del rng
        # Berkeley Lite and Microduck both expose joint velocity to the actor.
        frame = jp.concatenate([
            self.get_gyro(data),
            self.get_gravity(data),
            info["command"],
            data.qpos[7:] - self._default_pose,
            data.qvel[6:],
            info["last_act"],
        ])
        wanted = self._config.obs_history_size * frame.size
        if obs_history.shape[0] != wanted:
            obs_history = jp.zeros(wanted)
        return jp.roll(obs_history, frame.size).at[:frame.size].set(frame)

    def _get_reward(self, data, action, info, metrics, done):
        del action, metrics, done
        com_y = data.subtree_com[self._torso_body_id, 1]
        # A robot that simply drags both feet sideways has not transferred
        # support. Score COM relative to the soles at this instant, instead
        # of the feet's reset positions.
        sole_mid_y = jp.mean(data.site_xpos[self._feet_site_id, 1])
        spacing = jp.maximum(info["initial_sole_spacing"], 1e-6)
        target_y = sole_mid_y + info["command"][0] * spacing
        # A missed full commanded shift (0.2 spacing) scores exp(-1); this
        # physical error scale still has a gradient on the opposite side.
        com_match = jp.exp(-jp.abs(com_y - target_y) / (0.2 * spacing))
        # Same 1-cos(tilt) proxy and std²=0.05 as Microduck's
        # body_upright_gaussian, expressed through the ape's gravity sensor.
        upright = jp.exp(-(1.0 - self.get_gravity(data)[2]) / 0.05)
        both_soles = jp.all(data.sensordata[self._sole_contact_adr] > 0)
        return {"goal": com_match * upright * both_soles}


locomotion.register_environment("ApeSupportTransfer", ApeSupportTransfer, default_config)
registry.ALL_ENVS += ("ApeSupportTransfer",)


def ppo_config(env_name, impl=None):
    cfg = compact_ppo_config(env_name, impl)
    cfg.episode_length = 100
    cfg.num_resets_per_eval = 1
    return cfg


locomotion_params.brax_ppo_config = ppo_config


if __name__ == "__main__":
    app.run(train_jax_ppo.main)
