"""One Playground PPO lesson: hold contact from validated contact poses."""

from absl import app
import jax
import jax.numpy as jp
from mujoco import mjx

from mujoco_playground import locomotion, registry

from playground_relative_contact import CompactRelativeLateralContact, default_config
from learning import train_jax_ppo


class CompactContactHold(CompactRelativeLateralContact):
    def reset(self, rng):
        state = super().reset(rng)
        _, pose_rng, obs_rng = jax.random.split(rng, 3)
        index = jax.random.randint(pose_rng, (), 0, len(self._contact_targets))
        target = self._contact_targets[index]
        info = dict(state.info)
        info["command"] = target
        data = state.data.replace(
            qpos=self._contact_qpos[index],
            qvel=jp.zeros_like(state.data.qvel),
            ctrl=self._contact_ctrl[index],
            mocap_pos=state.data.mocap_pos.at[0].set(target),
        )
        data = mjx.forward(self.mjx_model, data)
        obs = self._get_obs(data, info, jp.zeros_like(state.obs), obs_rng)
        return state.replace(data=data, info=info, obs=obs)


locomotion.register_environment("CompactContactHold", CompactContactHold, default_config)
registry.ALL_ENVS += ("CompactContactHold",)


if __name__ == "__main__":
    app.run(train_jax_ppo.main)
