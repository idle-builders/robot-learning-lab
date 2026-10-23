"""Same contact task with the floor target observed in the pelvis frame."""

from absl import app
import jax
import jax.numpy as jp

from mujoco_playground import locomotion, registry

from playground_contact_emphasis import CompactContactEmphasis, default_config
from playground_contact_lateral import CompactLateralContact
from playground_reach import CompactFloorReach
from learning import train_jax_ppo


class BodyRelativeTargetObservation:
    def _get_obs(self, data, info, obs_history, rng):
        pelvis = self._torso_body_id
        to_target = data.mocap_pos[0] - data.xpos[pelvis]
        local_target = data.xmat[pelvis].T @ to_target
        observed_info = dict(info)
        observed_info["command"] = local_target
        return super()._get_obs(data, observed_info, obs_history, rng)

    def reset(self, rng):
        state = super().reset(rng)
        _, obs_rng = jax.random.split(rng)
        obs = self._get_obs(state.data, state.info, jp.zeros_like(state.obs), obs_rng)
        return state.replace(obs=obs)


class CompactRelativeFloorReach(BodyRelativeTargetObservation, CompactFloorReach):
    """Upright-start physical evaluation with the same observation contract."""


class CompactRelativeContact(BodyRelativeTargetObservation, CompactContactEmphasis):
    """Contact-reset training task; only the three target observation slots differ."""


class CompactRelativeLateralContact(BodyRelativeTargetObservation, CompactLateralContact):
    """Lateral lesson with only the target observation slots changed."""


locomotion.register_environment("CompactRelativeContact", CompactRelativeContact, default_config)
registry.ALL_ENVS += ("CompactRelativeContact",)
locomotion.register_environment("CompactRelativeLateralContact", CompactRelativeLateralContact, default_config)
registry.ALL_ENVS += ("CompactRelativeLateralContact",)


if __name__ == "__main__":
    app.run(train_jax_ppo.main)
