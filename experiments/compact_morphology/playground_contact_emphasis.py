"""Contact-first reward variant of the same ape MJX curriculum task."""

from absl import app
from mujoco_playground import locomotion, registry

from playground_contact_curriculum import CompactContactCurriculum
from playground_reach_stable import default_config as stable_config
from learning import train_jax_ppo


def default_config():
    config = stable_config()
    # The previous task paid 5 for survival and 2 for contact each step.
    # The upstream Playground manipulation task pays separately for waiting
    # in a successful state. Here the only task condition is hand contact.
    config.reward_config.scales.alive = 1.0
    config.reward_config.scales.touch = 10.0
    return config


class CompactContactEmphasis(CompactContactCurriculum):
    def __init__(self, config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)


locomotion.register_environment("CompactContactEmphasis", CompactContactEmphasis, default_config)
registry.ALL_ENVS += ("CompactContactEmphasis",)


if __name__ == "__main__":
    app.run(train_jax_ppo.main)
