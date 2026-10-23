"""First floor-reach lesson: targets beside the feet, within expert reach."""

from absl import app
import jax
import jax.numpy as jp
from mujoco_playground import locomotion, registry

from playground_contact_emphasis import CompactContactEmphasis, default_config
from learning import train_jax_ppo


class CompactLateralContact(CompactContactEmphasis):
    def sample_command(self, rng):
        x_key, y_key, side_key = jax.random.split(rng, 3)
        x = jax.random.uniform(x_key, minval=0.08, maxval=0.18)
        y = jax.random.uniform(y_key, minval=0.04, maxval=0.06)
        return jp.array([x, jp.where(jax.random.bernoulli(side_key), y, -y), 0.018])


locomotion.register_environment("CompactLateralContact", CompactLateralContact, default_config)
registry.ALL_ENVS += ("CompactLateralContact",)


if __name__ == "__main__":
    app.run(train_jax_ppo.main)
