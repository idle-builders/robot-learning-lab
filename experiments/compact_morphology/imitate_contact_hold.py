"""Fine-tune the existing Brax PPO actor on successful MJX contact demos.

The network, observation normalization, simulator, and evaluator stay unchanged.
Only actor weights receive a small supervised update; the saved value net is kept.
"""

import argparse
import functools
from pathlib import Path

from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks
import jax
import jax.numpy as jp
import numpy as np
import optax

from playground_train import compact_ppo_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parent", type=Path)
    parser.add_argument("demonstrations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--extra-demonstrations", type=Path)
    parser.add_argument("--updates", type=int, default=1000)
    args = parser.parse_args()

    files = [args.demonstrations] + ([args.extra_demonstrations] if args.extra_demonstrations else [])
    observations, commanded_actions, labels = [], [], []
    for path in files:
        with np.load(path) as examples:
            for name in sorted(k.removesuffix("/obs") for k in examples.files if k.endswith("/obs")):
                observations.append(examples[f"{name}/obs"])
                commanded_actions.append(examples[f"{name}/actions"])
                labels.append(f"{path.name}:{name}")
    obs = jp.asarray(np.concatenate(observations))
    actions = jp.asarray(np.concatenate(commanded_actions))
    parent = args.parent.resolve()
    normalizer, actor, critic = checkpoint.load(str(parent))
    config = compact_ppo_config("CompactFloorReach", "jax")
    net = networks.make_ppo_networks(
        obs.shape[-1], actions.shape[-1],
        preprocess_observations_fn=running_statistics.normalize,
        **config.network_factory,
    )

    def loss_fn(params, batch_obs, batch_actions):
        logits = net.policy_network.apply(normalizer, params, batch_obs)
        prediction = net.parametric_action_distribution.mode(logits)
        return jp.mean(jp.square(prediction - batch_actions))

    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(actor)

    @jax.jit
    def update(params, state, batch_obs, batch_actions):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_obs, batch_actions)
        changes, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, changes), state, loss

    initial = float(loss_fn(actor, obs, actions))
    rng = jax.random.PRNGKey(20260927)
    for step in range(1, args.updates + 1):
        rng, batch_key = jax.random.split(rng)
        indices = jax.random.randint(batch_key, (128,), 0, len(obs))
        actor, opt_state, loss = update(actor, opt_state, obs[indices], actions[indices])
        if step in (100, 300, args.updates):
            print(step, "batch_mse", float(loss), "full_mse", float(loss_fn(actor, obs, actions)), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    save_config = checkpoint.network_config(
        jax.ShapeDtypeStruct((obs.shape[-1],), jp.float32), actions.shape[-1],
        True, functools.partial(networks.make_ppo_networks, **config.network_factory),
    )
    checkpoint.save(
        str(args.output.resolve()), args.updates,
        [normalizer, actor, critic], save_config,
    )
    print("demonstrations", labels, "transitions", len(obs), "initial_mse", initial, flush=True)
    print("saved", args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
