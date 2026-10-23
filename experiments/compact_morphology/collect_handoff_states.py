"""Capture first physical contacts reached by the saved upright-start policy."""

import argparse
import hashlib
import json
from pathlib import Path

from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks
import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np

from playground_relative_contact import CompactRelativeFloorReach
from playground_reach import REACH_MODEL
from playground_train import compact_ppo_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    leaf = max((p for p in args.checkpoint_root.iterdir() if p.is_dir() and p.name.isdigit()),
               key=lambda p: int(p.name)).resolve()
    env = CompactRelativeFloorReach()
    cfg = compact_ppo_config("CompactFloorReach", "jax")
    network = networks.make_ppo_networks(
        env._config.obs_history_size * (9 + 2 * env.action_size), env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **cfg.network_factory,
    )
    policy = networks.make_inference_fn(network)(checkpoint.load(leaf), deterministic=True)

    def rollout(rng):
        rng, reset_key = jax.random.split(rng)
        state = env.reset(reset_key)
        rng, x_key, y_key, side_key = jax.random.split(rng, 4)
        x = jax.random.uniform(x_key, minval=0.08, maxval=0.18)
        y = jax.random.uniform(y_key, minval=0.04, maxval=0.06)
        target = jp.array([x, jp.where(jax.random.bernoulli(side_key), y, -y), 0.018])
        info = dict(state.info)
        info["command"] = target
        data = mjx.forward(env.mjx_model, state.data.replace(
            mocap_pos=state.data.mocap_pos.at[0].set(target)))
        state = state.replace(data=data, info=info,
                              obs=env._get_obs(data, info, jp.zeros_like(state.obs), rng))

        def step(carry, index):
            state, rng, found, qpos, qvel, ctrl, obs, last_act, last_last_act, first_step = carry
            rng, action_key = jax.random.split(rng)
            action, _ = policy(state.obs, action_key)
            state = jax.lax.cond(state.done > 0, lambda s: s, lambda s: env.step(s, action), state)
            touched = jp.any(state.data.sensordata[env._hand_contacts] > 0)
            new = touched & (state.done == 0) & ~found
            qpos = jp.where(new, state.data.qpos, qpos)
            qvel = jp.where(new, state.data.qvel, qvel)
            ctrl = jp.where(new, state.data.ctrl, ctrl)
            obs = jp.where(new, state.obs, obs)
            last_act = jp.where(new, state.info["last_act"], last_act)
            last_last_act = jp.where(new, state.info["last_last_act"], last_last_act)
            first_step = jp.where(new, index + 1, first_step)
            return (state, rng, found | new, qpos, qvel, ctrl, obs, last_act, last_last_act, first_step), None

        initial = (state, rng, jp.array(False), jp.zeros_like(state.data.qpos),
                   jp.zeros_like(state.data.qvel), jp.zeros_like(state.data.ctrl),
                   jp.zeros_like(state.obs), jp.zeros_like(state.info["last_act"]),
                   jp.zeros_like(state.info["last_last_act"]), jp.array(-1))
        final, _ = jax.lax.scan(step, initial, jp.arange(args.steps))
        return target, *final[2:]

    keys = jax.random.split(jax.random.PRNGKey(20260928), args.seeds)
    run = jax.jit(rollout)
    rows = [run(key) for key in keys]
    target, found, qpos, qvel, ctrl, obs, last_act, last_last_act, first_step = [
        np.stack([np.asarray(row[i]) for row in rows]) for i in range(9)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, target=target, found=found, qpos=qpos,
                        qvel=qvel, ctrl=ctrl, obs=obs, last_act=last_act,
                        last_last_act=last_last_act, first_step=first_step)
    summary = {"checkpoint": str(leaf), "model_sha256": hashlib.sha256(REACH_MODEL.read_bytes()).hexdigest(),
               "seeds": args.seeds, "steps": args.steps, "valid_first_contacts": int(found.sum()),
               "first_contact_steps": first_step.tolist(), "dataset": str(args.output)}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
