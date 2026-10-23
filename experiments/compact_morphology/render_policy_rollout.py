"""Export a learned policy rollout; render it locally without GPU pod GL setup."""

import argparse
from pathlib import Path

from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks
import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

from playground_train import CompactJoystick, compact_ppo_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("locomotion", "reach"))
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reach-trial", type=int, default=None)
    parser.add_argument("--lateral", action="store_true")
    parser.add_argument("--body-relative-target", action="store_true")
    parser.add_argument("--switch-to-checkpoint-root", type=Path)
    parser.add_argument("--walk-lesson", action="store_true", help="Use the forward-walk lesson action scale and reward environment")
    args = parser.parse_args()
    if args.task != "reach" and (args.lateral or args.body_relative_target):
        parser.error("--lateral and --body-relative-target require reach")
    if args.switch_to_checkpoint_root and args.task != "reach":
        parser.error("--switch-to-checkpoint-root requires reach")
    if args.walk_lesson and args.task != "locomotion":
        parser.error("--walk-lesson requires locomotion")

    if args.task == "reach":
        if args.body_relative_target:
            from playground_relative_contact import CompactRelativeFloorReach

            env = CompactRelativeFloorReach()
        else:
            from playground_reach import CompactFloorReach

            env = CompactFloorReach()
        command = None
    else:
        if args.walk_lesson:
            from playground_walk import CompactForwardWalk
            env = CompactForwardWalk()
        else:
            env = CompactJoystick()
        command = jp.array([0.22, 0.0, 0.0])

    checkpoint_dir = max(
        (p for p in args.checkpoint_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    ).resolve()
    config = compact_ppo_config("CompactJoystick", "jax")
    network = networks.make_ppo_networks(
        (env._config.obs_history_size * (9 + 2 * env.action_size),),
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **config.network_factory,
    )
    policy = networks.make_inference_fn(network)(checkpoint.load(checkpoint_dir), deterministic=True)
    hold_policy = None
    if args.switch_to_checkpoint_root:
        hold_dir = max(
            (p for p in args.switch_to_checkpoint_root.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        ).resolve()
        hold_policy = networks.make_inference_fn(network)(checkpoint.load(hold_dir), deterministic=True)

    def rollout(key):
        key, reset_key = jax.random.split(key)
        state = env.reset(reset_key)
        if args.lateral:
            key, x_key, y_key, side_key = jax.random.split(key, 4)
            x = jax.random.uniform(x_key, minval=0.08, maxval=0.18)
            y = jax.random.uniform(y_key, minval=0.04, maxval=0.06)
            target = jp.array([x, jp.where(jax.random.bernoulli(side_key), y, -y), 0.018])
            info = dict(state.info)
            info["command"] = target
            data = state.data.replace(mocap_pos=state.data.mocap_pos.at[0].set(target))
            data = mjx.forward(env.mjx_model, data)
            state = state.replace(
                data=data, info=info,
                obs=env._get_obs(data, info, jp.zeros_like(state.obs), key),
            )
        if command is not None:
            info = dict(state.info)
            info["command"] = command
            key, obs_key = jax.random.split(key)
            obs = env._get_obs(state.data, info, jp.zeros_like(state.obs), obs_key)
            state = state.replace(info=info, obs=obs)

        def step(carry, _):
            state, key, switched = carry
            key, act_key = jax.random.split(key)
            action, _ = policy(state.obs, act_key)
            if hold_policy is not None:
                switched = switched | jp.any(state.data.sensordata[env._hand_contacts] > 0)
                hold_action, _ = hold_policy(state.obs, act_key)
                action = jp.where(switched, hold_action, action)
            state = jax.lax.cond(state.done > 0, lambda s: s, lambda s: env.step(s, action), state)
            contact = jp.any(state.data.sensordata[env._hand_contacts] > 0) if args.task == "reach" else jp.array(False)
            return (state, key, switched), (state.data.qpos, state.data.mocap_pos, state.done, contact, switched)

        _, trace = jax.lax.scan(step, (state, key, jp.array(False)), None, length=args.steps)
        return trace

    key = jax.random.PRNGKey(args.seed)
    if args.reach_trial is not None:
        if args.task != "reach" or not 0 <= args.reach_trial < 24:
            raise ValueError("--reach-trial requires reach and an index from 0 to 23")
        key = jax.random.split(jax.random.PRNGKey(20260928), 24)[args.reach_trial]
    qpos, mocap_pos, done, contact, switched = jax.jit(rollout)(key)
    qpos, mocap_pos, done, contact, switched = (np.asarray(x) for x in (qpos, mocap_pos, done, contact, switched))
    label = ("_lateral" if args.lateral else "") + ("_relative" if args.body_relative_target else "")
    label += f"_trial{args.reach_trial}" if args.reach_trial is not None else ""
    label += "_skill_switch" if args.switch_to_checkpoint_root else ""
    path = args.checkpoint_root.parent / f"{args.task}_learned_rollout{label}.npz"
    np.savez_compressed(path, qpos=qpos, mocap_pos=mocap_pos, done=done, contact=contact, switched=switched)
    streak = longest = 0
    for touched in contact[:int(np.flatnonzero(done > 0)[0] + 1) if (done > 0).any() else len(contact)]:
        streak = streak + 1 if touched else 0
        longest = max(longest, streak)
    print(path, "bytes", path.stat().st_size, "fell", bool((done > 0).any()), "longest_contact_streak", longest)


if __name__ == "__main__":
    main()
