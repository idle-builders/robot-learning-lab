"""Check whether C MuJoCo's slow contact demonstrations replay in MJX.

These are authored training demonstrations, never a learned-policy result.
"""

import json
import argparse
from pathlib import Path

from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks
import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

from playground_reach import CompactFloorReach
from playground_train import compact_ppo_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-root", type=Path)
    parser.add_argument("--aggregate", action="store_true", help="Label learner/mixed rollouts with expert actions")
    parser.add_argument("--body-relative-target", action="store_true", help="Record pelvis-relative target observations")
    parser.add_argument("--contact-bank-hold", action="store_true", help="Collect static-servo holds from validated contact poses")
    parser.add_argument("--contact-bank-dagger-cycle", type=int, help="Every Nth action comes from the policy; label visited states with bank servo")
    args = parser.parse_args()
    if args.aggregate and not args.policy_root:
        parser.error("--aggregate requires --policy-root")
    if args.aggregate and args.contact_bank_hold:
        parser.error("--aggregate and --contact-bank-hold are separate lessons")
    if args.contact_bank_dagger_cycle is not None and (
        not args.contact_bank_hold or not args.policy_root or args.contact_bank_dagger_cycle < 2
    ):
        parser.error("--contact-bank-dagger-cycle needs --contact-bank-hold, --policy-root, and N >= 2")
    bank = json.loads(Path(__file__).with_name("contact_reset_bank.json").read_text())
    if args.body_relative_target:
        from playground_relative_contact import CompactRelativeFloorReach

        env = CompactRelativeFloorReach()
    else:
        env = CompactFloorReach()
    default = np.asarray(env._default_pose)
    steps = 150
    policy = None
    if args.policy_root:
        root = args.policy_root.resolve()
        leaf = root if root.name.isdigit() else max(
            (p for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        )
        config = compact_ppo_config("CompactFloorReach", "jax")
        net = networks.make_ppo_networks(
            env._config.obs_history_size * (9 + 2 * env.action_size), env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
            **config.network_factory,
        )
        policy = networks.make_inference_fn(net)(checkpoint.load(str(leaf)), deterministic=True)

    def rollout(target, goal, beta, initial_qpos):
        rng = jax.random.PRNGKey(0)
        state = env.reset(rng)
        info = dict(state.info)
        info["command"] = target
        data = state.data.replace(
            qpos=initial_qpos if args.contact_bank_hold else state.data.qpos,
            qvel=jp.zeros_like(state.data.qvel),
            ctrl=goal if args.contact_bank_hold else state.data.ctrl,
            mocap_pos=state.data.mocap_pos.at[0].set(target),
        )
        data = mjx.forward(env.mjx_model, data)
        state = state.replace(
            data=data,
            info=info,
            obs=env._get_obs(data, info, jp.zeros_like(state.obs), rng),
        )

        def step(state, index):
            seconds = index * env.dt
            blend = jp.clip((seconds - 0.3) / 1.6, 0.0, 1.0)
            blend = blend * blend * (3.0 - 2.0 * blend)
            expert_action = (goal - env._default_pose) / env._config.action_scale
            if not args.contact_bank_hold:
                expert_action = blend * expert_action
            if policy is None:
                action = expert_action
            else:
                learned_action, _ = policy(state.obs, jax.random.PRNGKey(0))
                if args.contact_bank_dagger_cycle is not None:
                    action = jp.where(index % args.contact_bank_dagger_cycle == 0, learned_action, expert_action)
                else:
                    action = beta * expert_action + (1.0 - beta) * learned_action if args.aggregate else learned_action
            next_state = jax.lax.cond(
                state.done > 0, lambda s: s, lambda s: env.step(s, action), state
            )
            contact = jp.any(next_state.data.sensordata[env._hand_contacts] > 0)
            label = expert_action if args.aggregate or args.contact_bank_dagger_cycle is not None else action
            return next_state, (state.obs, label, next_state.data.qpos, next_state.done, contact)

        _, trace = jax.lax.scan(step, state, jp.arange(steps))
        return trace

    run = jax.jit(rollout)
    records = {}
    for beta in ((0.8, 0.4, 0.0) if args.aggregate else (0.0,)):
        for pose in bank["poses"]:
            target = np.asarray(pose["target_xyz_m"])
            if abs(target[1]) < 0.05 and not args.contact_bank_hold:
                continue
            goal = np.asarray(pose["ctrl"])
            assert np.max(np.abs((goal - default) / 1.8)) <= 1
            obs, actions, qpos, done, contact = jax.tree.map(
                np.asarray, run(jp.asarray(target), jp.asarray(goal), jp.asarray(beta), jp.asarray(pose["qpos"]))
            )
            hold = contact[95:]
            upright = ~done.astype(bool)
            print(
                pose["side"], target[:2].tolist(), "expert_mix", beta,
                "contact_hold", round(float(hold.mean()), 3),
                "upright_hold", round(float(upright[95:].mean()), 3),
                "first_contact_step", int(np.flatnonzero(contact)[0] + 1) if contact.any() else None,
                flush=True,
            )
            label = f"{pose['side']}_{target[0]:.2f}_{abs(target[1]):.2f}_beta{beta:.1f}"
            if args.aggregate:
                valid = int(np.flatnonzero(done)[0]) if done.any() else steps
                if valid < 10:
                    continue
            elif hold.mean() >= 0.8 and upright.all():
                valid = steps
            else:
                continue
            records[label + "/obs"] = obs[:valid]
            records[label + "/actions"] = actions[:valid]
            records[label + "/qpos"] = qpos[:valid]
            records[label + "/target"] = target
    if policy is None or args.aggregate or args.contact_bank_dagger_cycle is not None:
        path = Path("dagger_mjx_demos.npz" if args.aggregate else "expert_mjx_demos.npz")
        if args.contact_bank_hold:
            path = Path("dagger_mjx_contact_bank.npz" if args.contact_bank_dagger_cycle is not None else "expert_mjx_contact_bank.npz")
        if args.body_relative_target:
            path = path.with_stem(path.stem + "_relative")
        np.savez_compressed(path, **records)
        print("retained", len(records) // 4, "rollouts in", path, flush=True)
    else:
        print("policy contact holds", len(records) // 4, flush=True)


if __name__ == "__main__":
    main()
