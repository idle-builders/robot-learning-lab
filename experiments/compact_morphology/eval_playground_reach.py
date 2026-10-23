"""Fresh-process floor-target contact evaluation for learned reach policies."""

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

from playground_reach import CompactFloorReach
from playground_train import SHAPE, compact_ppo_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--scalar", action="store_true", help="Run each seed as its own JIT environment")
    parser.add_argument("--lateral", action="store_true", help="Use reachable-side targets with |y| in 4–6 cm")
    parser.add_argument("--body-relative-target", action="store_true", help="Observe target coordinates in the pelvis frame")
    parser.add_argument("--contact-reset-bank", action="store_true", help="Start from each physics-checked near-contact pose")
    parser.add_argument("--handoff-reset-bank", action="store_true", help="Start from each recorded first-contact state of the approach policy")
    parser.add_argument("--expert-bank-pose", action="store_true", help="Hold each bank's servo targets instead of a learned policy")
    parser.add_argument("--expert-takeover-step", type=int, help="Switch from policy to bank servo after this many control steps")
    parser.add_argument("--switch-to-checkpoint-root", type=Path, help="Use a second learned policy after first physical hand contact")
    parser.add_argument("--switch-to-bank-servo", action="store_true", help="Diagnostic: switch to nearest validated pose servo after contact")
    args = parser.parse_args()
    if args.expert_bank_pose and not args.contact_reset_bank:
        parser.error("--expert-bank-pose requires --contact-reset-bank")
    if args.expert_takeover_step is not None and (not args.contact_reset_bank or args.expert_bank_pose):
        parser.error("--expert-takeover-step requires --contact-reset-bank and a learned policy")
    if args.lateral and args.contact_reset_bank:
        parser.error("--lateral and --contact-reset-bank select different target sets")
    if args.handoff_reset_bank and (args.contact_reset_bank or args.lateral):
        parser.error("--handoff-reset-bank selects its own start and target distribution")
    if args.switch_to_checkpoint_root and (args.expert_bank_pose or args.expert_takeover_step is not None):
        parser.error("--switch-to-checkpoint-root cannot be combined with a servo controller")
    if args.switch_to_bank_servo and (args.switch_to_checkpoint_root or args.expert_bank_pose or args.expert_takeover_step is not None):
        parser.error("--switch-to-bank-servo needs the first learned policy only")

    checkpoint_dir = max(
        (p for p in args.checkpoint_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    ).resolve()
    if args.body_relative_target:
        from playground_relative_contact import CompactRelativeFloorReach

        env = CompactRelativeFloorReach()
    else:
        env = CompactFloorReach()
    ppo_config = compact_ppo_config("CompactFloorReach", "jax")
    network = networks.make_ppo_networks(
        (env._config.obs_history_size * (9 + 2 * env.action_size),),
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **ppo_config.network_factory,
    )
    policy = None if args.expert_bank_pose else networks.make_inference_fn(network)(checkpoint.load(checkpoint_dir), deterministic=True)
    switch_policy = None
    switch_checkpoint_dir = None
    if args.switch_to_checkpoint_root:
        switch_checkpoint_dir = max(
            (p for p in args.switch_to_checkpoint_root.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        ).resolve()
        switch_policy = networks.make_inference_fn(network)(checkpoint.load(switch_checkpoint_dir), deterministic=True)
    if args.contact_reset_bank or args.switch_to_bank_servo:
        bank = json.loads(Path(__file__).with_name("contact_reset_bank.json").read_text())
        if bank["model_sha256"] != hashlib.sha256(Path(env._xml_path).read_bytes()).hexdigest():
            raise ValueError("Contact reset bank does not match the reach model")
        bank_qpos = jp.asarray([pose["qpos"] for pose in bank["poses"]])
        bank_ctrl = jp.asarray([pose["ctrl"] for pose in bank["poses"]])
        bank_target = jp.asarray([pose["target_xyz_m"] for pose in bank["poses"]])
        if args.contact_reset_bank and args.seeds != len(bank["poses"]):
            parser.error(f"--contact-reset-bank requires --seeds={len(bank['poses'])}")
    if args.handoff_reset_bank:
        path = Path(__file__).with_name("handoff_first_contacts.npz")
        summary = json.loads(path.with_suffix(".json").read_text())
        if summary["model_sha256"] != hashlib.sha256(Path(env._xml_path).read_bytes()).hexdigest():
            raise ValueError("Handoff states do not match the reach model")
        with np.load(path) as saved:
            valid = saved["found"].astype(bool)
            handoff_qpos = jp.asarray(saved["qpos"][valid])
            handoff_qvel = jp.asarray(saved["qvel"][valid])
            handoff_ctrl = jp.asarray(saved["ctrl"][valid])
            handoff_target = jp.asarray(saved["target"][valid])
            handoff_obs = jp.asarray(saved["obs"][valid])
            handoff_last_act = jp.asarray(saved["last_act"][valid])
            handoff_last_last_act = jp.asarray(saved["last_last_act"][valid])
        if args.seeds != len(handoff_target):
            parser.error(f"--handoff-reset-bank requires --seeds={len(handoff_target)}")

    def rollout(rng, index):
        rng, reset_key = jax.random.split(rng)
        state = env.reset(reset_key)
        if args.handoff_reset_bank:
            target_pose = handoff_target[index]
            info = dict(state.info)
            info["command"] = target_pose
            info["last_act"] = handoff_last_act[index]
            info["last_last_act"] = handoff_last_last_act[index]
            info["last_vel"] = handoff_qvel[index, 6:]
            info["motor_targets"] = handoff_ctrl[index]
            data = mjx.forward(env.mjx_model, state.data.replace(
                qpos=handoff_qpos[index], qvel=handoff_qvel[index], ctrl=handoff_ctrl[index],
                mocap_pos=state.data.mocap_pos.at[0].set(target_pose),
            ))
            state = state.replace(
                data=data, info=info,
                obs=handoff_obs[index],
            )
        elif args.contact_reset_bank:
            target_pose = bank_target[index]
            info = dict(state.info)
            info["command"] = target_pose
            data = state.data.replace(
                qpos=bank_qpos[index],
                qvel=jp.zeros_like(state.data.qvel),
                ctrl=bank_ctrl[index],
                mocap_pos=state.data.mocap_pos.at[0].set(target_pose),
            )
            data = mjx.forward(env.mjx_model, data)
            state = state.replace(
                data=data, info=info,
                obs=env._get_obs(data, info, jp.zeros_like(state.obs), rng),
            )
        elif args.lateral:
            rng, x_key, y_key, side_key = jax.random.split(rng, 4)
            x = jax.random.uniform(x_key, minval=0.08, maxval=0.18)
            y = jax.random.uniform(y_key, minval=0.04, maxval=0.06)
            y = jp.where(jax.random.bernoulli(side_key), y, -y)
            lateral_target = jp.array([x, y, 0.018])
            info = dict(state.info)
            info["command"] = lateral_target
            data = state.data.replace(mocap_pos=state.data.mocap_pos.at[0].set(lateral_target))
            data = mjx.forward(env.mjx_model, data)
            state = state.replace(
                data=data, info=info,
                obs=env._get_obs(data, info, jp.zeros_like(state.obs), rng),
            )
        target = state.info["command"]

        def step(carry, step_index):
            state, rng, switched = carry
            rng, action_key = jax.random.split(rng)
            if args.expert_bank_pose:
                action = (bank_ctrl[index] - env._default_pose) / env._config.action_scale
            else:
                action, _ = policy(state.obs, action_key)
                if args.expert_takeover_step is not None:
                    servo_action = (bank_ctrl[index] - env._default_pose) / env._config.action_scale
                    action = jp.where(step_index >= args.expert_takeover_step, servo_action, action)
                if switch_policy is not None:
                    switched = switched | jp.any(state.data.sensordata[env._hand_contacts] > 0)
                    hold_action, _ = switch_policy(state.obs, action_key)
                    action = jp.where(switched, hold_action, action)
                if args.switch_to_bank_servo:
                    switched = switched | jp.any(state.data.sensordata[env._hand_contacts] > 0)
                    nearest = jp.argmin(jp.linalg.norm(bank_target - target, axis=1))
                    servo_action = (bank_ctrl[nearest] - env._default_pose) / env._config.action_scale
                    action = jp.where(switched, servo_action, action)
            state = jax.lax.cond(
                state.done > 0,
                lambda s: s,
                lambda s: env.step(s, action),
                state,
            )
            contacts = state.data.sensordata[env._hand_contacts]
            distance = jp.min(
                jp.linalg.norm(state.data.site_xpos[env._hand_sites] - target, axis=1)
            )
            measured = jp.array([
                state.done,
                jp.any(contacts > 0),
                distance,
                state.data.xpos[env._torso_body_id, 2],
                jp.sum(jp.abs(state.data.actuator_force)),
                switched,
            ])
            return (state, rng, switched), measured

        _, trace = jax.lax.scan(step, (state, rng, jp.array(False)), jp.arange(args.steps))
        return target, trace

    keys = jax.random.split(jax.random.PRNGKey(20260928), args.seeds)
    if args.scalar:
        run = jax.jit(rollout)
        rows = [run(key, jp.asarray(i)) for i, key in enumerate(keys)]
        targets = jp.stack([row[0] for row in rows])
        traces = jp.stack([row[1] for row in rows])
    else:
        targets, traces = jax.jit(jax.vmap(rollout))(keys, jp.arange(args.seeds))
    targets = np.asarray(targets)
    traces = np.asarray(traces)
    trials = []
    for trial_index, (target, trace) in enumerate(zip(targets, traces)):
        done = trace[:, 0] > 0
        first_done = int(np.flatnonzero(done)[0] + 1) if done.any() else args.steps
        contacts = trace[:first_done, 1] > 0
        streak = 0
        longest_streak = 0
        for touched in contacts:
            streak = streak + 1 if touched else 0
            longest_streak = max(longest_streak, streak)
        trials.append({
            "pose_index": trial_index if args.contact_reset_bank or args.handoff_reset_bank else None,
            "target_xyz_m": [round(float(v), 4) for v in target],
            "touch": bool(contacts.any()),
            "touch_and_survive": bool(contacts.any() and first_done == args.steps),
            "first_touch_step": int(np.flatnonzero(contacts)[0] + 1) if contacts.any() else None,
            "contact_steps": int(contacts.sum()),
            "longest_contact_streak_steps": longest_streak,
            "sustained_contact_30": longest_streak >= 30,
            "survival_steps": first_done,
            "min_hand_distance_m": round(float(trace[:first_done, 2].min()), 4),
            "min_pelvis_height_m": round(float(trace[:first_done, 3].min()), 4),
            "mean_abs_actuator_force": round(float(trace[:first_done, 4].mean()), 4),
            "switched_to_hold": bool(np.any(trace[:first_done, 5] > 0)),
        })
    result = {
        "shape": SHAPE,
        "checkpoint": str(checkpoint_dir),
        "seeds": args.seeds,
        "steps": args.steps,
        "evaluation_mode": "single_environment" if args.scalar else "batched_24_environments",
        "target_distribution": "approach_contact_state_bank" if args.handoff_reset_bank else "contact_pose_bank" if args.contact_reset_bank else "lateral" if args.lateral else "broad",
        "target_observation": "pelvis_relative" if args.body_relative_target else "world_xyz",
        "controller": "bank_servo_expert" if args.expert_bank_pose else f"servo_takeover_after_{args.expert_takeover_step}" if args.expert_takeover_step is not None else "learned_reach_then_learned_hold" if args.switch_to_checkpoint_root else "learned_reach_then_nearest_bank_servo" if args.switch_to_bank_servo else "learned_checkpoint",
        "hold_checkpoint": str(switch_checkpoint_dir) if switch_checkpoint_dir else None,
        "touch_rate": round(sum(t["touch"] for t in trials) / args.seeds, 3),
        "touch_and_survive_rate": round(sum(t["touch_and_survive"] for t in trials) / args.seeds, 3),
        "sustained_contact_30_rate": round(sum(t["sustained_contact_30"] for t in trials) / args.seeds, 3),
        "mean_min_hand_distance_m": round(float(np.mean([t["min_hand_distance_m"] for t in trials])), 4),
        "mean_survival_steps": round(float(np.mean([t["survival_steps"] for t in trials])), 1),
        "longest_contact_streak_steps": max(t["longest_contact_streak_steps"] for t in trials),
        "trials": trials,
    }
    label = "floor_reach_eval" + ("_lateral" if args.lateral else "") + ("_scalar" if args.scalar else "")
    label += "_relative" if args.body_relative_target else ""
    label += "_bank" if args.contact_reset_bank else ""
    label += "_handoff" if args.handoff_reset_bank else ""
    label += "_expert" if args.expert_bank_pose else ""
    label += f"_takeover_{args.expert_takeover_step}" if args.expert_takeover_step is not None else ""
    label += "_skill_switch" if args.switch_to_checkpoint_root else ""
    label += "_servo_switch" if args.switch_to_bank_servo else ""
    path = args.checkpoint_root.parent / (label + ".json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "trials"}, indent=2))


if __name__ == "__main__":
    main()
