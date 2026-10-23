"""Fresh closed-loop physical replay of an ApeSupportTransfer PPO checkpoint."""

import argparse
import json
from pathlib import Path

from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks
import jax
import jax.numpy as jp
import numpy as np

from ape_support_transfer import ApeSupportTransfer, ppo_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path, nargs="?")
    parser.add_argument("--zero-actions", action="store_true")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.zero_actions and args.checkpoint_root is None:
        parser.error("checkpoint_root is required unless --zero-actions is set")
    env = ApeSupportTransfer()
    if args.zero_actions:
        leaf = None
        policy = None
    else:
        root = args.checkpoint_root
        leaf = root if root.name.isdigit() else max(
            (p for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        )
        leaf = leaf.resolve()
        cfg = ppo_config("ApeSupportTransfer", "jax")
        net = networks.make_ppo_networks(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
            **cfg.network_factory,
        )
        policy = jax.jit(networks.make_inference_fn(net)(checkpoint.load(leaf), deterministic=True))
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    hand_adrs = np.asarray([
        env.mj_model.sensor_adr[env.mj_model.sensor(f"{side}_hand_floor_found").id]
        for side in ("left", "right")
    ])
    torque_limits = np.asarray(env.mj_model.actuator_forcerange[:, 1])
    episodes = []

    for seed in range(args.seeds):
        for command in (-0.2, 0.0, 0.2):
            rng = jax.random.PRNGKey(20260929 + seed)
            rng, key = jax.random.split(rng)
            state = reset(key)
            info = dict(state.info)
            info["command"] = jp.asarray([command])
            rng, key = jax.random.split(rng)
            obs = env._get_obs(state.data, info, jp.zeros_like(state.obs), key)
            state = state.replace(info=info, obs=obs)
            target_relative_y = float(command * info["initial_sole_spacing"])
            trace = []
            for _ in range(args.steps):
                rng, key = jax.random.split(rng)
                action = jp.zeros(env.action_size) if args.zero_actions else policy(state.obs, key)[0]
                state = step(state, action)
                data = state.data
                loads = np.maximum(np.asarray(data.sensordata[env._sole_load_adr]), 0.0)
                contacts = np.asarray(data.sensordata[env._sole_contact_adr]) > 0
                com_y = float(data.subtree_com[env._torso_body_id, 1])
                sole_mid_y = float(jp.mean(data.site_xpos[env._feet_site_id, 1]))
                trace.append({
                    "com_y_m": round(com_y, 5),
                    "sole_mid_y_m": round(sole_mid_y, 5),
                    "relative_com_y_m": round(com_y - sole_mid_y, 5),
                    "com_error_m": round(com_y - sole_mid_y - target_relative_y, 5),
                    "left_load_share": round(float(loads[0] / max(loads.sum(), 1e-6)), 4),
                    "both_soles": bool(np.all(contacts)),
                    "goal_reward": round(float(state.metrics["reward/goal"]), 5),
                    "upright_z": round(float(env.get_gravity(data)[2]), 5),
                    "pelvis_z_m": round(float(data.qpos[2]), 5),
                    "hand_floor": bool(np.any(np.asarray(data.sensordata)[hand_adrs] > 0)),
                    "sole_speed_mps": np.round(np.linalg.norm(
                        np.asarray(data.sensordata[env._foot_linvel_sensor_adr])[:, :2], axis=1
                    ), 5).tolist(),
                    "max_torque_fraction": round(float(np.max(
                        np.abs(np.asarray(data.actuator_force)) / np.maximum(torque_limits, 1e-6)
                    )), 4),
                    "done": bool(state.done),
                })
                if bool(state.done):
                    break
            episodes.append({
                "seed": seed,
                "command": command,
                "target_relative_com_y_m": round(target_relative_y, 5),
                "steps": len(trace),
                "fell": trace[-1]["done"],
                "mean_abs_com_error_m": round(float(np.mean([abs(t["com_error_m"]) for t in trace])), 5),
                "late_left_load_share": round(float(np.mean([
                    t["left_load_share"] for t in trace[len(trace) // 2:]
                ])), 4),
                "mean_goal_reward": round(float(np.mean([t["goal_reward"] for t in trace])), 5),
                "both_soles_fraction": round(float(np.mean([t["both_soles"] for t in trace])), 4),
                "hand_floor_fraction": round(float(np.mean([t["hand_floor"] for t in trace])), 4),
                "sole_slip_m": np.round(np.sum([
                    t["sole_speed_mps"] for t in trace
                ], axis=0) * env.dt, 5).tolist(),
                "max_torque_fraction": max(t["max_torque_fraction"] for t in trace),
                "trace": trace,
            })

    result = {
        "checkpoint": None if leaf is None else str(leaf.resolve()),
        "controller": "zero-action" if args.zero_actions else "learned-ppo",
        "task": "ApeSupportTransfer",
        "episodes": episodes,
    }
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v if k != "episodes" else [
        {name: value for name, value in e.items() if name != "trace"} for e in episodes
    ] for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
