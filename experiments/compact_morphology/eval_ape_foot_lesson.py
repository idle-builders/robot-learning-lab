"""Inspect learned ape foot lessons with fresh, closed-loop starts.

Actions come only from a Brax PPO checkpoint. Prints physical measurements,
not a reward-only pass/fail result.
"""

import argparse
import json
from pathlib import Path

from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks
import jax
import jax.numpy as jp
import numpy as np

from playground_foot_lesson import ApeFootLesson, STAGE, ppo_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.checkpoint_root
    leaf = root if root.name.isdigit() else max(
        (p for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    leaf = leaf.resolve()
    env = ApeFootLesson()
    cfg = ppo_config("ApeFootLesson", "jax")
    net = networks.make_ppo_networks(
        (env._config.obs_history_size * (9 + 2 * env.action_size),),
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **cfg.network_factory,
    )
    policy = jax.jit(networks.make_inference_fn(net)(checkpoint.load(leaf), deterministic=True))
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    observed = []

    for seed in range(args.seeds):
        rng = jax.random.PRNGKey(20260928 + seed)
        rng, key = jax.random.split(rng)
        state = reset(key)
        side = float(state.info["command"][0])
        swing = 0 if side > 0 else 1
        initial_xy = np.asarray(state.data.qpos[:2])
        trace = []
        weighted_rewards = {}
        for _ in range(args.steps):
            rng, key = jax.random.split(rng)
            action, _ = policy(state.obs, key)
            state = step(state, action)
            for name, value in state.metrics.items():
                if name.startswith("reward/"):
                    weighted_rewards.setdefault(name[7:], []).append(float(value))
            data = state.data
            loads = np.maximum(np.asarray(data.sensordata[env._sole_load]), 0)
            contacts = np.asarray(data.sensordata[env._sole_contact]) > 0
            stance = 1 - swing
            trace.append({
                "support_ratio": float(loads[stance] / max(loads.sum(), 1e-6)),
                "swing_contact": bool(contacts[swing]),
                "both_soles_contact": bool(np.all(contacts)),
                "swing_height_m": float(data.site_xpos[env._feet_site_ids[swing], 2]),
                "stance_slip_mps": float(np.linalg.norm(np.asarray(data.sensordata[env._foot_linvel_sensor_adr])[stance, :2])),
                "root_xy_m": np.asarray(data.qpos[:2]).tolist(),
                "pelvis_height_m": float(data.qpos[2]),
                "upright_z": float(env.get_gravity(data)[2]),
                "hand_floor_count": int(np.sum(np.asarray(data.sensordata[env._hand_floor]) > 0)),
                "done": bool(state.done),
            })
            if bool(state.done):
                break
        observed.append({
            "side": "stand" if STAGE == "stand" else ("left_swing" if side > 0 else "right_swing"),
            "seed": seed,
            "steps": len(trace),
            "fell": trace[-1]["done"],
            "mean_support_ratio": round(float(np.mean([t["support_ratio"] for t in trace])), 3),
            "late_support_ratio": round(float(np.mean([t["support_ratio"] for t in trace[len(trace) // 2:]])), 3),
            "max_support_ratio": round(max(t["support_ratio"] for t in trace), 3),
            "controlled_shift_fraction": round(float(np.mean([
                t["support_ratio"] > 0.7 and t["both_soles_contact"]
                and t["upright_z"] > 0.95 and not t["done"]
                for t in trace
            ])), 3),
            "single_support_fraction": round(float(np.mean([not t["swing_contact"] for t in trace])), 3),
            "both_soles_contact_fraction": round(float(np.mean([t["both_soles_contact"] for t in trace])), 3),
            "root_drift_m": round(float(np.linalg.norm(np.asarray(trace[-1]["root_xy_m"]) - initial_xy)), 4),
            "max_swing_height_m": round(max(t["swing_height_m"] for t in trace), 4),
            "stance_slip_m": round(sum(t["stance_slip_mps"] for t in trace) * env.dt, 4),
            "min_pelvis_height_m": round(min(t["pelvis_height_m"] for t in trace), 4),
            "min_upright_z": round(min(t["upright_z"] for t in trace), 3),
            "hand_floor_fraction": round(float(np.mean([t["hand_floor_count"] > 0 for t in trace])), 3),
            # Upstream OP3 records weighted reward components before multiplying
            # their sum by dt and clipping negative totals to zero.
            "mean_weighted_reward_terms": {
                name: round(float(np.mean(values)), 5)
                for name, values in weighted_rewards.items()
            },
        })

    output = {"checkpoint": str(leaf), "lesson": STAGE, "episodes": observed}
    if args.output:
        args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
