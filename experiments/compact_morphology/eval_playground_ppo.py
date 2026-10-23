"""Fresh-process, closed-loop measurements of an upstream Brax PPO checkpoint.

Run on the GPU after training. This is an evaluation script, not a controller:
actions come only from the checkpoint's deterministic policy.
"""

import argparse
import json
from pathlib import Path

from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks
import jax
import jax.numpy as jp
import numpy as np

from playground_train import CompactJoystick, SHAPE, compact_ppo_config


COMMANDS = {
    "stand": [0.0, 0.0, 0.0],
    "forward": [0.22, 0.0, 0.0],
    "turn_while_walking": [0.15, 0.0, 0.35],
    "slow_held_out": [0.08, 0.0, 0.0],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--walk-lesson", action="store_true", help="Evaluate the forward-walk curriculum checkpoint")
    args = parser.parse_args()

    checkpoint_dir = max(
        (p for p in args.checkpoint_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    ).resolve()
    if args.walk_lesson:
        from playground_walk import CompactForwardWalk
        env = CompactForwardWalk()
    else:
        env = CompactJoystick()
    ppo_config = compact_ppo_config("CompactJoystick", "jax")
    network = networks.make_ppo_networks(
        (env._config.obs_history_size * (9 + 2 * env.action_size),),
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **ppo_config.network_factory,
    )
    policy = networks.make_inference_fn(network)(checkpoint.load(checkpoint_dir), deterministic=True)
    total_steps = args.steps

    def rollout(rng, command):
        rng, reset_key = jax.random.split(rng)
        state = env.reset(reset_key)
        info = dict(state.info)
        info["command"] = command
        rng, obs_key = jax.random.split(rng)
        initial_obs = env._get_obs(state.data, info, jp.zeros_like(state.obs), obs_key)
        state = state.replace(info=info, obs=initial_obs)
        initial_xy = state.data.qpos[:2]

        def step(carry, _):
            state, rng = carry
            rng, action_key = jax.random.split(rng)
            action, _ = policy(state.obs, action_key)
            state = jax.lax.cond(
                state.done > 0,
                lambda s: s,
                lambda s: env.step(s, action),
                state,
            )
            velocity = env.get_local_linvel(state.data)
            gravity = env.get_gravity(state.data)
            foot_contact = jp.array([
                state.data.sensordata[env.mj_model.sensor_adr[env.mj_model.sensor(f"{side}_sole_floor_found").id]] > 0
                for side in ("left", "right")
            ])
            foot_vel = state.data.sensordata[env._foot_linvel_sensor_adr]
            foot_slip = jp.linalg.norm(foot_vel[:, :2], axis=-1) * foot_contact
            foot_height = state.data.site_xpos[env._feet_site_id, 2]
            measured = jp.array([
                state.done,
                state.reward,
                state.data.qpos[0] - initial_xy[0],
                state.data.qpos[1] - initial_xy[1],
                velocity[0],
                velocity[1],
                env.get_gyro(state.data)[2],
                gravity[2],
                jp.sum(jp.abs(state.data.actuator_force)),
                foot_contact[0], foot_contact[1],
                foot_slip[0], foot_slip[1],
                foot_height[0], foot_height[1],
            ])
            return (state, rng), measured

        _, trace = jax.lax.scan(step, (state, rng), None, length=total_steps)
        return trace

    evaluate = jax.jit(jax.vmap(rollout, in_axes=(0, None)))
    output = {
        "shape": SHAPE,
        "checkpoint": str(checkpoint_dir),
        "model_axes": env.action_size,
        "lesson": "forward_walk" if args.walk_lesson else "original_joystick",
        "sim_seconds": round(total_steps * env.dt, 3),
        "seeds": args.seeds,
        "tasks": {},
    }
    keys = jax.random.split(jax.random.PRNGKey(20260927), args.seeds)
    for name, command in COMMANDS.items():
        trace = np.asarray(evaluate(keys, jp.array(command)))
        trials = []
        for episode in trace:
            done = episode[:, 0] > 0
            survived = int(np.flatnonzero(done)[0] + 1) if done.any() else total_steps
            active = episode[:survived]
            displacement = active[-1, 2:4].tolist()
            target_velocity = np.array(command)
            velocity_error = np.linalg.norm(
                active[:, [4, 5, 6]] - target_velocity, axis=1
            )
            contact = active[:, 9:11] > 0
            trials.append({
                "survival_steps": survived,
                "fell": survived < total_steps,
                "displacement_xy_m": [round(float(v), 4) for v in displacement],
                "mean_velocity_error": round(float(velocity_error.mean()), 4),
                "mean_reward_per_step": round(float(active[:, 1].mean()), 5),
                "mean_abs_actuator_force": round(float(active[:, 8].mean()), 4),
                "upright_fraction": round(float((active[:, 7] > 0.65).mean()), 4),
                "both_soles_contact_fraction": round(float((contact.sum(axis=1) == 2).mean()), 4),
                "single_support_fraction": round(float((contact.sum(axis=1) == 1).mean()), 4),
                "sole_slip_distance_m": [round(float(v), 4) for v in (active[:, 11:13] * env.dt).sum(axis=0)],
                "max_sole_height_m": [round(float(v), 4) for v in active[:, 13:15].max(axis=0)],
                "foot_touchdowns": [int(v) for v in (contact[1:] & ~contact[:-1]).sum(axis=0)],
            })
        output["tasks"][name] = {
            "command": command,
            "survival_rate": round(sum(not t["fell"] for t in trials) / len(trials), 3),
            "mean_survival_steps": round(float(np.mean([t["survival_steps"] for t in trials])), 1),
            "mean_velocity_error": round(float(np.mean([t["mean_velocity_error"] for t in trials])), 4),
            "mean_forward_m": round(float(np.mean([t["displacement_xy_m"][0] for t in trials])), 4),
            "mean_single_support_fraction": round(float(np.mean([t["single_support_fraction"] for t in trials])), 4),
            "mean_sole_slip_distance_m": [round(float(v), 4) for v in np.mean([t["sole_slip_distance_m"] for t in trials], axis=0)],
            "trials": trials,
        }
    path = args.checkpoint_root.parent / ("forward_walk_eval.json" if args.walk_lesson else "closed_loop_eval.json")
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "shape": output["shape"],
        "checkpoint": output["checkpoint"],
        "tasks": {k: {key: value for key, value in v.items() if key != "trials"} for k, v in output["tasks"].items()},
        "output": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
