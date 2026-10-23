"""Record physically checked v5 held-to-floor place demonstrations for Robomimic."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import ape_walk.task  # register native tasks
from ape_walk.place_tracking import BC_TASK_ID
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=142)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    cfg = load_env_cfg(BC_TASK_ID, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = args.first_seed
    cfg.auto_reset = False
    cfg.commands["lift_height"].resampling_time_range = (100.0, 100.0)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    robot, cube = env.scene["robot"], env.scene["cube"]
    action = env.action_manager.get_term("joint_pos")
    indexes = {name: action._target_names.index("right_" + name)
               for name in ("shoulder_pitch", "elbow_pitch", "gripper")}
    hand_site = robot.site_names.index("right_hand_center")
    sensors = env.scene.sensors
    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as out:
        data = out.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": BC_TASK_ID, "type": "mjlab", "env_kwargs": {}})
        accepted = 0
        for seed in range(args.first_seed, args.first_seed + args.episodes):
            obs, _ = env.reset(seed=seed)
            initial_cube = cube.data.root_link_pos_w[0].detach().cpu().tolist()
            actor, next_actor, actions, samples = [], [], [], []
            release = best_release = 0
            for step in range(round(3.0 / env.step_dt)):
                t = (step + 1) * env.step_dt
                lower = max(0.0, min(1.0, (t - 0.4) / 1.1))
                withdraw = max(0.0, min(1.0, (t - 1.8) / 0.7))
                target = {
                    "shoulder_pitch": 0.103 + lower * (-0.035 - 0.103) + withdraw * (0.103 + 0.035),
                    "elbow_pitch": -1.754 + lower * (-0.986 + 1.754) + withdraw * (-1.754 + 0.986),
                    "gripper": 0.35 * (1 - max(0.0, min(1.0, (t - 1.5) / 0.3))),
                }
                command = torch.zeros((1, action.action_dim), device="cuda:0")
                for name, value in target.items():
                    i = indexes[name]
                    command[0, i] = (value - action._offset[0, i]) / action._scale[0, i]
                actor.append(obs["actor"][0].detach().cpu().numpy().astype(np.float32).copy())
                actions.append(command[0].detach().cpu().numpy().astype(np.float32).copy())
                obs, _, terminated, timed_out, _ = env.step(command)
                next_actor.append(obs["actor"][0].detach().cpu().numpy().astype(np.float32).copy())
                cube_xyz = cube.data.root_link_pos_w[0].detach().cpu().numpy()
                hand_xyz = robot.data.site_pos_w[0, hand_site].detach().cpu().numpy()
                fixed = bool(sensors["right_hand_cube_contact"].data.found[0].any())
                moving = bool(sensors["right_finger_tip_cube_contact"].data.found[0].any())
                grounded = bool(sensors["feet_ground_contact"].data.found[0].all())
                placed = cube_xyz[2] <= 0.015 and not fixed and not moving and hand_xyz[2] - cube_xyz[2] >= 0.03
                release = release + 1 if placed else 0
                best_release = max(best_release, release)
                samples.append((cube_xyz.copy(), fixed, moving, grounded))
                if bool(terminated[0]) or bool(timed_out[0]):
                    break
            final_xyz = samples[-1][0]
            tail_drift = max(float(np.linalg.norm(s[0][:2] - final_xyz[:2])) for s in samples[-25:])
            goal_xy = np.array([0.08, -0.0845])
            goal_error = float(np.linalg.norm(final_xyz[:2] - goal_xy))
            held_initially = all(s[1] and s[2] for s in (samples[0], samples[19], samples[49]))
            accepted_run = (
                len(samples) == round(3.0 / env.step_dt)
                and held_initially and best_release * env.step_dt >= 0.5
                and goal_error <= 0.015 and tail_drift <= 0.002
                and all(s[3] for s in samples)
            )
            row = {
                "seed": seed,
                "initial_cube_xyz_m": [round(float(v), 6) for v in initial_cube],
                "final_cube_xyz_m": [round(float(v), 6) for v in final_xyz],
                "longest_released_place_s": round(best_release * env.step_dt, 3),
                "final_xy_goal_error_m": round(goal_error, 5),
                "last_0p5s_xy_drift_m": round(tail_drift, 5),
                "held_initially": held_initially,
                "accepted": accepted_run,
            }
            if accepted_run:
                group = data.create_group(f"demo_{accepted}")
                group.create_dataset("actions", data=np.stack(actions))
                group.create_dataset("obs/actor", data=np.stack(actor))
                group.create_dataset("next_obs/actor", data=np.stack(next_actor))
                group.create_dataset("rewards", data=np.zeros(len(actions), dtype=np.float32))
                dones = np.zeros(len(actions), dtype=np.int64)
                dones[-1] = 1
                group.create_dataset("dones", data=dones)
                group.attrs["num_samples"] = len(actions)
                group.attrs["seed"] = seed
                accepted += 1
            rows.append(row)
            print(json.dumps(row), flush=True)
        data.attrs["total"] = accepted * round(3.0 / env.step_dt)
        out.attrs["task"] = BC_TASK_ID
        out.attrs["teacher"] = "scripted_place_feasibility_not_learned_policy"
    args.summary.write_text(json.dumps({
        "task": BC_TASK_ID,
        "startup_seed": args.first_seed,
        "teacher": "scripted_place_feasibility_not_learned_policy",
        "accepted": accepted,
        "attempted": args.episodes,
        "reserved_holdout_seeds": list(range(args.first_seed + 1000, args.first_seed + 1005)),
        "episodes": rows,
    }, indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    main()
