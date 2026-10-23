"""Record physically checked v5 jaw-lift teachers in Robomimic's HDF5 layout.

These are scripted demonstration inputs, not learned-policy rollouts.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

import ape_walk.task  # register local native tasks
from ape_walk.grasp import ACQUIRE_COLLISION_TASK_ID, HELD_COLLISION_TASK_ID
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def _actor(obs):
    return obs["actor"][0].detach().cpu().numpy().astype(np.float32).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=142)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--acquire", action="store_true", help="Record open-jaw close-and-lift demonstrations")
    args = parser.parse_args()
    task_id = ACQUIRE_COLLISION_TASK_ID if args.acquire else HELD_COLLISION_TASK_ID
    cfg = load_env_cfg(task_id, play=True)
    cfg.seed = args.first_seed  # record a reproducible startup friction draw
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    cfg.commands["lift_height"].resampling_time_range = (100.0, 100.0)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    robot, cube = env.scene["robot"], env.scene["cube"]
    action = env.action_manager.get_term("joint_pos")
    shoulder = action._target_names.index("right_shoulder_pitch")
    elbow = action._target_names.index("right_elbow_pitch")
    gripper = action._target_names.index("right_gripper")
    sensors = env.scene.sensors
    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as out:
        data = out.create_group("data")
        data.attrs["env_args"] = json.dumps({
            "env_name": task_id,
            "type": "mjlab",
            "env_kwargs": {},
        })
        accepted = 0
        for seed in range(args.first_seed, args.first_seed + args.episodes):
            obs, _ = env.reset(seed=seed)
            actor, next_actor, actions = [], [], []
            contact_hold = best_hold = 0
            max_height = 0.0
            both_feet_frames = 0
            initial_cube = cube.data.root_link_pos_w[0].detach().cpu().tolist()
            for step in range(round(3.0 / env.step_dt)):
                t = (step + 1) * env.step_dt
                alpha = max(0.0, min(1.0, (t - (1.0 if args.acquire else 0.4)) / 1.1))
                desired_shoulder = -0.035 + alpha * 0.137
                desired_elbow = -0.986 - alpha * 0.770
                command = torch.zeros((1, action.action_dim), device="cuda:0")
                command[0, shoulder] = (desired_shoulder - action._offset[0, shoulder]) / action._scale[0, shoulder]
                command[0, elbow] = (desired_elbow - action._offset[0, elbow]) / action._scale[0, elbow]
                if args.acquire:
                    desired_gripper = 0.35 * max(0.0, min(1.0, (t - 0.1) / 0.7))
                    command[0, gripper] = (desired_gripper - action._offset[0, gripper]) / action._scale[0, gripper]
                actor.append(_actor(obs))
                actions.append(command[0].detach().cpu().numpy().astype(np.float32).copy())
                obs, _, terminated, timed_out, _ = env.step(command)
                next_actor.append(_actor(obs))
                height = float(cube.data.root_link_pos_w[0, 2])
                max_height = max(max_height, height)
                fixed = bool(sensors["right_hand_cube_contact"].data.found[0].any())
                moving = bool(sensors["right_finger_tip_cube_contact"].data.found[0].any())
                contact_hold = contact_hold + 1 if height >= 0.04 and fixed and moving else 0
                best_hold = max(best_hold, contact_hold)
                both_feet_frames += bool(sensors["feet_ground_contact"].data.found[0].all())
                if bool(terminated[0]) or bool(timed_out[0]):
                    break
            held_s = round(best_hold * env.step_dt, 3)
            stable = both_feet_frames == len(actions) and float(robot.data.root_link_pos_w[0, 2]) > 0.09
            success = held_s >= 1.0 and stable and len(actions) == round(3.0 / env.step_dt)
            row = {
                "seed": seed,
                "initial_cube_xyz_m": [round(float(v), 6) for v in initial_cube],
                "steps": len(actions),
                "max_cube_z_m": round(max_height, 5),
                "longest_two_jaw_lift_s": held_s,
                "both_feet_grounded_fraction": round(both_feet_frames / len(actions), 3),
                "accepted": success,
            }
            if success:
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
        out.attrs["task"] = task_id
        out.attrs["teacher"] = "scripted_servo_feasibility_not_learned_policy"
    args.summary.write_text(json.dumps({
        "task": task_id,
        "teacher": "scripted_servo_feasibility_not_learned_policy",
        "dataset": str(args.out),
        "accepted": accepted,
        "attempted": args.episodes,
        "reserved_holdout_seeds": list(range(args.first_seed + 1000, args.first_seed + 1005)),
        "episodes": rows,
    }, indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    main()
