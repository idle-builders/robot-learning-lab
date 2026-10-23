"""Record only physically successful native foot-transition teachers for BC."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from ape_walk.foot_transition_tracking import TASK_ID, MOTION
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def actor(obs):
    return obs["actor"][0].detach().cpu().numpy().astype(np.float32).copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--first-seed", type=int, default=142)
    parser.add_argument("--action-scale", type=float, default=3.0,
                        help="Divide native actions for Robomimic's [-1, 1] tanh actor")
    args = parser.parse_args()
    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = args.first_seed
    cfg.auto_reset = False
    cfg.commands["motion"].pose_range = {
        "x": (-0.001, 0.001), "y": (-0.001, 0.001),
        "roll": (-0.005, 0.005), "pitch": (-0.005, 0.005),
    }
    cfg.commands["motion"].joint_position_range = (-0.005, 0.005)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    robot = env.scene["robot"]
    motion = np.load(MOTION)["joint_pos"]
    action_term = env.action_manager.get_term("joint_pos")
    sole_ids = [robot.site_names.index(f"{side}_sole_center") for side in ("left", "right")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    accepted = 0
    with h5py.File(args.out, "w") as out:
        group = out.create_group("data")
        group.attrs["env_args"] = json.dumps({"env_name": TASK_ID, "type": "mjlab", "env_kwargs": {}})
        for seed in range(args.first_seed, args.first_seed + args.episodes):
            obs, _ = env.reset(seed=seed)
            observations, next_observations, actions = [], [], []
            held, total_hold = 0, 0
            first_right = None
            max_right_move = 0.0
            min_pelvis = 1.0
            hand_support = False
            for step in range(round(10.4 / env.step_dt)):
                target = torch.as_tensor(motion[step], device="cuda:0")[None, :]
                command = (target - action_term.offset) / action_term.scale
                observations.append(actor(obs))
                actions.append((command[0] / args.action_scale).detach().cpu().numpy().astype(np.float32).copy())
                obs, _, terminated, timed_out, _ = env.step(command)
                next_observations.append(actor(obs))
                soles = robot.data.site_pos_w[0, sole_ids].cpu().numpy()
                if first_right is None:
                    first_right = soles[1, :2].copy()
                max_right_move = max(max_right_move, float(np.linalg.norm(soles[1, :2] - first_right)))
                grounded = [bool(v) for v in env.scene.sensors["feet_ground_contact"].data.found[0].cpu().tolist()]
                t = (step + 1) * env.step_dt
                if 7.0 <= t < 8.0:
                    total_hold += 1
                    held += grounded == [False, True] and soles[0, 2] > 0.03
                min_pelvis = min(min_pelvis, float(robot.data.root_link_pos_w[0, 2]))
                hand_support |= bool(env.scene.sensors["hand_ground_contact"].data.found[0].any())
                if bool(terminated[0]) or bool(timed_out[0]):
                    break
            grounded = [bool(v) for v in env.scene.sensors["feet_ground_contact"].data.found[0].cpu().tolist()]
            success = (len(actions) >= round(10.4 / env.step_dt) and total_hold > 0
                       and held / total_hold >= 0.9 and grounded == [True, True]
                       and max_right_move < 0.01 and min_pelvis > 0.15 and not hand_support)
            row = {"seed": seed, "steps": len(actions),
                   "hold_single_support_fraction": round(held / total_hold, 3) if total_hold else 0.0,
                   "max_right_sole_move_m": round(max_right_move, 4),
                   "min_pelvis_z_m": round(min_pelvis, 4),
                   "final_two_foot_stand": grounded == [True, True],
                   "accepted": success}
            if success:
                demo = group.create_group(f"demo_{accepted}")
                demo.create_dataset("actions", data=np.stack(actions))
                demo.create_dataset("obs/actor", data=np.stack(observations))
                demo.create_dataset("next_obs/actor", data=np.stack(next_observations))
                demo.create_dataset("rewards", data=np.zeros(len(actions), dtype=np.float32))
                dones = np.zeros(len(actions), dtype=np.int64)
                dones[-1] = 1
                demo.create_dataset("dones", data=dones)
                demo.attrs["num_samples"] = len(actions)
                demo.attrs["seed"] = seed
                accepted += 1
            rows.append(row)
            print(json.dumps(row), flush=True)
        group.attrs["total"] = sum(d.attrs["num_samples"] for d in group.values())
        out.attrs["teacher"] = "authored_native_servo_reference_not_learned"
        out.attrs["action_scale"] = args.action_scale
    args.summary.write_text(json.dumps({"task": TASK_ID, "attempted": args.episodes,
                                        "accepted": accepted, "episodes": rows}, indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    main()
