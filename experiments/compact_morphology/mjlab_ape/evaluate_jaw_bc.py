"""Fresh closed-loop MJLab replay of a saved Robomimic ape held-lift policy."""

import argparse
import json
from pathlib import Path

import imageio
import numpy as np
import torch

import ape_walk.task  # register native tasks
from ape_walk.grasp import ACQUIRE_COLLISION_TASK_ID, HELD_COLLISION_TASK_ID
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from robomimic.utils.file_utils import policy_from_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--qpos-out", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--first-seed", type=int, default=1142)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--phase", action="store_true", help="Append elapsed episode fraction for a phase-conditioned checkpoint")
    parser.add_argument("--acquire", action="store_true", help="Evaluate the open-jaw acquisition task")
    parser.add_argument("--xy-span-mm", type=float, help="Override cube spawn to ±millimetres around the nominal x/y center")
    args = parser.parse_args()
    task_id = ACQUIRE_COLLISION_TASK_ID if args.acquire else HELD_COLLISION_TASK_ID
    cfg = load_env_cfg(task_id, play=True)
    cfg.seed = 142  # pin startup fingertip-friction randomization; per-episode seeds remain separate
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    cfg.commands["lift_height"].resampling_time_range = (100.0, 100.0)
    if args.xy_span_mm is not None:
        span = args.xy_span_mm / 1000.0
        cfg.commands["lift_height"].object_pose_range.x = (0.080 - span, 0.080 + span)
        cfg.commands["lift_height"].object_pose_range.y = (-0.084 - span, -0.084 + span)
    if args.video:
        cfg.viewer.width, cfg.viewer.height = 640, 480
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if args.video else None)
    policy, _ = policy_from_checkpoint(ckpt_path=str(args.checkpoint), device=torch.device("cuda:0"))
    robot, cube = env.scene["robot"], env.scene["cube"]
    sensors = env.scene.sensors
    results, qposes, frames = [], [], []
    for seed in range(args.first_seed, args.first_seed + args.episodes):
        obs, _ = env.reset(seed=seed)
        initial_cube_xyz = cube.data.root_link_pos_w[0].detach().cpu().tolist()
        initial_gripper = float(robot.data.joint_pos[0, robot.joint_names.index("right_gripper")])
        policy.start_episode()
        rows, poses = [], []
        best_hold = hold = 0
        for step in range(round(3.0 / env.step_dt)):
            actor = obs["actor"][0].detach().cpu().numpy()
            if args.phase:
                actor = np.concatenate((actor, np.array([step / round(3.0 / env.step_dt)], dtype=np.float32)))
            with torch.no_grad():
                predicted = policy({"actor": actor})
            command = torch.as_tensor(predicted, dtype=torch.float32, device="cuda:0").unsqueeze(0)
            obs, _, terminated, timed_out, _ = env.step(command)
            if args.video and seed == args.first_seed and step % 2 == 0:
                frames.append(env.render())
            height = float(cube.data.root_link_pos_w[0, 2])
            jaw1 = bool(sensors["right_hand_cube_contact"].data.found[0].any())
            jaw2 = bool(sensors["right_finger_tip_cube_contact"].data.found[0].any())
            hold = hold + 1 if height >= 0.04 and jaw1 and jaw2 else 0
            best_hold = max(best_hold, hold)
            root_pose = robot.data.root_link_pose_w[0].detach().cpu().numpy()
            joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
            poses.append(np.concatenate((root_pose, joint_pos)))
            rows.append({
                "time_s": round((step + 1) * env.step_dt, 3),
                "cube_z_m": round(height, 5),
                "jaw_contacts": [jaw1, jaw2],
                "pelvis_z_m": round(float(robot.data.root_link_pos_w[0, 2]), 5),
                "both_feet_grounded": bool(sensors["feet_ground_contact"].data.found[0].all()),
                "max_leg_torque_nm": round(float(robot.data.actuator_force[0, :10].abs().max()), 4),
                "max_abs_action": round(float(command.abs().max()), 4),
            })
            if bool(terminated[0]) or bool(timed_out[0]):
                break
        qposes.append(np.stack(poses))
        result = {
            "seed": seed,
            "initial_cube_xyz_m": [round(float(v), 6) for v in initial_cube_xyz],
            "initial_gripper_rad": round(initial_gripper, 5),
            "steps": len(rows),
            "max_cube_z_m": max(row["cube_z_m"] for row in rows),
            "longest_two_jaw_lift_s": round(best_hold * env.step_dt, 3),
            "held_lift_1s": best_hold * env.step_dt >= 1.0,
            "first_two_jaw_contact_s": next((row["time_s"] for row in rows if all(row["jaw_contacts"])), None),
            "both_feet_grounded_fraction": round(sum(row["both_feet_grounded"] for row in rows) / len(rows), 3),
            "max_leg_torque_nm": max(row["max_leg_torque_nm"] for row in rows),
            "max_abs_action": max(row["max_abs_action"] for row in rows),
            "timeseries": rows,
        }
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "timeseries"}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "task": task_id,
        "xy_span_mm": args.xy_span_mm,
        "checkpoint": str(args.checkpoint),
        "status": "saved_robomimic_policy_closed_loop_replay",
        "episodes": results,
    }, indent=2) + "\n")
    if args.qpos_out:
        np.savez(args.qpos_out, **{f"seed_{r['seed']}": q for r, q in zip(results, qposes, strict=True)})
    if args.video:
        imageio.mimwrite(args.video, frames, fps=25, macro_block_size=16)
    env.close()


if __name__ == "__main__":
    main()
