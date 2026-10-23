"""Fresh closed-loop MJLab replay of a saved Robomimic held-to-floor place policy."""

import argparse
import json
from pathlib import Path

import imageio
import numpy as np
import torch

import ape_walk.task  # register native tasks
from ape_walk.place_tracking import BC_TASK_ID
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from robomimic.utils.file_utils import policy_from_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--first-seed", type=int, default=1142)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    cfg = load_env_cfg(BC_TASK_ID, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = 142
    cfg.auto_reset = False
    cfg.commands["lift_height"].resampling_time_range = (100.0, 100.0)
    if args.video:
        cfg.viewer.width, cfg.viewer.height = 640, 480
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if args.video else None)
    policy, _ = policy_from_checkpoint(ckpt_path=str(args.checkpoint), device=torch.device("cuda:0"))
    robot, cube = env.scene["robot"], env.scene["cube"]
    sensors = env.scene.sensors
    hand_site = robot.site_names.index("right_hand_center")
    action_term = env.action_manager.get_term("joint_pos")
    tracked = ("right_shoulder_pitch", "right_elbow_pitch", "right_gripper")
    action_indexes = {name: action_term._target_names.index(name) for name in tracked}
    joint_indexes = {name: robot.joint_names.index(name) for name in tracked}
    results, frames = [], []
    for seed in range(args.first_seed, args.first_seed + args.episodes):
        obs, _ = env.reset(seed=seed)
        initial_cube = cube.data.root_link_pos_w[0].detach().cpu().tolist()
        policy.start_episode()
        rows, release = [], 0
        best_release = 0
        for step in range(round(3.0 / env.step_dt)):
            actor = obs["actor"][0].detach().cpu().numpy()
            actor = np.concatenate((actor, np.array([step / round(3.0 / env.step_dt)], dtype=np.float32)))
            with torch.no_grad():
                predicted = policy({"actor": actor})
            command = torch.as_tensor(predicted, dtype=torch.float32, device="cuda:0").unsqueeze(0)
            obs, _, terminated, timed_out, _ = env.step(command)
            if args.video and seed == args.first_seed and step % 2 == 0:
                frames.append(env.render())
            cube_xyz = cube.data.root_link_pos_w[0].detach().cpu().numpy()
            hand_xyz = robot.data.site_pos_w[0, hand_site].detach().cpu().numpy()
            fixed = bool(sensors["right_hand_cube_contact"].data.found[0].any())
            moving = bool(sensors["right_finger_tip_cube_contact"].data.found[0].any())
            grounded = bool(sensors["feet_ground_contact"].data.found[0].all())
            placed = cube_xyz[2] <= 0.015 and not fixed and not moving and hand_xyz[2] - cube_xyz[2] >= 0.03
            release = release + 1 if placed else 0
            best_release = max(best_release, release)
            rows.append({
                "time_s": round((step + 1) * env.step_dt, 3),
                "cube_xyz_m": [round(float(v), 5) for v in cube_xyz],
                "hand_xyz_m": [round(float(v), 5) for v in hand_xyz],
                "jaw_contacts": [fixed, moving],
                "both_feet_grounded": grounded,
                "max_leg_torque_nm": round(float(robot.data.actuator_force[0, :10].abs().max()), 4),
                "max_abs_action": round(float(command.abs().max()), 4),
                "tracked_action": {name: round(float(command[0, action_indexes[name]]), 4) for name in tracked},
                "tracked_joint_rad": {name: round(float(robot.data.joint_pos[0, joint_indexes[name]]), 4) for name in tracked},
            })
            if bool(terminated[0]) or bool(timed_out[0]):
                break
        final_xy = np.array(rows[-1]["cube_xyz_m"][:2])
        goal_error = float(np.linalg.norm(final_xy - np.array([0.08, -0.0845])))
        tail_drift = max(float(np.linalg.norm(np.array(r["cube_xyz_m"][:2]) - final_xy)) for r in rows[-25:])
        retained_before_release = len(rows) > 50 and all(all(rows[i]["jaw_contacts"]) for i in (0, 19, 49))
        success = (
            len(rows) == round(3.0 / env.step_dt) and retained_before_release
            and best_release * env.step_dt >= 0.5 and goal_error <= 0.015
            and tail_drift <= 0.002 and all(r["both_feet_grounded"] for r in rows)
        )
        result = {
            "seed": seed,
            "initial_cube_xyz_m": [round(float(v), 6) for v in initial_cube],
            "final_cube_xyz_m": rows[-1]["cube_xyz_m"],
            "longest_released_place_s": round(best_release * env.step_dt, 3),
            "final_xy_goal_error_m": round(goal_error, 5),
            "last_0p5s_xy_drift_m": round(tail_drift, 5),
            "retained_before_release": retained_before_release,
            "both_feet_grounded_fraction": round(sum(r["both_feet_grounded"] for r in rows) / len(rows), 3),
            "max_leg_torque_nm": max(r["max_leg_torque_nm"] for r in rows),
            "max_abs_action": max(r["max_abs_action"] for r in rows),
            "placed_released_0p5s": success,
            "timeseries": rows,
        }
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "timeseries"}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "task": BC_TASK_ID,
        "checkpoint": str(args.checkpoint),
        "status": "saved_robomimic_policy_closed_loop_replay",
        "startup_seed": cfg.seed,
        "episodes": results,
    }, indent=2) + "\n")
    if args.video:
        imageio.mimwrite(args.video, frames, fps=25, macro_block_size=16)
    env.close()


if __name__ == "__main__":
    main()
