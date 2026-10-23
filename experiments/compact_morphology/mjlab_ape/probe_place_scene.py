"""Authored physics check for placing a previously held toy on the floor.

This verifies the reset and handoff for a future learned place lesson. It is
never evidence of a learned release or place policy.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import ape_walk.task
from ape_walk.grasp import HELD_TASK_ID
from ape_walk.place_tracking import BC_TASK_ID
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--motion-out", type=Path)
    parser.add_argument("--task-id", choices=(HELD_TASK_ID, BC_TASK_ID), default=HELD_TASK_ID)
    args = parser.parse_args()
    cfg = load_env_cfg(args.task_id, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = 142
    cfg.auto_reset = False
    pose = cfg.scene.entities["robot"].init_state.joint_pos
    pose[r"^right_shoulder_pitch$"] = 0.103
    pose[r"^right_elbow_pitch$"] = -1.754
    pose[r"^right_gripper$"] = 0.35
    cfg.actions["joint_pos"].scale[r"^right_gripper$"] = 0.5
    cmd = cfg.commands["lift_height"]
    cmd.object_pose_range.x = (0.088, 0.088)
    cmd.object_pose_range.y = (-0.0845, -0.0845)
    cmd.object_pose_range.z = (0.066, 0.066)
    cmd.target_position_range.x = (0.08, 0.08)
    cmd.target_position_range.y = (-0.0845, -0.0845)
    cmd.target_position_range.z = (0.01, 0.01)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    robot = env.scene["robot"]
    cube = env.scene["cube"]
    action = env.action_manager.get_term("joint_pos")
    indexes = {n: action._target_names.index("right_" + n) for n in
               ("shoulder_pitch", "elbow_pitch", "gripper")}
    rows = []
    motion = {name: [] for name in (
        "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
        "body_lin_vel_w", "body_ang_vel_w",
    )}
    for step in range(round(3.0 / env.step_dt)):
        t = (step + 1) * env.step_dt
        alpha = max(0.0, min(1.0, (t - 0.4) / 1.1))
        withdraw = max(0.0, min(1.0, (t - 1.8) / 0.7))
        target = {
            "shoulder_pitch": 0.103 + alpha * (-0.035 - 0.103) + withdraw * (0.103 + 0.035),
            "elbow_pitch": -1.754 + alpha * (-0.986 + 1.754) + withdraw * (-1.754 + 0.986),
            "gripper": 0.35 * (1 - max(0.0, min(1.0, (t - 1.5) / 0.3))),
        }
        a = torch.zeros((1, action.action_dim), device="cuda:0")
        for name, value in target.items():
            i = indexes[name]
            a[0, i] = (value - action._offset[0, i]) / action._scale[0, i]
        env.step(a)
        if args.motion_out:
            for name in motion:
                motion[name].append(getattr(robot.data, "body_link_" + name[5:] if name.startswith("body_") else name)[0].cpu().numpy().copy())
        rows.append({
            "time_s": round(t, 3),
            "cube_xyz_m": [round(float(v), 4) for v in cube.data.root_link_pos_w[0].tolist()],
            "hand_xyz_m": [round(float(v), 4) for v in robot.data.site_pos_w[0, robot.site_names.index("right_hand_center")].tolist()],
            "fixed_jaw_contact": bool(env.scene.sensors["right_hand_cube_contact"].data.found[0].any()),
            "moving_jaw_contact": bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any()),
            "gripper_rad": round(float(robot.data.joint_pos[0, robot.joint_names.index("right_gripper")]), 4),
        })
    out = {"status": "scripted_place_physics_probe_not_learned_policy", "task": args.task_id,
           "initial": rows[0], "final": rows[-1],
           "samples": [rows[min(round(t / env.step_dt)-1, len(rows)-1)]
                       for t in (0.4, 1.0, 1.5, 1.8, 2.4, 3.0)],
           "timeseries": rows}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    if args.motion_out:
        np.savez(args.motion_out, fps=np.array([round(1 / env.step_dt)]),
                 **{name: np.asarray(values) for name, values in motion.items()})
    print(json.dumps({k: v for k, v in out.items() if k != "timeseries"}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
