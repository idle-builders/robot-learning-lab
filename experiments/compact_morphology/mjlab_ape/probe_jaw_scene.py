"""Check the free-cube lift physically in the exact MJLab training scene.

This is a scripted feasibility probe, never a learned-policy evaluation.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

import ape_walk.task
from ape_walk.grasp import ACQUIRE_COLLISION_TASK_ID, HELD_COLLISION_TASK_ID, HELD_TASK_ID
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--motion-out", type=Path)
    parser.add_argument("--task-id", choices=(HELD_TASK_ID, HELD_COLLISION_TASK_ID, ACQUIRE_COLLISION_TASK_ID), default=HELD_TASK_ID)
    args = parser.parse_args()
    cfg = load_env_cfg(args.task_id, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = 142
    cfg.auto_reset = False
    cfg.commands["lift_height"].resampling_time_range = (100.0, 100.0)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    robot = env.scene["robot"]
    cube = env.scene["cube"]
    action = env.action_manager.get_term("joint_pos")
    shoulder = action._target_names.index("right_shoulder_pitch")
    elbow = action._target_names.index("right_elbow_pitch")
    gripper = action._target_names.index("right_gripper")
    gripper_joint = robot.joint_names.index("right_gripper")
    shoulder_joint = robot.joint_names.index("right_shoulder_pitch")
    elbow_joint = robot.joint_names.index("right_elbow_pitch")
    hand_site = robot.site_names.index("right_hand_center")
    fixed_jaw_geom = robot.geom_names.index("right_hand")
    moving_jaw_geom = robot.geom_names.index("right_finger_tip")
    model = env.sim.mj_model
    raw_geoms = {
        suffix: next(i for i in range(model.ngeom)
                     if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").endswith("/" + suffix))
        for suffix in ("right_hand", "right_finger_tip", "cube_geom")
    }
    raw_hand_site = next(i for i in range(model.nsite)
                         if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or "").endswith("/right_hand_center"))
    rows = []
    motion = {name: [] for name in (
        "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
        "body_lin_vel_w", "body_ang_vel_w",
    )}
    for step in range(round(3.0 / env.step_dt)):
        t = (step + 1) * env.step_dt
        acquiring = args.task_id == ACQUIRE_COLLISION_TASK_ID
        alpha = max(0.0, min(1.0, (t - (1.0 if acquiring else 0.4)) / 1.1))
        desired_gripper = 0.35 * max(0.0, min(1.0, (t - 0.1) / 0.7)) if acquiring else 0.35
        desired_shoulder = -0.035 + alpha * (0.102 + 0.035)
        desired_elbow = -0.986 + alpha * (-1.756 + 0.986)
        actions = torch.zeros((1, action.action_dim), device="cuda:0")
        actions[0, shoulder] = (desired_shoulder - action._offset[0, shoulder]) / action._scale[0, shoulder]
        actions[0, elbow] = (desired_elbow - action._offset[0, elbow]) / action._scale[0, elbow]
        if acquiring:
            actions[0, gripper] = (desired_gripper - action._offset[0, gripper]) / action._scale[0, gripper]
        env.step(actions)
        if args.motion_out:
            for name in motion:
                motion[name].append(getattr(robot.data, "body_link_" + name[5:] if name.startswith("body_") else name)[0].cpu().numpy().copy())
        rows.append({
            "time_s": round(t, 3),
            "cube_z_m": round(float(cube.data.root_link_pos_w[0, 2]), 4),
            "pelvis_z_m": round(float(robot.data.root_link_pos_w[0, 2]), 4),
            "pelvis_xyz_m": [round(float(v), 4) for v in robot.data.root_link_pos_w[0].tolist()],
            "shoulder_target_rad": round(desired_shoulder, 4),
            "elbow_target_rad": round(desired_elbow, 4),
            "gripper_target_rad": round(desired_gripper, 4),
            "shoulder_actual_rad": round(float(robot.data.joint_pos[0, shoulder_joint]), 4),
            "elbow_actual_rad": round(float(robot.data.joint_pos[0, elbow_joint]), 4),
            "gripper_actual_rad": round(float(robot.data.joint_pos[0, gripper_joint]), 4),
            "cube_xyz_m": [round(float(v), 4) for v in cube.data.root_link_pos_w[0].tolist()],
            "fixed_jaw_xyz_m": [round(float(v), 4) for v in robot.data.geom_pos_w[0, fixed_jaw_geom].tolist()],
            "moving_jaw_xyz_m": [round(float(v), 4) for v in robot.data.geom_pos_w[0, moving_jaw_geom].tolist()],
            "hand_site_xyz_m": [round(float(v), 4) for v in robot.data.site_pos_w[0, hand_site].tolist()],
            "raw_fixed_jaw_xyz_m": [round(float(v), 4) for v in env.sim.data.geom_xpos[0, raw_geoms["right_hand"]].tolist()],
            "raw_moving_jaw_xyz_m": [round(float(v), 4) for v in env.sim.data.geom_xpos[0, raw_geoms["right_finger_tip"]].tolist()],
            "raw_hand_site_xyz_m": [round(float(v), 4) for v in env.sim.data.site_xpos[0, raw_hand_site].tolist()],
            "fixed_jaw_cube_contact": bool(env.scene.sensors["right_hand_cube_contact"].data.found[0].any()),
            "moving_jaw_cube_contact": bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any()),
        })
    out = {
        "status": "scripted_physics_feasibility_not_learned_policy",
        "task": args.task_id,
        "initial_cube_z_m": rows[0]["cube_z_m"],
        "max_cube_z_m": max(row["cube_z_m"] for row in rows),
        "final_cube_z_m": rows[-1]["cube_z_m"],
        "samples": [rows[min(round(t / env.step_dt) - 1, len(rows) - 1)] for t in (0.4, 0.7, 1.0, 1.5, 2.4, 3.0)],
        "timeseries": rows,
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    if args.motion_out:
        np.savez(args.motion_out, fps=np.array([round(1 / env.step_dt)]),
                 **{name: np.asarray(values) for name, values in motion.items()})
    print(json.dumps({k: v for k, v in out.items() if k != "timeseries"}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
