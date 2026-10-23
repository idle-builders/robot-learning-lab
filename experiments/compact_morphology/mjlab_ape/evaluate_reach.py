"""Fresh closed-loop replay of the ape's native MJLab floor-toy lesson."""

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import mujoco

import ape_walk.task
from ape_walk.reach import TASK_ID
from ape_walk.grasp import TASK_ID as JAW_TASK_ID, PREGRASP_TASK_ID, HELD_TASK_ID, HELD_COLLISION_TASK_ID
from ape_walk.grasp_tracking import TASK_ID as MOTION_TASK_ID
from ape_walk.place_tracking import TASK_ID as PLACE_TASK_ID
from ape_walk.place_stages import LOWER_TASK_ID, RELEASE_TASK_ID
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--cove50-body-xml", type=Path)
    parser.add_argument("--task", choices=(TASK_ID, JAW_TASK_ID, PREGRASP_TASK_ID, HELD_TASK_ID, HELD_COLLISION_TASK_ID, MOTION_TASK_ID, PLACE_TASK_ID, LOWER_TASK_ID, RELEASE_TASK_ID), default=TASK_ID)
    args = parser.parse_args()
    cfg = load_env_cfg(args.task, play=True)
    if args.cove50_body_xml:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from ape_walk.physics.frame50 import configure_frame50_motors

        body = args.cove50_body_xml.resolve()
        if mujoco.MjModel.from_xml_path(str(body)).nu != 21:
            raise ValueError("Expected the selected 21-axis Cove body")
        cfg.scene.entities["robot"].spec_fn = partial(ape_walk.task._ape_spec, body)
        cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
        cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, 0.1794)
        pose = cfg.scene.entities["robot"].init_state.joint_pos
        pose[r".*hip_pitch.*"] = -1.16
        pose[r".*knee.*"] = 1.46
        pose[r".*ankle_pitch.*"] = -0.30
        shoulder_pose = {}
        for pattern, value in pose.items():
            if pattern == r".*shoulder_pitch.*":
                shoulder_pose[r"^left_shoulder_pitch$"] = -1.0
                shoulder_pose[r"^right_shoulder_pitch$"] = value
            else:
                shoulder_pose[pattern] = value
        pose.clear()
        pose.update(shoulder_pose)
        cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
        configure_frame50_motors(cfg, body, rail_v=12.0, delay_s=0.0)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.auto_reset = False
    cfg.commands["lift_height"].resampling_time_range = (100.0, 100.0)
    if args.video:
        cfg.viewer.width = 640
        cfg.viewer.height = 480
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if args.video else None),
        clip_actions=load_rl_cfg(args.task).clip_actions,
    )
    robot = env.unwrapped.scene["robot"]
    cube = env.unwrapped.scene["cube"]
    if args.checkpoint:
        runner = load_runner_cls(args.task)(env, asdict(load_rl_cfg(args.task)), device="cuda:0")
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
    else:
        policy = lambda _: torch.zeros_like(robot.data.joint_pos)
    command = env.unwrapped.command_manager.get_term("lift_height")
    motion = (env.unwrapped.command_manager.get_term("motion")
              if "motion" in cfg.commands else None)
    hand_id = robot.site_names.index("right_hand_center")
    other_hand_id = robot.site_names.index("left_hand_center")
    gripper_id = robot.joint_names.index("right_gripper")
    shoulder_id = robot.joint_names.index("right_shoulder_pitch")
    elbow_id = robot.joint_names.index("right_elbow_pitch")
    sensors = env.unwrapped.scene.sensors
    jaw_sensor_names = ("right_hand_cube_contact", "right_finger_tip_cube_contact")
    has_jaw_sensors = all(name in sensors for name in jaw_sensor_names)
    dt = env.unwrapped.step_dt
    frames, rows = [], []
    for i in range(round(args.duration / dt)):
        with torch.inference_mode():
            actions = policy(env.get_observations())
            _, reward, done, _ = env.step(actions)
        if args.video and i % 2 == 0:
            frames.append(env.unwrapped.render())
        quat = robot.data.root_link_pose_w[0, 3:7].tolist()
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (quat[1] ** 2 + quat[2] ** 2)))))
        hand = robot.data.site_pos_w[0, hand_id]
        other_hand = robot.data.site_pos_w[0, other_hand_id]
        object_pos = cube.data.root_link_pos_w[0]
        leg_effort = robot.data.actuator_force[0, :10].abs()
        jaw_contacts = [bool(sensors[name].data.found[0].any()) for name in jaw_sensor_names] if has_jaw_sensors else [False, False]
        both_jaws_contact = all(jaw_contacts)
        rows.append({
            "time_s": round((i + 1) * dt, 3),
            "pelvis_z_m": round(float(robot.data.root_link_pos_w[0, 2]), 4),
            "pelvis_xyz_m": [round(float(v), 4) for v in robot.data.root_link_pos_w[0].tolist()],
            "tilt_deg": round(tilt, 2),
            "hand_to_cube_m": round(float(torch.linalg.vector_norm(hand - object_pos)), 4),
            "right_gripper_rad": round(float(robot.data.joint_pos[0, gripper_id]), 4),
            "right_shoulder_pitch_rad": round(float(robot.data.joint_pos[0, shoulder_id]), 4),
            "right_elbow_pitch_rad": round(float(robot.data.joint_pos[0, elbow_id]), 4),
            "hand_ground_force_n": round(float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).max()), 3),
            "both_jaws_cube_contact": both_jaws_contact,
            "fixed_jaw_cube_contact": jaw_contacts[0],
            "moving_jaw_cube_contact": jaw_contacts[1],
            "feet_grounded": [bool(v) for v in sensors["feet_ground_contact"].data.found[0].tolist()],
            "other_hand_xyz_m": [round(float(v), 4) for v in other_hand.tolist()],
            "max_leg_torque_nm": round(float(leg_effort.max()), 3),
            "cube_z_m": round(float(object_pos[2]), 4),
            "cube_xyz_m": [round(float(v), 4) for v in object_pos.tolist()],
            "hand_site_xyz_m": [round(float(v), 4) for v in hand.tolist()],
            "target_z_m": round(float(command.target_pos[0, 2]), 4),
            "target_xyz_m": [round(float(v), 4) for v in command.target_pos[0].tolist()],
            "motion_tracking": ({
                name: {
                    "body_xyz_m": [round(float(v), 4) for v in robot.data.body_link_pos_w[0, robot.body_names.index(name)].tolist()],
                    "target_xyz_m": [round(float(v), 4) for v in motion.body_pos_w[0, motion.cfg.body_names.index(name)].tolist()],
                }
                for name in ("right_forearm", "right_finger")
            } if motion else None),
            "reward": round(float(reward[0]), 5),
        })
        if bool(done[0]):
            break
    lifted_contact_streak = longest_lifted_contact_streak = 0
    released_place_streak = longest_released_place_streak = 0
    for row in rows:
        if row["cube_z_m"] >= 0.04 and row["both_jaws_cube_contact"]:
            lifted_contact_streak += 1
            longest_lifted_contact_streak = max(longest_lifted_contact_streak, lifted_contact_streak)
        else:
            lifted_contact_streak = 0
        place_error_xy = math.dist(row["cube_xyz_m"][:2], row["target_xyz_m"][:2])
        if (0.007 <= row["cube_z_m"] <= 0.025 and place_error_xy <= 0.02
                and not row["fixed_jaw_cube_contact"] and not row["moving_jaw_cube_contact"]
                and row["hand_site_xyz_m"][2] >= 0.03):
            released_place_streak += 1
            longest_released_place_streak = max(longest_released_place_streak, released_place_streak)
        else:
            released_place_streak = 0
    result = {
        "task": args.task,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "cove50_body_xml": str(args.cove50_body_xml) if args.cove50_body_xml else None,
        "cove50_body_sha256": (hashlib.sha256(args.cove50_body_xml.read_bytes()).hexdigest()
                               if args.cove50_body_xml else None),
        "seed": args.seed,
        "duration_s": rows[-1]["time_s"],
        "initial": rows[0],
        "final": rows[-1],
        "closest_hand_to_cube_m": min(row["hand_to_cube_m"] for row in rows),
        "max_cube_z_m": max(row["cube_z_m"] for row in rows),
        "longest_held_lift_s": round(longest_lifted_contact_streak * dt, 3),
        "held_lift_1s": longest_lifted_contact_streak * dt >= 1.0,
        "hand_floor_contact_frames": sum(row["hand_ground_force_n"] >= 1.0 for row in rows),
        "both_feet_grounded_fraction": round(sum(all(row["feet_grounded"]) for row in rows) / len(rows), 3),
        "max_leg_torque_nm": max(row["max_leg_torque_nm"] for row in rows),
        "success": bool(command.episode_success[0]),
        "timeseries": rows,
    }
    model = env.unwrapped.sim.mj_model
    result["native_collision_model"] = {
        "excludes": int(model.nexclude),
        "left_arm": next(
            {"contype": int(model.geom_contype[i]), "conaffinity": int(model.geom_conaffinity[i])}
            for i in range(model.ngeom)
            if model.body(int(model.geom_bodyid[i])).name.endswith(("/left_arm", "/left_upper_arm"))
            and model.geom_contype[i]
        ),
    }
    if args.task in (PLACE_TASK_ID, RELEASE_TASK_ID):
        result["longest_released_place_s"] = round(longest_released_place_streak * dt, 3)
        result["released_place_0p5s"] = longest_released_place_streak * dt >= 0.5
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if args.video:
        import imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=25, macro_block_size=16)
    print(json.dumps({key: value for key, value in result.items() if key != "timeseries"}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
