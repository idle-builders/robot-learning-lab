"""Dynamically hold a side-reach geometry candidate with the selected XML servos.

This is a direct MuJoCo physics gate from an authored pose, not a learned
controller. The XML servos have rated force limits but lack the native
frame50 torque-speed and command-delay sensitivity model.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--body-xml", type=Path,
                        help="Override the geometry report body for a matched physics sensitivity case")
    parser.add_argument("--case", default="side_lean_repositioned")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--start-mode", choices=("pose", "stand"), default="pose")
    parser.add_argument("--stand-pose", choices=("neutral", "attained"),
                        default="neutral", help="Upright leg pose at the start of a transition")
    parser.add_argument("--transition-s", type=float, default=4.0)
    parser.add_argument("--motion-order", choices=("all", "squat_then_lean"),
                        default="all")
    parser.add_argument("--start-x-offset-m", type=float, default=0.0)
    parser.add_argument("--start-y-offset-m", type=float, default=0.0)
    parser.add_argument("--rail-v", type=float, choices=(10.0, 12.0), default=12.0)
    parser.add_argument("--command-delay-ms", type=float, default=0.0)
    parser.add_argument("--mass-scale", type=float, default=1.0)
    parser.add_argument("--sole-friction", type=float)
    parser.add_argument("--joint-target-offset", nargs=2, action="append",
                        metavar=("JOINT", "RAD"), default=[],
                        help="Small target change from the authored starting pose")
    parser.add_argument("--state-out", type=Path,
                        help="Save sampled native positions and motor targets for a teacher trace")
    parser.add_argument("--video", type=Path,
                        help="Render the authored replay with visible physics proxies")
    args = parser.parse_args()
    source = json.loads(args.geometry.read_text())
    candidate = next(row for row in source["results"] if row["case"] == args.case)
    body = args.body_xml or Path(source["selected_body_xml"])
    root = ET.parse(body).getroot()
    toy_pos = source["toy_center_m"]
    world = root.find("worldbody")
    toy = ET.SubElement(world, "body", name="probe_toy",
                        pos=" ".join(map(str, toy_pos)))
    ET.SubElement(toy, "freejoint", name="probe_toy_free")
    ET.SubElement(toy, "geom", name="probe_toy_geom", type="box",
                  size="0.01 0.01 0.01", mass="0.00625")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if args.mass_scale <= 0:
        parser.error("--mass-scale must be positive")
    if args.mass_scale != 1.0:
        for body_id in range(1, model.nbody):
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 body_id) != "probe_toy":
                model.body_mass[body_id] *= args.mass_scale
                model.body_inertia[body_id] *= args.mass_scale
    if args.sole_friction is not None:
        if args.sole_friction <= 0:
            parser.error("--sole-friction must be positive")
        for side in ("left", "right"):
            sole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                        f"{side}_sole")
            model.geom_friction[sole_id, 0] = args.sole_friction
    if args.rail_v != 12.0:
        for actuator_id in range(model.nu):
            driven_joint = int(model.actuator_trnid[actuator_id, 0])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, driven_joint)
            if name != "head_yaw" and not name.endswith("_gripper"):
                model.actuator_forcerange[actuator_id] *= args.rail_v / 12.0
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[:3] = candidate["pelvis_xyz_m"]
    roll = math.radians(candidate["torso_roll_deg"])
    data.qpos[3:7] = (math.cos(roll / 2), math.sin(roll / 2), 0, 0)

    def joint(name: str) -> int:
        result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if result < 0:
            raise ValueError(name)
        return result

    def put(name: str, angle: float) -> None:
        data.qpos[model.jnt_qposadr[joint(name)]] = angle

    for side, roll_target in zip(("left", "right"),
                                 candidate["leg_pose_rad"]["hip_roll_left_right"]):
        put(f"{side}_hip_roll", roll_target)
        for name, angle in zip(("hip_pitch", "knee", "ankle_pitch"),
                               candidate["leg_pose_rad"][side]):
            put(f"{side}_{name}", angle)
    put("left_shoulder_pitch", -1.0)
    for name, angle in candidate["arm_pose_rad"].items():
        put(name, angle)
    toy_joint = joint("probe_toy_free")
    data.qpos[model.jnt_qposadr[toy_joint]:model.jnt_qposadr[toy_joint] + 7] = (
        *toy_pos, 1, 0, 0, 0)
    motor_joints = [int(model.actuator_trnid[i, 0]) for i in range(model.nu)]
    target_ctrl = np.array([data.qpos[model.jnt_qposadr[i]] for i in motor_joints])
    target_offsets = {}
    allowed_offsets = set(candidate["arm_pose_rad"]) | {
        f"{side}_{axis}" for side in ("left", "right")
        for axis in ("hip_roll", "hip_pitch", "knee", "ankle_pitch")
    }
    for name, raw_offset in args.joint_target_offset:
        if name not in allowed_offsets:
            parser.error(f"Unsupported target offset joint: {name}")
        actuator_id = motor_joints.index(joint(name))
        offset = float(raw_offset)
        target = target_ctrl[actuator_id] + offset
        lower, upper = model.jnt_range[joint(name)]
        if not lower <= target <= upper:
            parser.error(f"Target {name}={target:.3f} is outside its joint range")
        target_ctrl[actuator_id] = target
        target_offsets[name] = offset
    if args.start_mode == "stand":
        data.qpos[:model.jnt_qposadr[toy_joint]] = model.qpos0[:model.jnt_qposadr[toy_joint]]
        data.qpos[:3] = (candidate["pelvis_xyz_m"][0] + args.start_x_offset_m,
                        candidate["pelvis_xyz_m"][1] + args.start_y_offset_m,
                        0.23)
        data.qpos[3:7] = (1, 0, 0, 0)
        if args.stand_pose == "attained":
            for side in ("left", "right"):
                for name, angle in (("hip_pitch", -0.198), ("knee", 0.409),
                                    ("ankle_pitch", -0.2)):
                    put(f"{side}_{name}", angle)
        mujoco.mj_forward(model, data)
        sole_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                      f"{side}_sole") for side in ("left", "right")]
        bottoms = []
        for sole_id in sole_ids:
            rotation = data.geom_xmat[sole_id].reshape(3, 3)
            bottoms.append(data.geom_xpos[sole_id, 2] -
                           np.dot(model.geom_size[sole_id], np.abs(rotation[2, :])))
        data.qpos[2] -= float(np.mean(bottoms))
    start_ctrl = np.array([data.qpos[model.jnt_qposadr[i]] for i in motor_joints])
    squat_ctrl = start_ctrl.copy()
    if args.motion_order == "squat_then_lean":
        for side in ("left", "right"):
            for name, angle in (("hip_pitch", -1.452), ("knee", 1.815),
                                ("ankle_pitch", -0.279)):
                squat_ctrl[motor_joints.index(joint(f"{side}_{name}"))] = angle
        squat_ctrl[motor_joints.index(joint("left_shoulder_pitch"))] = -1.0
    data.ctrl[:] = start_ctrl
    mujoco.mj_forward(model, data)
    initial_root = data.qpos[:3].copy()
    initial_toy = data.qpos[model.jnt_qposadr[toy_joint]:model.jnt_qposadr[toy_joint] + 3].copy()
    sampled = []
    qpos_samples = []
    ctrl_samples = []
    sample_times = []
    renderer = writer = None
    if args.video:
        import imageio.v2 as imageio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        for geom_id in range(model.ngeom):
            if (model.geom_bodyid[geom_id] != 0 and model.geom_group[geom_id] == 4
                    and model.geom_rgba[geom_id, 3] == 0):
                model.geom_rgba[geom_id, 3] = 1
                model.geom_group[geom_id] = 0
        model.geom_rgba[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "probe_toy_geom")] = (1, 0.45, 0.1, 1)
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (0.1, 0.0, 0.17)
        camera.distance = 0.65
        camera.azimuth = 135
        camera.elevation = -20
        writer = imageio.get_writer(args.video, fps=20, codec="libx264",
                                    quality=8, macro_block_size=16)
    max_leg_force = 0.0
    leg_rated_frames = 0
    leg_rated_by_joint = np.zeros(10, dtype=int)
    max_root_tilt_deg = 0.0
    toy_finger_frames = 0
    late_toy_finger_frames = 0
    late_frames = 0
    consecutive_toy_finger_frames = 0
    longest_toy_finger_run_frames = 0
    hand_floor_frames = 0
    foot_ground_frames = [0, 0]
    both_feet_ground_frames = 0
    nsteps = round(args.duration_s / model.opt.timestep)
    lag_steps = round(args.command_delay_ms / (1000 * model.opt.timestep))
    command_queue = deque((start_ctrl.copy() for _ in range(lag_steps + 1)),
                          maxlen=lag_steps + 1)
    stride = max(1, round(0.05 / model.opt.timestep))
    for step in range(nsteps):
        if args.start_mode == "stand":
            progress = min(1.0, (step * model.opt.timestep) / args.transition_s)
            if args.motion_order == "squat_then_lean":
                first = min(1.0, 2 * progress)
                second = max(0.0, 2 * progress - 1)
                first = first * first * (3 - 2 * first)
                second = second * second * (3 - 2 * second)
                command = (start_ctrl + first * (squat_ctrl - start_ctrl)
                           + second * (target_ctrl - squat_ctrl))
            else:
                blend = progress * progress * (3 - 2 * progress)
                command = start_ctrl + blend * (target_ctrl - start_ctrl)
        else:
            command = target_ctrl
        command_queue.append(command.copy())
        data.ctrl[:] = command_queue[0]
        mujoco.mj_step(model, data)
        contacts = set()
        for contact in data.contact:
            names = frozenset((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                                 contact.geom1),
                               mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                                 contact.geom2)))
            contacts.add(names)
        feet_grounded = [frozenset(("floor", f"{side}_sole")) in contacts
                         for side in ("left", "right")]
        for i, grounded in enumerate(feet_grounded):
            foot_ground_frames[i] += grounded
        both_feet_ground_frames += all(feet_grounded)
        finger_touch = frozenset(("probe_toy_geom", "right_finger_tip")) in contacts
        toy_finger_frames += finger_touch
        consecutive_toy_finger_frames = (consecutive_toy_finger_frames + 1
                                         if finger_touch else 0)
        longest_toy_finger_run_frames = max(longest_toy_finger_run_frames,
                                            consecutive_toy_finger_frames)
        if (step + 1) * model.opt.timestep >= 1.0:
            late_frames += 1
            late_toy_finger_frames += finger_touch
        hand_floor_frames += any("floor" in pair and any(
            name in pair for name in ("right_hand", "right_finger_tip"))
            for pair in contacts)
        forces = np.abs(data.actuator_force[:10])
        rated = np.array([model.actuator_forcerange[i, 1] for i in range(10)])
        max_leg_force = max(max_leg_force, float(forces.max()))
        leg_at_rated = forces >= 0.995 * rated
        leg_rated_frames += bool(np.any(leg_at_rated))
        leg_rated_by_joint += leg_at_rated
        max_root_tilt_deg = max(max_root_tilt_deg, math.degrees(
            2 * math.acos(min(1, abs(float(data.qpos[3]))))))
        if step % stride == 0 or step == nsteps - 1:
            q = data.qpos[model.jnt_qposadr[toy_joint]:model.jnt_qposadr[toy_joint] + 3]
            tilt = math.degrees(2 * math.acos(min(1, abs(float(data.qpos[3])))))
            sampled.append({
                "time_s": round((step + 1) * model.opt.timestep, 3),
                "root_xyz_m": data.qpos[:3].round(5).tolist(),
                "root_tilt_deg": round(tilt, 2),
                "toy_xyz_m": q.round(5).tolist(),
                "toy_finger_contact": finger_touch,
                "sole_ground_contact": feet_grounded,
                "hand_floor_contact": any("floor" in pair and any(
                    name in pair for name in ("right_hand", "right_finger_tip"))
                    for pair in contacts),
            })
            if args.state_out:
                qpos_samples.append(data.qpos.copy())
                ctrl_samples.append(data.ctrl.copy())
                sample_times.append((step + 1) * model.opt.timestep)
            if writer:
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
    if writer:
        writer.close()
        renderer.close()
    if args.state_out:
        args.state_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.state_out, time_s=np.asarray(sample_times),
                            qpos=np.asarray(qpos_samples), ctrl=np.asarray(ctrl_samples))
    final_toy = data.qpos[model.jnt_qposadr[toy_joint]:model.jnt_qposadr[toy_joint] + 3]
    report = {
        "schema": "robot-learning-lab.cove50-side-reach-hold.v1",
        "geometry_report": str(args.geometry),
        "geometry_sha256": hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
        "case": args.case,
        "start_mode": args.start_mode,
        "stand_pose": args.stand_pose if args.start_mode == "stand" else None,
        "transition_s": args.transition_s if args.start_mode == "stand" else None,
        "motion_order": args.motion_order if args.start_mode == "stand" else None,
        "start_xy_offset_m": ([args.start_x_offset_m, args.start_y_offset_m]
                              if args.start_mode == "stand" else None),
        "joint_target_offsets_rad": target_offsets,
        "selected_body_xml": str(body),
        "selected_body_sha256": hashlib.sha256(body.read_bytes()).hexdigest(),
        "duration_s": args.duration_s,
        "rail_v": args.rail_v,
        "command_delay_ms": args.command_delay_ms,
        "mass_scale": args.mass_scale,
        "sole_friction_override": args.sole_friction,
        "initial_root_xyz_m": initial_root.tolist(),
        "final_root_xyz_m": data.qpos[:3].tolist(),
        "root_travel_m": float(np.linalg.norm(data.qpos[:2] - initial_root[:2])),
        "initial_toy_xyz_m": initial_toy.tolist(),
        "final_toy_xyz_m": final_toy.tolist(),
        "toy_travel_m": float(np.linalg.norm(final_toy - initial_toy)),
        "fraction_frames_both_soles_grounded": both_feet_ground_frames / nsteps,
        "fraction_frames_finger_toy_contact": toy_finger_frames / nsteps,
        "fraction_frames_finger_toy_contact_after_1s": late_toy_finger_frames / max(1, late_frames),
        "longest_continuous_finger_toy_contact_s": longest_toy_finger_run_frames * model.opt.timestep,
        "fraction_frames_hand_floor_contact": hand_floor_frames / nsteps,
        "fraction_frames_leg_at_rated_force": leg_rated_frames / nsteps,
        "fraction_frames_at_rated_force_by_leg_joint": {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id):
                float(leg_rated_by_joint[i] / nsteps)
            for i, joint_id in enumerate(motor_joints[:10])
        },
        "max_leg_force_nm": max_leg_force,
        "max_root_tilt_deg": max_root_tilt_deg,
        "state_npz": str(args.state_out) if args.state_out else None,
        "video": str(args.video) if args.video else None,
        "trace": sampled,
        "qualification": "Authored direct MuJoCo XML position-servo probe. The force cap is scaled with rail voltage and position commands can be delayed; the native frame50 torque-speed law, learned policy, and any prior locomotion to the repositioned start are untested.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "trace"},
                     indent=2))


if __name__ == "__main__":
    main()
