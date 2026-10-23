"""Replay a screened single-foot reset through the selected MJLab motors.

This is a constant joint-target probe from a raised-foot reset. It is not a
stand-to-lift controller or learned policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

import ape_walk.task as ape
from ape_walk.physics.frame50 import (clip_frame50_leg_targets,
                                      configure_frame50_motors,
                                      reveal_frame50_proxy_geometry_for_video)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from experiments.compact_morphology.cove50_support_recovery.residual import configure_residual_actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        help="Evaluate a saved policy from this raised-foot reset")
    parser.add_argument("--preload-json", type=Path,
                        help="Static stance-wrench feedforward position targets")
    parser.add_argument("--residual-policy", action="store_true",
                        help="Checkpoint uses 0.2-rad corrections around preload targets")
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--stance", choices=("left", "right"), required=True)
    parser.add_argument("--seed", type=int, default=2144)
    parser.add_argument("--rail-v", type=float, default=12.0)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--tilt-noise-deg", type=float, default=0.0)
    parser.add_argument("--joint-noise-std", type=float, default=0.0)
    parser.add_argument("--z-noise-m", type=float, default=0.0)
    parser.add_argument("--com-feedback-x", type=float, default=0.0)
    parser.add_argument("--com-feedback-y", type=float, default=0.0)
    parser.add_argument("--target-com-y", type=float, default=0.018)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--audit-contacts", action="store_true",
                        help="Attribute self contacts by replaying sampled qpos in CPU MuJoCo")
    parser.add_argument("--freeze-swing", action="store_true",
                        help="Ablate learned swing-leg corrections around the preloaded pose")
    parser.add_argument("--swing-roll-target", type=float,
                        help="Override the swing hip-roll target to check thigh-torso clearance")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.pose_json.read_text())
    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    if model.nu != 21 or model.nq != 28:
        raise ValueError("Expected exact 21-axis Cove body")
    joint_names = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    pose = [spec["joints"][name] for name in joint_names]
    target_joints = (json.loads(args.preload_json.read_text())["joints"]
                     if args.preload_json else spec["joints"])
    target_pose = [target_joints[name] for name in joint_names]
    _, _, z, roll, pitch = spec["root_xyz_roll_pitch"]
    task_id = (ape.ELBOW_FLAMINGO_LEFT_TASK_ID if args.stance == "left"
               else ape.ELBOW_FLAMINGO_COMMAND_LIMIT_TASK_ID)
    cfg = load_env_cfg(task_id, play=True)
    cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
    if args.residual_policy:
        if not args.checkpoint or not args.preload_json:
            parser.error("--residual-policy requires --checkpoint and --preload-json")
        configure_residual_actions(cfg, target_joints)
    cfg.rewards["pose_flamingo"].params["target_overrides"] = dict(enumerate(pose))
    cfg.rewards["gravity_flamingo"].params["target"] = (
        math.sin(pitch), -math.sin(roll) * math.cos(pitch),
        -math.cos(roll) * math.cos(pitch),
    )
    cfg.rewards["swing_foot_clear"].params["target"] = 0.10
    cfg.events["set_flamingo_state"].params.update({
        "joint_pose": pose, "base_roll": roll, "base_pitch": pitch,
        "z_min": z, "z_max": z + args.z_noise_m,
        "tilt_noise": math.radians(args.tilt_noise_deg),
        "joint_noise_std": args.joint_noise_std, "standing_prob": 0.0,
    })
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, z)
    configure_frame50_motors(cfg, args.body_xml,
                             rail_v=args.rail_v, delay_s=args.delay_ms / 1000.0)
    clip_frame50_leg_targets(cfg, args.body_xml, margin_rad=0.02)
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ape.ContactSensorCfg(
            name="self_body_contact",
            primary=ape.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=ape.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"), reduce="none", num_slots=1,
        ),
        ape.ContactSensorCfg(
            name="nonsole_ground_contact",
            primary=ape.ContactMatch(
                mode="geom", pattern=r"^(?!(?:left_sole|right_sole)$).*", entity="robot"
            ),
            secondary=ape.ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
        ),
    )
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.auto_reset = False
    cfg.episode_length_s = args.duration + 0.5
    if args.video:
        cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
        cfg.viewer.lookat = (0.0, 0.0, 0.24)
        cfg.viewer.distance = 0.75
        cfg.viewer.azimuth = 45.0
        cfg.viewer.elevation = -14.0
        cfg.viewer.width = 640
        cfg.viewer.height = 480
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device="cuda:0",
                              render_mode="rgb_array" if args.video else None),
        clip_actions=100.0,
    )
    robot = env.unwrapped.scene["robot"]
    if args.video:
        reveal_frame50_proxy_geometry_for_video(env.unwrapped.sim.mj_model)
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    target = torch.tensor(target_pose, device="cuda:0", dtype=robot.data.joint_pos.dtype)[None, :]
    base_action = torch.where(action_term.scale.abs() > 1e-8,
                              (target - action_term.offset) / action_term.scale,
                              torch.zeros_like(target))
    if args.checkpoint:
        runner = load_runner_cls(task_id)(env, asdict(load_rl_cfg(task_id)), device="cuda:0")
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
    hip_roll_index = joint_names.index(f"{args.stance}_hip_roll")
    hip_pitch_index = joint_names.index(f"{args.stance}_hip_pitch")
    swing_name = "left" if args.stance == "right" else "right"
    swing_joint_indices = [i for i, name in enumerate(joint_names)
                           if name.startswith(f"{swing_name}_") and
                           any(name.endswith(axis) for axis in
                               ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch"))]
    swing_roll_index = joint_names.index(f"{swing_name}_hip_roll")
    stance_index = 0 if args.stance == "left" else 1
    swing_index = 1 - stance_index
    site_ids = [robot.site_names.index(f"{side}_sole_center") for side in ("left", "right")]
    start_stance = robot.data.site_pos_w[0, site_ids[stance_index], :2].cpu().numpy().copy()
    rated = torch.tensor([a.effort_limit for a in
                          cfg.scene.entities["robot"].articulation.actuators[:10]],
                         device="cuda:0")
    initial_hash = hashlib.sha256()
    for tensor in (env.unwrapped.sim.data.qpos, env.unwrapped.sim.data.qvel):
        initial_hash.update(tensor.cpu().numpy().tobytes())
    rows, frames = [], []
    audit_data = mujoco.MjData(model) if args.audit_contacts else None
    audited_pairs = Counter()
    audited_contact_samples = []
    first_action_hash = None
    dt = env.unwrapped.step_dt
    sensors = env.unwrapped.scene.sensors
    for step in range(round(args.duration / dt)):
        with torch.inference_mode():
            action = policy(env.get_observations()) if args.checkpoint else base_action.clone()
        if args.freeze_swing or args.swing_roll_target is not None:
            action = action.clone()
            if args.freeze_swing:
                action[:, swing_joint_indices] = 0.0
            if args.swing_roll_target is not None:
                action[:, swing_roll_index] = (
                    args.swing_roll_target - action_term.offset[0, swing_roll_index]
                ) / action_term.scale[0, swing_roll_index]
        if not args.checkpoint and (args.com_feedback_x or args.com_feedback_y):
            quat_now = robot.data.root_link_pose_w[0, 3:7]
            yaw_now = torch.atan2(
                2 * (quat_now[0] * quat_now[3] + quat_now[1] * quat_now[2]),
                1 - 2 * (quat_now[2] ** 2 + quat_now[3] ** 2),
            )
            com_now = env.unwrapped.sim.data.subtree_com[0, robot.indexing.root_body_id, :2]
            sole_now = robot.data.site_pos_w[0, site_ids[stance_index], :2]
            delta = com_now - sole_now
            local_x = torch.cos(yaw_now) * delta[0] + torch.sin(yaw_now) * delta[1]
            local_y = -torch.sin(yaw_now) * delta[0] + torch.cos(yaw_now) * delta[1]
            y_target = args.target_com_y * (1 if args.stance == "right" else -1)
            roll_shift = args.com_feedback_y * (local_y - y_target)
            pitch_shift = args.com_feedback_x * (0.0 - local_x)
            action[:, hip_roll_index] += roll_shift / action_term.scale[0, hip_roll_index]
            action[:, hip_pitch_index] += pitch_shift / action_term.scale[0, hip_pitch_index]
        if first_action_hash is None:
            first_action_hash = hashlib.sha256(action.cpu().numpy().tobytes()).hexdigest()
        with torch.inference_mode():
            _, _, done, _ = env.step(action)
        soles = robot.data.site_pos_w[0, site_ids].cpu().numpy()
        contacts = sensors["feet_ground_contact"].data.found[0].cpu().tolist()
        effort = robot.data.actuator_force[0, :10].abs()
        quat = robot.data.root_link_pose_w[0, 3:7].cpu().numpy()
        tilt = math.degrees(math.acos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1)))
        com = env.unwrapped.sim.data.subtree_com[0, robot.indexing.root_body_id, :2].cpu().numpy()
        rows.append({
            "time_s": round((step + 1) * dt, 3),
            "pelvis_z_m": round(float(robot.data.root_link_pos_w[0, 2]), 4),
            "tilt_deg": round(tilt, 2),
            "feet_grounded": [bool(x) for x in contacts],
            "swing_z_m": round(float(soles[swing_index, 2]), 4),
            "stance_slip_m": round(float(np.linalg.norm(soles[stance_index, :2] - start_stance)), 4),
            "com_minus_stance_xy_m": np.round(com - soles[stance_index, :2], 4).tolist(),
            "body_contact": bool(sensors["self_body_contact"].data.found[0].any()),
            "nonsole_ground_contact": bool(sensors["nonsole_ground_contact"].data.found[0].any()),
            "hand_force_n": round(float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).max()), 3),
            "max_leg_torque_nm": round(float(effort.max()), 3),
            "rated_leg": bool((effort >= .995 * rated).any()),
        })
        if args.video and step % 2 == 0:
            frames.append(env.unwrapped.render())
        if audit_data is not None:
            audit_data.qpos[:] = env.unwrapped.sim.data.qpos[0].cpu().numpy()
            audit_data.qvel[:] = env.unwrapped.sim.data.qvel[0].cpu().numpy()
            mujoco.mj_forward(model, audit_data)
            for contact in audit_data.contact[:audit_data.ncon]:
                pair = tuple(sorted((model.geom(contact.geom1).name,
                                     model.geom(contact.geom2).name)))
                if "floor" not in pair:
                    audited_pairs[pair] += 1
                    if len(audited_contact_samples) < 6 and (
                            len(audited_contact_samples) < 3 or step % 50 == 0):
                        audited_contact_samples.append({
                            "time_s": round((step + 1) * dt, 3),
                            "pair": list(pair), "penetration_m": round(max(0.0, -contact.dist), 5),
                            "joints": {name: round(float(audit_data.qpos[model.joint(name).qposadr[0]]), 4)
                                       for name in ("left_hip_roll", "left_hip_pitch", "left_knee",
                                                    "right_hip_roll", "right_hip_pitch", "right_knee")},
                        })
        if bool(done[0]):
            break
    after = [r for r in rows if r["time_s"] >= .5]
    support_pair = [True, False] if args.stance == "left" else [False, True]
    longest = current = 0
    for row in rows:
        current = current + 1 if row["feet_grounded"] == support_pair and row["swing_z_m"] > .025 else 0
        longest = max(longest, current)
    report = {
        "schema": "robot-learning-lab.cove50-native-support-pose.v1",
        "method": (
            "Saved actor from screened raised-foot reset through native MJLab DC motors; no stand-to-lift transition"
            if args.checkpoint else
            "COM-corrected joint targets from screened raised-foot reset through native MJLab DC motors; no stand-to-lift transition"
            if args.com_feedback_x or args.com_feedback_y else
            "Constant joint target from screened raised-foot reset through native MJLab DC motors; no stand-to-lift transition"
        ),
        "pose_json": str(args.pose_json),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
        "preload_json": str(args.preload_json) if args.preload_json else None,
        "residual_policy": args.residual_policy,
        "freeze_swing": args.freeze_swing,
        "swing_roll_target_rad": args.swing_roll_target,
        "body_xml": str(args.body_xml),
        "body_sha256": hashlib.sha256(args.body_xml.read_bytes()).hexdigest(),
        "stance": args.stance, "seed": args.seed,
        "rail_v": args.rail_v, "delay_ms": args.delay_ms,
        "tilt_noise_deg": args.tilt_noise_deg,
        "joint_noise_std": args.joint_noise_std,
        "z_noise_m": args.z_noise_m,
        "initial_state_sha256": initial_hash.hexdigest(),
        "first_action_sha256": first_action_hash,
        "com_feedback_x": args.com_feedback_x,
        "com_feedback_y": args.com_feedback_y,
        "target_com_y": args.target_com_y,
        "duration_s": rows[-1]["time_s"],
        "longest_single_support_s": round(longest * dt, 3),
        "min_swing_after_half_s_m": min((r["swing_z_m"] for r in after), default=None),
        "max_stance_slip_m": max(r["stance_slip_m"] for r in rows),
        "rated_leg_fraction": round(sum(r["rated_leg"] for r in rows) / len(rows), 4),
        "body_contact_frames": sum(r["body_contact"] for r in rows),
        "nonsole_ground_contact_frames": sum(r["nonsole_ground_contact"] for r in rows),
        "cpu_audited_self_contact_pairs": [
            {"pair": list(pair), "sampled_frames": count}
            for pair, count in audited_pairs.most_common()
        ] if args.audit_contacts else None,
        "cpu_audited_self_contact_samples": audited_contact_samples if args.audit_contacts else None,
        "max_hand_force_n": max(r["hand_force_n"] for r in rows),
        "last": rows[-1], "timeseries": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "timeseries"}, indent=2))
    if args.video:
        import imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=25, macro_block_size=16)
        print("video:", args.video)
    env.close()


if __name__ == "__main__":
    main()
