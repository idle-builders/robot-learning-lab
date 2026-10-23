"""Replay an upright → right single-foot → upright actor on selected motors."""

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
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from ape_walk import task as ape
from ape_walk.physics.frame50 import (clip_frame50_leg_targets,
                                      configure_frame50_motors,
                                      reveal_frame50_proxy_geometry_for_video)
from experiments.compact_morphology.cove50_support_recovery.phase_actions import PhasePositionActionCfg
from experiments.compact_morphology.cove50_support_recovery.phase_rewards import configure_phase_rewards, endpoint_com_offset
from experiments.compact_morphology.cove50_support_recovery.residual import configure_residual_actions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pose-json", type=Path, required=True)
    p.add_argument("--stance", choices=("left", "right"), default="right")
    p.add_argument("--preload-json", type=Path, required=True)
    p.add_argument("--body-xml", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rail-v", type=float, default=12.0)
    p.add_argument("--delay-ms", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=2144)
    p.add_argument("--freeze-swing", action="store_true")
    p.add_argument("--phase-rewards", action="store_true")
    p.add_argument("--lift-s", type=float, default=2.0)
    p.add_argument("--hold-s", type=float, default=3.0)
    p.add_argument("--lower-s", type=float, default=2.0)
    p.add_argument("--arm-clearance-bump-rad", type=float, default=0.0)
    p.add_argument("--staged", action="store_true")
    p.add_argument("--urgent-support", action="store_true")
    p.add_argument("--video", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    spec = json.loads(args.pose_json.read_text())
    preload = json.loads(args.preload_json.read_text())["joints"]
    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    names = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    swing = "left" if args.stance == "right" else "right"
    stance_slot = 1 if args.stance == "right" else 0
    swing_slot = 1 - stance_slot
    pose = [spec["joints"][name] for name in names]
    _, _, _, roll, pitch = spec["root_xyz_roll_pitch"]
    source = (ape.ELBOW_FLAMINGO_COMMAND_LIMIT_TASK_ID if args.stance == "right"
              else ape.ELBOW_FLAMINGO_LEFT_TASK_ID)
    cfg = load_env_cfg(source, play=True)
    configure_frame50_motors(cfg, args.body_xml,
                             rail_v=args.rail_v, delay_s=args.delay_ms / 1000.0)
    cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
    clip_frame50_leg_targets(cfg, args.body_xml, margin_rad=0.02)
    configure_residual_actions(cfg, preload)
    old = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = PhasePositionActionCfg(
        entity_name=old.entity_name, actuator_names=old.actuator_names,
        scale=old.scale, offset=old.offset, clip=old.clip,
        preserve_order=old.preserve_order, use_default_offset=False,
        target_joints=tuple(preload[name] for name in names),
        lift_s=args.lift_s, hold_s=args.hold_s, lower_s=args.lower_s,
        freeze_swing_side=swing if args.freeze_swing else None,
        arm_clearance_bump_rad=args.arm_clearance_bump_rad,
        staged=args.staged,
        stance_side=args.stance,
    )
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, float(model.qpos0[2]))
    target_duration_s = args.lift_s + args.hold_s + args.lower_s + 1.0
    cfg.episode_length_s = target_duration_s + 0.5
    cfg.events.pop("set_flamingo_state")
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    cfg.events["reset_robot_joints"].params.update({
        "position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
    })
    cfg.rewards["pose_flamingo"].params["target_overrides"] = dict(enumerate(pose))
    cfg.rewards["gravity_flamingo"].params["target"] = (
        math.sin(pitch), -math.sin(roll) * math.cos(pitch),
        -math.cos(roll) * math.cos(pitch),
    )
    cfg.rewards["swing_foot_clear"].params["target"] = 0.07
    cfg.rewards["joint_torques_l2"].weight = -0.05
    if args.phase_rewards:
        configure_phase_rewards(
            cfg, pose,
            (math.sin(pitch), -math.sin(roll) * math.cos(pitch),
             -math.cos(roll) * math.cos(pitch)),
            lift_s=args.lift_s, hold_s=args.hold_s, lower_s=args.lower_s,
            arm_index=names.index(f"{swing}_shoulder_roll"),
            arm_bump=args.arm_clearance_bump_rad,
            swing_indices=[i for i, name in enumerate(names) if
                           name.startswith(f"{swing}_") and any(name.endswith(axis)
                           for axis in ("hip_yaw", "hip_roll", "hip_pitch",
                                        "knee", "ankle_pitch"))],
            staged=args.staged,
            com_offset_xy=endpoint_com_offset(model, spec, args.stance),
            stance=args.stance, urgent_support=args.urgent_support)
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ape.ContactSensorCfg(
            name="thigh_torso_contact",
            primary=ape.ContactMatch(
                mode="geom", pattern=r"^(left_thigh_frame_budget|right_thigh_frame_budget)$",
                entity="robot"),
            secondary=ape.ContactMatch(
                mode="geom", pattern="torso_frame_compute_budget", entity="robot"),
            fields=("found", "force"), reduce="maxforce", num_slots=1,
        ),
        ape.ContactSensorCfg(
            name="nonsole_ground_contact",
            primary=ape.ContactMatch(
                mode="geom", pattern=r"^(?!(?:left_sole|right_sole)$).*", entity="robot"),
            secondary=ape.ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
        ),
    )
    cfg.terminations["thigh_torso_contact"] = TerminationTermCfg(
        func=ape.mdp.illegal_contact,
        params={"sensor_name": "thigh_torso_contact", "force_threshold": 0.1},
    )
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.auto_reset = False
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
    if args.video:
        reveal_frame50_proxy_geometry_for_video(env.unwrapped.sim.mj_model)
    runner = load_runner_cls(source)(env, asdict(load_rl_cfg(source)), device="cuda:0")
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    robot = env.unwrapped.scene["robot"]
    sensors = env.unwrapped.scene.sensors
    site_ids = [robot.site_names.index(f"{side}_sole_center") for side in ("left", "right")]
    first_stance = robot.data.site_pos_w[0, site_ids[stance_slot], :2].cpu().numpy().copy()
    rated = torch.tensor([a.effort_limit for a in
                          cfg.scene.entities["robot"].articulation.actuators[:10]],
                         device="cuda:0")
    rows, frames = [], []
    audited_pairs = Counter()
    audit = mujoco.MjData(model)
    root_body_id = model.body("pelvis").id
    stance_site_id = model.site(f"{args.stance}_sole_center").id
    stance_sole_half_xy = model.geom(f"{args.stance}_sole").size[:2].copy()
    dt = env.unwrapped.step_dt
    for step in range(round(target_duration_s / dt)):
        with torch.inference_mode():
            action = policy(env.get_observations())
            _, _, done, _ = env.step(action)
        feet = [bool(x) for x in sensors["feet_ground_contact"].data.found[0].cpu().tolist()]
        soles = robot.data.site_pos_w[0, site_ids].cpu().numpy()
        effort = robot.data.actuator_force[0, :10].abs()
        quat = robot.data.root_link_pose_w[0, 3:7].cpu().numpy()
        tilt = math.degrees(math.acos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1)))
        audit.qpos[:] = env.unwrapped.sim.data.qpos[0].cpu().numpy()
        audit.qvel[:] = env.unwrapped.sim.data.qvel[0].cpu().numpy()
        mujoco.mj_forward(model, audit)
        com_local = (audit.site_xmat[stance_site_id].reshape(3, 3).T @
                     (audit.subtree_com[root_body_id] - audit.site_xpos[stance_site_id]))[:2]
        for contact in audit.contact[:audit.ncon]:
            pair = tuple(sorted((model.geom(contact.geom1).name,
                                 model.geom(contact.geom2).name)))
            if "floor" not in pair:
                audited_pairs[pair] += 1
        rows.append({
            "time_s": round((step + 1) * dt, 3),
            "pelvis_z_m": round(float(robot.data.root_link_pos_w[0, 2]), 4),
            "tilt_deg": round(tilt, 2), "feet_grounded": feet,
            "swing_z_m": round(float(soles[swing_slot, 2]), 4),
            "stance_slip_m": round(float(np.linalg.norm(
                soles[stance_slot, :2] - first_stance)), 4),
            "com_in_stance_sole_xy_m": [round(float(x), 4) for x in com_local],
            "com_stance_sole_margin_xy_m": [round(float(x), 4) for x in
                                             stance_sole_half_xy - np.abs(com_local)],
            "nonsole_floor_contact": bool(sensors["nonsole_ground_contact"].data.found[0].any()),
            "thigh_torso_contact": bool(sensors["thigh_torso_contact"].data.found[0].any()),
            "hand_force_n": round(float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).max()), 3),
            "rated_leg": bool((effort >= .995 * rated).any()),
        })
        if args.video and step % 2 == 0:
            frames.append(env.unwrapped.render())
        if bool(done[0]):
            break
    longest = current = 0
    support_pair = [args.stance == "left", args.stance == "right"]
    for row in rows:
        current = current + 1 if row["feet_grounded"] == support_pair and row["swing_z_m"] > .025 else 0
        longest = max(longest, current)
    final = rows[-1]
    support_rows = [row for row in rows if
                    row["feet_grounded"] == support_pair and
                    row["swing_z_m"] > .025]
    report = {
        "schema": "robot-learning-lab.cove50-native-stand-transition.v1",
        "method": "Timed stand-target ramp plus PPO residual through selected native MJLab motors; starts upright",
        "stance": args.stance,
        "freeze_swing_actions": args.freeze_swing,
        "phase_reward_and_command": args.phase_rewards,
        "arm_clearance_bump_rad": args.arm_clearance_bump_rad,
        "staged_shift_then_lift": args.staged,
        "urgent_support_shaping": args.urgent_support,
        "schedule_s": {"lift": args.lift_s, "hold": args.hold_s, "lower": args.lower_s},
        "body_sha256": hashlib.sha256(args.body_xml.read_bytes()).hexdigest(),
        "rail_v": args.rail_v, "command_delay_ms": args.delay_ms,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "pose_json": str(args.pose_json), "preload_json": str(args.preload_json),
        "checkpoint": str(args.checkpoint), "seed": args.seed,
        "duration_s": final["time_s"],
        "longest_single_support_s": round(longest * dt, 3),
        "final_two_foot_stand": final["time_s"] >= target_duration_s - .02 and
        final["feet_grounded"] == [True, True] and
        final["pelvis_z_m"] > 0.20 and final["tilt_deg"] < 15.0,
        "max_stance_slip_m": max(row["stance_slip_m"] for row in rows),
        "stance_sole_half_widths_xy_m": [float(x) for x in stance_sole_half_xy],
        "min_com_stance_sole_margin_xy_m": [
            min(row["com_stance_sole_margin_xy_m"][axis] for row in rows)
            for axis in range(2)],
        "min_single_support_com_margin_xy_m": [
            min(row["com_stance_sole_margin_xy_m"][axis] for row in support_rows)
            for axis in range(2)] if support_rows else None,
        "rated_leg_fraction": round(sum(row["rated_leg"] for row in rows) / len(rows), 4),
        "nonsole_floor_contact_frames": sum(row["nonsole_floor_contact"] for row in rows),
        "thigh_torso_contact_frames": sum(row["thigh_torso_contact"] for row in rows),
        "hand_contact_frames": sum(row["hand_force_n"] >= 1.0 for row in rows),
        "cpu_audited_self_contact_pairs": [
            {"pair": list(pair), "sampled_frames": count}
            for pair, count in audited_pairs.most_common()],
        "last": final, "timeseries": rows,
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
