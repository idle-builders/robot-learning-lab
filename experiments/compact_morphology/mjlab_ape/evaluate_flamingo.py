"""Fresh saved-policy replay of the ape one-foot hold; zero-action control too."""

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import mujoco

import ape_walk.task
import ape_walk.foot_place_command  # Register the commanded touchdown lesson.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--zero-actions", action="store_true")
    parser.add_argument("--hold-pose-json", type=Path,
                        help="Physics positive control: constant servo targets from a screened pose")
    parser.add_argument("--task", default=ape_walk.task.ELBOW_FLAMINGO_HOLD_TASK_ID)
    parser.add_argument("--body-xml", type=Path)
    parser.add_argument("--cove50-body-xml", type=Path,
                        help="Retarget the selected 21-axis Cove body, motor law and raised-foot reset")
    parser.add_argument("--stance", choices=("left", "right"), default="right")
    parser.add_argument("--deterministic-reset", action="store_true",
                        help="Remove reset joint/tilt noise for a physics feasibility probe")
    parser.add_argument("--reset-joint-noise", type=float)
    parser.add_argument("--reset-tilt-noise-deg", type=float)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    parser.add_argument("--camera-elevation", type=float, default=-30.0)
    parser.add_argument("--state-out", type=Path)
    parser.add_argument("--target-dx", type=float,
                        help="Fix a forward-touchdown command for the placement lesson")
    args = parser.parse_args()
    if args.checkpoint is None and not args.zero_actions and not args.hold_pose_json:
        parser.error("checkpoint required unless --zero-actions or --hold-pose-json")
    if args.body_xml and args.cove50_body_xml:
        parser.error("Use one body override")

    task = args.task
    cfg = load_env_cfg(task, play=True)
    if args.target_dx is not None:
        cfg.commands["twist"].ranges.lin_vel_x = (args.target_dx, args.target_dx)
    if args.body_xml:
        cfg.scene.entities["robot"].spec_fn = partial(ape_walk.task._ape_spec, args.body_xml.resolve())
    if args.cove50_body_xml:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from experiments.compact_morphology.cove50_policy_transfer.train_foot_hold import retarget
        body = args.cove50_body_xml.resolve()
        retarget(cfg, body, mujoco.MjModel.from_xml_path(str(body)),
                 args.stance, variance=False)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    if args.deterministic_reset:
        cfg.events["set_flamingo_state"].params["tilt_noise"] = 0.0
        cfg.events["set_flamingo_state"].params["joint_noise_std"] = 0.0
    if args.reset_joint_noise is not None:
        cfg.events["set_flamingo_state"].params["joint_noise_std"] = args.reset_joint_noise
    if args.reset_tilt_noise_deg is not None:
        cfg.events["set_flamingo_state"].params["tilt_noise"] = math.radians(args.reset_tilt_noise_deg)
    cfg.auto_reset = False
    cfg.episode_length_s = 6.5
    if args.video:
        cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_BODY
        cfg.viewer.entity_name = "robot"
        cfg.viewer.body_name = "pelvis"
        cfg.viewer.distance = 0.8
        cfg.viewer.azimuth = args.camera_azimuth
        cfg.viewer.elevation = args.camera_elevation
        cfg.viewer.width = 640
        cfg.viewer.height = 480
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device="cuda:0",
                              render_mode="rgb_array" if args.video else None),
        clip_actions=load_rl_cfg(task).clip_actions,
    )
    robot = env.unwrapped.scene["robot"]
    leg_rated_effort = torch.tensor(
        ([act.effort_limit for act in cfg.scene.entities["robot"].articulation.actuators[:10]]
         if args.cove50_body_xml else [1.56] * 10),
        device="cuda:0",
    )
    if args.hold_pose_json:
        pose = json.loads(args.hold_pose_json.read_text())["joints"]
        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        target = robot.data.default_joint_pos.clone()
        for name, angle in pose.items():
            target[:, robot.joint_names.index(name)] = angle
        scale = action_term.scale
        offset = action_term.offset
        raw = torch.where(scale != 0, (target - offset) / torch.clamp(scale, min=1e-6),
                          torch.zeros_like(target))
        def policy(_obs):
            return raw
    elif args.zero_actions:
        def policy(_obs):
            return torch.zeros_like(robot.data.joint_pos)
    else:
        runner = load_runner_cls(task)(env, asdict(load_rl_cfg(task)), device="cuda:0")
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")

    sensors = env.unwrapped.scene.sensors
    stance_idx = 0 if args.stance == "left" else 1
    swing_idx = 1 - stance_idx
    dt = env.unwrapped.step_dt
    records = []
    frames = []
    qpos_frames = []
    stance_slip = 0.0
    sole_ids = [robot.site_names.index(f"{side}_sole_center") for side in ("left", "right")]
    initial_soles = robot.data.site_pos_w[0, sole_ids, :].cpu().tolist()
    initial_stance_xy = initial_soles[stance_idx][:2]
    initial_swing_xy = initial_soles[swing_idx][:2]
    initial_quat = robot.data.root_link_quat_w[0].cpu().tolist()
    iw, ix, iy, iz = initial_quat
    initial_yaw = math.atan2(2 * (iw * iz + ix * iy), 1 - 2 * (iy * iy + iz * iz))
    initial_dx = initial_swing_xy[0] - initial_stance_xy[0]
    initial_dy = initial_swing_xy[1] - initial_stance_xy[1]
    initial_forward = math.cos(initial_yaw) * initial_dx + math.sin(initial_yaw) * initial_dy
    for step in range(round(6.0 / dt)):
        with torch.inference_mode():
            action = policy(env.get_observations())
            _, reward, done, _ = env.step(action)
        if args.state_out:
            qpos_frames.append(np.concatenate((
                robot.data.root_link_pose_w[0].cpu().numpy(),
                robot.data.joint_pos[0].cpu().numpy(),
            )))
        if args.video and step % 2 == 0:
            frames.append(env.unwrapped.render())
        soles = sensors["foot_height_scan"].data.frame_pos_w[0].cpu().tolist()
        contact = sensors["feet_ground_contact"].data.found[0].cpu().tolist()
        forces = sensors["feet_ground_contact"].data.force[0].norm(dim=-1).cpu().tolist()
        com = env.unwrapped.sim.data.subtree_com[
            0, robot.indexing.root_body_id
        ].cpu().tolist()
        com_distance = math.hypot(com[0] - soles[stance_idx][0], com[1] - soles[stance_idx][1])
        quat = robot.data.root_link_pose_w[0, 3:7].cpu().tolist()
        qw, qx, qy, qz = quat
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        dx = soles[swing_idx][0] - soles[stance_idx][0]
        dy = soles[swing_idx][1] - soles[stance_idx][1]
        local_forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
        tilt = math.degrees(math.acos(max(-1.0, min(1.0,
            1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)))))
        if bool(contact[stance_idx]):
            stance_slip += float(
                sensors[f"robot/{args.stance}_sole_center_global_linvel"].data[0, :2].norm()
            ) * dt
        effort = robot.data.actuator_force[0, :10].abs()
        self_body_contact = (sensors["self_body_contact"].data.found[0].any().item()
                             if "self_body_contact" in sensors else None)
        records.append({
            "time_s": round((step + 1) * dt, 3),
            "pelvis_z_m": round(float(robot.data.root_link_pos_w[0, 2]), 4),
            "tilt_deg": round(tilt, 2),
            "root_xy_m": [round(v, 4) for v in robot.data.root_link_pos_w[0, :2].cpu().tolist()],
            "swing_height_m": round(soles[swing_idx][2], 4),
            "stance_height_m": round(soles[stance_idx][2], 4),
            "swing_x_from_stance_m": round(soles[swing_idx][0] - soles[stance_idx][0], 4),
            "swing_y_from_stance_m": round(soles[swing_idx][1] - soles[stance_idx][1], 4),
            "swing_forward_from_stance_m": round(local_forward, 4),
            "swing_lateral_from_stance_m": round(local_lateral, 4),
            "root_yaw_rad": round(yaw, 4),
            "feet_grounded": [bool(v) for v in contact],
            "foot_force_n": [round(v, 2) for v in forces],
            "whole_com_from_stance_xy_m": [
                round(com[i] - soles[stance_idx][i], 4) for i in (0, 1)
            ],
            "whole_com_horizontal_distance_from_stance_m": round(com_distance, 4),
            "stance_slip_m": round(stance_slip, 4),
            "stance_from_initial_xy_m": round(math.dist(soles[stance_idx][:2], initial_stance_xy), 4),
            "swing_from_initial_xy_m": round(math.dist(soles[swing_idx][:2], initial_swing_xy), 4),
            "hand_force_n": round(float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).max()), 3),
            "self_body_contact": self_body_contact,
            "max_leg_torque_nm": round(float(effort.max()), 3),
            "leg_joint_effort_nm": [round(v, 3) for v in robot.data.actuator_force[0, :10].cpu().tolist()],
            "leg_joint_pos_rad": [round(v, 3) for v in robot.data.joint_pos[0, :10].cpu().tolist()],
            "leg_raw_action": [round(v, 3) for v in action[0, :10].cpu().tolist()],
            "leg_torque_saturated": bool((effort >= leg_rated_effort * 0.995).any()),
            "reward": round(float(reward[0]), 4),
        })
        if bool(done[0]):
            break

    def window(t0: float, t1: float) -> dict:
        rows = [r for r in records if t0 <= r["time_s"] < t1]
        if not rows:
            return {}
        return {
            "single_support_fraction": round(sum(
                r["feet_grounded"][stance_idx] and not r["feet_grounded"][swing_idx] and
                r["swing_height_m"] > 0.03 for r in rows
            ) / len(rows), 3),
            "mean_swing_height_m": round(sum(r["swing_height_m"] for r in rows) / len(rows), 4),
            "mean_com_xy_from_stance_m": [round(sum(
                r["whole_com_from_stance_xy_m"][i] for r in rows
            ) / len(rows), 4) for i in (0, 1)],
            "torque_saturated_fraction": round(sum(r["leg_torque_saturated"] for r in rows) / len(rows), 3),
            "mean_tilt_deg": round(sum(r["tilt_deg"] for r in rows) / len(rows), 2),
        }

    longest_hold_steps = 0
    hold_steps = 0
    for row in records:
        held = (row["feet_grounded"][stance_idx] and not row["feet_grounded"][swing_idx]
                and row["swing_height_m"] > 0.03
                and row["pelvis_z_m"] > 0.12
                and row["hand_force_n"] == 0.0)
        hold_steps = hold_steps + 1 if held else 0
        longest_hold_steps = max(longest_hold_steps, hold_steps)
    result = {
        "task": task, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "body_xml": str(args.body_xml or args.cove50_body_xml or getattr(cfg.scene.entities["robot"].spec_fn, "args", (ape_walk.task.ELBOW_MODEL,))[0]),
        "body_sha256": hashlib.sha256(Path(args.body_xml or args.cove50_body_xml or getattr(cfg.scene.entities["robot"].spec_fn, "args", (ape_walk.task.ELBOW_MODEL,))[0]).read_bytes()).hexdigest(),
        "leg_rated_effort_limits_nm": [round(float(v), 6) for v in leg_rated_effort],
        "stance_side": args.stance,
        "target_dx_m": args.target_dx,
        "deterministic_reset": args.deterministic_reset,
        "reset_joint_noise": cfg.events["set_flamingo_state"].params["joint_noise_std"],
        "reset_tilt_noise_deg": math.degrees(cfg.events["set_flamingo_state"].params["tilt_noise"]),
        "zero_actions": args.zero_actions,
        "hold_pose_json": str(args.hold_pose_json) if args.hold_pose_json else None,
        "hold_action": raw[0].cpu().tolist() if args.hold_pose_json else None,
        "seed": args.seed,
        "duration_s": records[-1]["time_s"],
        "fell_or_terminated": (records[-1]["time_s"] < 5.99 or
                               min(r["pelvis_z_m"] for r in records) < 0.10),
        "longest_single_support_s": round(longest_hold_steps * dt, 3),
        "initial_forward_placement_m": round(initial_forward, 4),
        "final_forward_placement_m": records[-1]["swing_forward_from_stance_m"],
        "forward_change_from_reset_m": round(records[-1]["swing_forward_from_stance_m"] - initial_forward, 4),
        "final_two_foot_stand": (records[-1]["feet_grounded"] == [True, True]
                                 and records[-1]["pelvis_z_m"] > 0.16),
        "early": window(0.5, 1.5), "late": window(4.0, 6.0),
        "peak_hand_force_n": max(r["hand_force_n"] for r in records),
        "stance_slip_m": round(stance_slip, 4),
        "max_stance_from_initial_xy_m": max(r["stance_from_initial_xy_m"] for r in records),
        "final_stance_from_initial_xy_m": records[-1]["stance_from_initial_xy_m"],
        "final_swing_from_initial_xy_m": records[-1]["swing_from_initial_xy_m"],
        "max_leg_torque_nm": max(r["max_leg_torque_nm"] for r in records),
        "self_body_contact_fraction": (round(sum(bool(r["self_body_contact"]) for r in records) / len(records), 3)
                                       if "self_body_contact" in sensors else None),
        "timeseries": records,
    }
    if args.target_dx is not None:
        def landed_in_zone(row):
            return (row["feet_grounded"] == [True, True]
                    and abs(row["swing_forward_from_stance_m"] - args.target_dx) <= 0.012
                    and abs(row["swing_lateral_from_stance_m"] - ape_walk.foot_place_command.LANE_Y_M) <= 0.015)
        landed = [r for r in records if landed_in_zone(r)]
        result["forward_touchdown"] = {
            "first_target_contact_s": landed[0]["time_s"] if landed else None,
            "target_contact_fraction": round(len(landed) / len(records), 3),
            "last_half_second_target_fraction": round(sum(landed_in_zone(r) for r in records[-25:]) / min(25, len(records)), 3),
            "final_target_error_m": round(records[-1]["swing_forward_from_stance_m"] - args.target_dx, 4),
            "moved_foot_forward_from_reset": (records[-1]["swing_forward_from_stance_m"] - initial_forward >= 0.025),
            "success": (not result["fell_or_terminated"] and result["final_two_foot_stand"]
                        and sum(landed_in_zone(r) for r in records[-25:]) >= 20),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if args.state_out:
        args.state_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.state_out, qpos=np.asarray(qpos_frames), dt=dt)
    print(json.dumps({k: v for k, v in result.items() if k != "timeseries"}, indent=2))
    if args.video:
        import imageio
        imageio.mimwrite(args.video, frames, fps=25, macro_block_size=16)
    env.close()


if __name__ == "__main__":
    main()
