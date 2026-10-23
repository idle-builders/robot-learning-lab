"""Replay one saved ape arm policy through raise/lower commands in native MJLab."""

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import mujoco

import ape_walk.task
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--device", default="cuda:0",
                        help="Native simulation device; CPU traces are diagnostic only")
    parser.add_argument("--zero-actions", action="store_true")
    parser.add_argument("--constant-shoulder-action", type=float)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--independent-arm", choices=("left", "right", "both"))
    parser.add_argument("--body-xml", type=Path)
    parser.add_argument("--cove50-body-xml", type=Path,
                        help="Retarget the 21-axis Cove body, motor law and standing height")
    parser.add_argument("--motor-rail-v", type=float, default=12.0)
    parser.add_argument("--command-delay-ms", type=float, default=0.0)
    parser.add_argument("--shoulder-target-margin-rad", type=float)
    parser.add_argument("--action-smoothing-alpha", type=float,
                        help="Apply causal exponential smoothing to policy actions before the motor target")
    parser.add_argument("--state-out", type=Path)
    parser.add_argument("--native-trajectory-out", type=Path,
                        help="Save aligned pre-action observations, full actions, targets and states")
    parser.add_argument("--student-bc", action="store_true",
                        help="Load the shared Direct Frame squat/arm BC candidate")
    args = parser.parse_args()
    if not args.zero_actions and args.constant_shoulder_action is None and args.checkpoint is None:
        parser.error("checkpoint required unless --zero-actions is set")
    if args.body_xml and args.cove50_body_xml:
        parser.error("Use one body override")
    if args.action_smoothing_alpha is not None and not 0 < args.action_smoothing_alpha <= 1:
        parser.error("Action smoothing alpha must be in (0, 1]")
    if args.student_bc and (not args.cove50_body_xml or args.independent_arm
                            or args.zero_actions or args.constant_shoulder_action is not None):
        parser.error("Shared BC arm replay requires the Cove both-arm task and checkpoint")

    task = (ape_walk.task.ELBOW_ARM_INDEPENDENT_TASK_ID
            if args.independent_arm else ape_walk.task.ELBOW_ARM_TASK_ID)
    cfg = load_env_cfg(task, play=True)
    if args.body_xml:
        cfg.scene.entities["robot"].spec_fn = partial(
            ape_walk.task._ape_spec, args.body_xml.resolve()
        )
    if args.cove50_body_xml:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from experiments.compact_morphology.cove50_policy_transfer.train_posture_arm import retarget
        from ape_walk.physics.frame50 import (configure_frame50_motors,
                                              clip_frame50_shoulder_targets,
                                              reveal_frame50_proxy_geometry_for_video)
        body = args.cove50_body_xml.resolve()
        retarget(cfg, body, mujoco.MjModel.from_xml_path(str(body)),
                 "independent-arms" if args.independent_arm else "arms",
                 variance=False)
        configure_frame50_motors(cfg, body, rail_v=args.motor_rail_v,
                                 delay_s=args.command_delay_ms / 1000.0)
        if args.shoulder_target_margin_rad is not None:
            clip_frame50_shoulder_targets(cfg, body,
                                          margin_rad=args.shoulder_target_margin_rad)
    elif (args.motor_rail_v != 12.0 or args.command_delay_ms != 0.0
          or args.shoulder_target_margin_rad is not None):
        parser.error("Motor and target sensitivity requires --cove50-body-xml")
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.auto_reset = False
    cfg.episode_length_s = 10.5
    cfg.commands["twist"].resampling_time_range = (100.0, 100.0)
    if args.independent_arm:
        cfg.commands["right_arm"].resampling_time_range = (100.0, 100.0)
    if args.video:
        cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
        cfg.viewer.lookat = (0.0, 0.0, 0.22)
        cfg.viewer.distance = 0.7
        cfg.viewer.azimuth = 90.0
        cfg.viewer.elevation = -12.0
        cfg.viewer.width = 640
        cfg.viewer.height = 480
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(
                               cfg=cfg, device=args.device,
                               render_mode="rgb_array" if args.video else None),
                           clip_actions=load_rl_cfg(task).clip_actions)
    robot = env.unwrapped.scene["robot"]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    if args.video and args.cove50_body_xml:
        reveal_frame50_proxy_geometry_for_video(env.unwrapped.sim.mj_model)
    shoulder_indices = [robot.joint_names.index(f"{side}_shoulder_pitch")
                        for side in ("left", "right")]
    elbow_indices = [robot.joint_names.index(f"{side}_elbow_pitch")
                     for side in ("left", "right")]
    shoulder_effort_indices = [13, 18] if args.cove50_body_xml else [12, 17]
    shoulder_rated_effort_nm = (0.980665 * args.motor_rail_v / 12.0
                                if args.cove50_body_xml else None)
    if args.constant_shoulder_action is not None:
        def policy(_):
            action = torch.zeros_like(robot.data.joint_pos)
            action[:, shoulder_indices] = args.constant_shoulder_action
            return action
    elif args.zero_actions:
        def policy(_):
            return torch.zeros_like(robot.data.joint_pos)
    elif args.student_bc:
        from experiments.compact_morphology.cove50_low_level.multicommand_bc import load_student
        policy = load_student(args.checkpoint, args.device, action_term.target_names,
                              action_term.scale, action_term.offset, "both_arms")
    else:
        runner = load_runner_cls(task)(env, asdict(load_rl_cfg(task)), device=args.device)
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
    command = env.unwrapped.command_manager.get_term("twist")
    right_command = (env.unwrapped.command_manager.get_term("right_arm")
                     if args.independent_arm else None)
    sensor = env.unwrapped.scene["hand_ground_contact"]
    dt = env.unwrapped.step_dt
    rows = []
    frames = []
    qpos_frames = []
    smoothed_action = None
    initial_qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy().copy()
    initial_qvel = env.unwrapped.sim.data.qvel[0].cpu().numpy().copy()
    native_frames = {name: [] for name in (
        "actor_observation", "command", "raw_action", "unsmoothed_action", "applied_action",
        "processed_target", "pre_qpos", "pre_qvel", "post_qpos", "post_qvel",
    )} if args.native_trajectory_out else None
    for step in range(round(10.0 / dt)):
        t = step * dt
        raised = float(2 <= t < 6)
        command.vel_command_b[:, 0] = raised if args.independent_arm != "right" else 0.0
        if right_command is not None:
            right_command.vel_command_b[:, 0] = raised if args.independent_arm != "left" else 0.0
        with torch.inference_mode():
            observations = env.get_observations()
            if native_frames is not None:
                native_frames["actor_observation"].append(
                    observations["actor"][0].cpu().numpy().copy())
                native_frames["command"].append(np.asarray((
                    float(command.vel_command_b[0, 0]),
                    float(right_command.vel_command_b[0, 0]) if right_command is not None
                    else float(command.vel_command_b[0, 0]),
                ), dtype=np.float32))
                native_frames["pre_qpos"].append(
                    env.unwrapped.sim.data.qpos[0].cpu().numpy().copy())
                native_frames["pre_qvel"].append(
                    env.unwrapped.sim.data.qvel[0].cpu().numpy().copy())
            action = policy(observations)
            if args.action_smoothing_alpha is not None:
                if smoothed_action is None:
                    smoothed_action = action.clone()
                else:
                    smoothed_action = (
                        args.action_smoothing_alpha * action
                        + (1.0 - args.action_smoothing_alpha) * smoothed_action
                    )
                applied_action = smoothed_action
            else:
                applied_action = action
            _, _, done, _ = env.step(applied_action)
        if native_frames is not None:
            for name, values in (
                ("raw_action", action),
                ("unsmoothed_action", action),
                ("applied_action", applied_action),
                ("processed_target", action_term._processed_actions),
                ("post_qpos", env.unwrapped.sim.data.qpos),
                ("post_qvel", env.unwrapped.sim.data.qvel),
            ):
                native_frames[name].append(values[0].detach().cpu().numpy().copy())
        if args.state_out:
            qpos_frames.append(np.concatenate((
                robot.data.root_link_pose_w[0].cpu().numpy(),
                robot.data.joint_pos[0].cpu().numpy(),
            )))
        if args.video and step % 2 == 0:
            frames.append(env.unwrapped.render())
        hands = robot.data.site_pos_w[0, [robot.site_names.index("left_hand_center"),
                                          robot.site_names.index("right_hand_center")], 2]
        quat = robot.data.root_link_quat_w[0]
        trunk_upright = torch.clamp(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2), -1.0, 1.0)
        rows.append({
            "t": round(t + dt, 3),
            "raise_command": [int(command.vel_command_b[0, 0]),
                              int(right_command.vel_command_b[0, 0]) if right_command is not None
                              else int(command.vel_command_b[0, 0])],
            "shoulder_pitch_rad": [round(float(x), 3) for x in robot.data.joint_pos[0, shoulder_indices]],
            "elbow_pitch_rad": [round(float(x), 3) for x in robot.data.joint_pos[0, elbow_indices]],
            "hand_height_m": [round(float(x), 4) for x in hands],
            "pelvis_height_m": round(float(robot.data.root_link_pos_w[0, 2]), 4),
            "trunk_tilt_deg": round(float(torch.rad2deg(torch.acos(trunk_upright))), 2),
            "hand_force_n": round(float(sensor.data.force[0].norm(dim=-1).max()), 3),
            # Actuator order comes from XML (legs, arms, head), unlike MJLab joint order.
            "shoulder_effort_nm": [round(float(x), 4) for x in robot.data.actuator_force[0, shoulder_effort_indices]],
            "shoulder_raw_action": [round(float(x), 3) for x in action[0, shoulder_indices]],
            "shoulder_applied_action": [round(float(x), 3) for x in applied_action[0, shoulder_indices]],
        })
        if bool(done[0]):
            break

    def window(lo, hi):
        r = [x for x in rows if lo <= x["t"] < hi]
        if not r:
            return None
        return {
            "hand_height_m": [round(sum(x["hand_height_m"][i] for x in r) / len(r), 4)
                              for i in range(2)],
            "shoulder_pitch_rad": [round(sum(x["shoulder_pitch_rad"][i] for x in r) / len(r), 3)
                                   for i in range(2)],
            "elbow_pitch_rad": round(sum(sum(x["elbow_pitch_rad"]) / 2 for x in r) / len(r), 3),
            "pelvis_height_m": round(sum(x["pelvis_height_m"] for x in r) / len(r), 4),
            "max_trunk_tilt_deg": round(max(x["trunk_tilt_deg"] for x in r), 2),
            "max_shoulder_effort_nm": round(max(abs(z) for x in r for z in x["shoulder_effort_nm"]), 3),
            "mean_abs_shoulder_effort_nm": round(sum(sum(abs(z) for z in x["shoulder_effort_nm"]) / 2 for x in r) / len(r), 3),
            "shoulder_saturation_fraction": round(sum(
                max(abs(z) for z in x["shoulder_effort_nm"])
                >= (shoulder_rated_effort_nm - 0.002 if shoulder_rated_effort_nm else 0.98)
                for x in r) / len(r), 3),
            "shoulder_saturation_fraction_by_side": [round(sum(
                abs(x["shoulder_effort_nm"][i])
                >= (shoulder_rated_effort_nm - 0.002 if shoulder_rated_effort_nm else 0.98)
                for x in r) / len(r), 3) for i in range(2)],
        }

    report = {"task": task, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
              "independent_arm": args.independent_arm,
              "body_xml": str(args.body_xml or args.cove50_body_xml or ape_walk.task.ELBOW_MODEL),
              "body_sha256": hashlib.sha256(
                  (args.body_xml or args.cove50_body_xml or ape_walk.task.ELBOW_MODEL).read_bytes()
              ).hexdigest(),
              "zero_actions": args.zero_actions,
              "device": args.device,
              "student_bc": args.student_bc,
              "constant_shoulder_action": args.constant_shoulder_action, "seed": args.seed,
              "motor_rail_v": args.motor_rail_v,
              "command_delay_ms": args.command_delay_ms,
              "shoulder_target_margin_rad": args.shoulder_target_margin_rad,
              "action_smoothing_alpha": args.action_smoothing_alpha,
              "shoulder_rated_effort_nm": shoulder_rated_effort_nm,
              "duration_s": rows[-1]["t"], "before": window(0.5, 1.5),
              "raised": window(4.5, 5.5), "lowered": window(8.5, 9.5),
              "peak_hand_force_n": max(x["hand_force_n"] for x in rows),
              "timeseries": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.state_out:
        args.state_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.state_out, qpos=np.asarray(qpos_frames), dt=dt)
    if args.native_trajectory_out:
        args.native_trajectory_out.parent.mkdir(parents=True, exist_ok=True)
        def action_vector(value):
            value = value[0].cpu().numpy() if isinstance(value, torch.Tensor) else value
            return np.broadcast_to(value, (len(action_term.target_names),)).copy()
        np.savez_compressed(
            args.native_trajectory_out,
            initial_qpos=initial_qpos, initial_qvel=initial_qvel,
            joint_names=np.asarray(robot.joint_names),
            target_names=np.asarray(action_term.target_names),
            action_scale=action_vector(action_term.scale),
            action_offset=action_vector(action_term.offset),
            step_dt_s=np.asarray(dt),
            **{name: np.asarray(values) for name, values in native_frames.items()},
        )
    print(json.dumps({k: v for k, v in report.items() if k != "timeseries"}, indent=2))
    if args.video:
        import imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=25, macro_block_size=16)
    env.close()


if __name__ == "__main__":
    main()
