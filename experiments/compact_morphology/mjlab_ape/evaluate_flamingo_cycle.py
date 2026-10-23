"""Fresh saved-policy replay of a commanded stand → one-foot → stand cycle."""

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
import ape_walk.foot_upright
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--zero-actions", action="store_true")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--task", default=ape_walk.task.ELBOW_FLAMINGO_CYCLE_TASK_ID)
    parser.add_argument("--body-xml", type=Path)
    parser.add_argument("--cove50-body-xml", type=Path)
    parser.add_argument("--swing-contact-weight", type=float)
    parser.add_argument("--start-in-pose", action="store_true")
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--state-out", type=Path)
    parser.add_argument("--swing-hip-pitch-bias", type=float, default=0.0)
    parser.add_argument("--swing-knee-bias", type=float, default=0.0)
    parser.add_argument("--swing-ankle-pitch-bias", type=float, default=0.0)
    parser.add_argument("--hold-checkpoint", type=Path,
                        help="Diagnostic handoff to a separately learned one-foot-hold actor")
    parser.add_argument("--handoff-at", type=float, default=4.0)
    parser.add_argument("--handoff-blend-s", type=float, default=0.5)
    args = parser.parse_args()
    if args.checkpoint is None and not args.zero_actions:
        parser.error("checkpoint required unless --zero-actions")
    if args.body_xml and args.cove50_body_xml:
        parser.error("Use one body override")

    task = args.task
    cfg = load_env_cfg(task, play=True)
    if args.body_xml:
        cfg.scene.entities["robot"].spec_fn = partial(ape_walk.task._ape_spec, args.body_xml.resolve())
        cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
    if args.cove50_body_xml:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from ape_walk.physics.frame50 import configure_frame50_motors
        from experiments.compact_morphology.cove50_policy_transfer.train_foot_hold import (
            RAISED_Z, cove_pose,
        )

        body = args.cove50_body_xml.resolve()
        if mujoco.MjModel.from_xml_path(str(body)).nu != 21:
            raise ValueError("Expected the selected 21-axis Cove body")
        cfg.scene.entities["robot"].spec_fn = partial(ape_walk.task._ape_spec, body)
        cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
        cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, 0.2305)
        configure_frame50_motors(cfg, body, rail_v=12.0, delay_s=0.0)
        cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
        right, left = cove_pose("right"), cove_pose("left")
        cfg.rewards["pose_track"].params.update(
            pose_right=right, pose_left=left,
        )
        cfg.rewards["swing_foot_clear"].params["target"] = 0.09
        reset = cfg.events["set_flamingo_cycle_state"].params
        reset.update(pose_right=right, pose_left=left,
                     z_min=RAISED_Z, z_max=RAISED_Z + 0.004,
                     standing_z_min=0.2305, standing_z_max=0.2345)
    if args.swing_contact_weight is not None:
        cfg.rewards["swing_foot_contact"].weight = args.swing_contact_weight
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.auto_reset = False
    cfg.episode_length_s = 10.5
    cfg.commands["twist"].resampling_time_range = (100.0, 100.0)
    cfg.events["set_flamingo_cycle_state"].params["in_pose_prob"] = (
        1.0 if args.start_in_pose else 0.0
    )
    if args.video:
        cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_BODY
        cfg.viewer.entity_name = "robot"
        cfg.viewer.body_name = "pelvis"
        cfg.viewer.distance = 0.8
        cfg.viewer.width = 640
        cfg.viewer.height = 480
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device="cuda:0",
                              render_mode="rgb_array" if args.video else None),
        clip_actions=load_rl_cfg(task).clip_actions,
    )
    robot = env.unwrapped.scene["robot"]
    if args.zero_actions:
        def policy(_obs):
            return torch.zeros_like(robot.data.joint_pos)
    else:
        runner = load_runner_cls(task)(env, asdict(load_rl_cfg(task)), device="cuda:0")
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
    hold_policy = None
    if args.hold_checkpoint:
        hold_task = (ape_walk.foot_upright.LEFT_CLEAR_TASK_ID if args.side == "left"
                     else ape_walk.foot_upright.CLEAR_TASK_ID)
        hold_runner = load_runner_cls(hold_task)(
            env, asdict(load_rl_cfg(hold_task)), device="cuda:0"
        )
        hold_runner.load(str(args.hold_checkpoint),
                         load_cfg={"actor": True}, strict=True, map_location="cuda:0")
        hold_policy = hold_runner.get_inference_policy(device="cuda:0")

    sensors = env.unwrapped.scene.sensors
    command = env.unwrapped.command_manager.get_term("twist")
    side = ("right" if float(command.stance_side[0]) > 0 else "left") if args.start_in_pose else args.side
    stance_idx = 0 if side == "left" else 1
    swing_idx = 1 - stance_idx
    command._stance_side[:] = -1.0 if side == "left" else 1.0
    dt = env.unwrapped.step_dt
    rows = []
    frames = []
    qpos_frames = []
    action_frames = []
    slip = 0.0
    for step in range(round(10.0 / dt)):
        t = step * dt
        lift = args.start_in_pose or 2.0 <= t < 6.0
        command.vel_command_b[:, 0] = float(lift)
        command.vel_command_b[:, 1] = float(command._stance_side[0]) if lift else 0.0
        command.vel_command_b[:, 2] = 0.0
        with torch.inference_mode():
            actions = policy(env.get_observations())
            if hold_policy is not None:
                # The shift actor was trained with fixed arms; keep those slots
                # at its zero-action home until the learned hold actor takes over.
                actions[:, 10:] = 0.0
                if args.handoff_at <= t < 6.0:
                    hold_actions = hold_policy(env.get_observations())
                    mix = min(1.0, (t - args.handoff_at) / args.handoff_blend_s)
                    actions = (1.0 - mix) * actions + mix * hold_actions
            # Deliberate feasibility probe after the learned support shift.
            # This is an external action intervention, never a learned lift.
            if any((args.swing_hip_pitch_bias, args.swing_knee_bias,
                    args.swing_ankle_pitch_bias)):
                blend = max(0.0, min(1.0, (t - 3.0) / 0.5, (6.0 - t) / 0.5))
                swing_base = 0 if side == "right" else 5
                actions[0, swing_base + 2] += blend * args.swing_hip_pitch_bias
                actions[0, swing_base + 3] += blend * args.swing_knee_bias
                actions[0, swing_base + 4] += blend * args.swing_ankle_pitch_bias
            _, reward, done, _ = env.step(actions)
        if args.state_out:
            pose = robot.data.root_link_pose_w[0].cpu().numpy()
            joints = robot.data.joint_pos[0].cpu().numpy()
            qpos_frames.append(np.concatenate((pose, joints)))
            action_frames.append(actions[0].cpu().numpy().copy())
        if args.video and step % 2 == 0:
            frames.append(env.unwrapped.render())
        soles = sensors["foot_height_scan"].data.frame_pos_w[0].cpu().tolist()
        contact = sensors["feet_ground_contact"].data.found[0].cpu().tolist()
        force = sensors["feet_ground_contact"].data.force[0].norm(dim=-1).cpu().tolist()
        com = env.unwrapped.sim.data.subtree_com[
            0, robot.indexing.root_body_id
        ].cpu().tolist()
        quat = robot.data.root_link_pose_w[0, 3:7].cpu().tolist()
        tilt = math.degrees(math.acos(max(-1.0, min(1.0,
            1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)))))
        if bool(contact[stance_idx]):
            slip += float(sensors[f"robot/{side}_sole_center_global_linvel"].data[0, :2].norm()) * dt
        effort = robot.data.actuator_force[0, :10].abs()
        rows.append({
            "time_s": round(t + dt, 3),
            "command_lift": lift,
            "target_blend": round(float(command.alpha[0]), 3),
            "command_side": side,
            "pelvis_z_m": round(float(robot.data.root_link_pos_w[0, 2]), 4),
            "root_xy_m": [round(float(v), 4) for v in robot.data.root_link_pos_w[0, :2]],
            "tilt_deg": round(tilt, 2),
            "feet_grounded": [bool(v) for v in contact],
            "foot_force_n": [round(v, 2) for v in force],
            "swing_height_m": round(soles[swing_idx][2], 4),
            "whole_com_from_stance_xy_m": [round(com[i] - soles[stance_idx][i], 4) for i in (0, 1)],
            "stance_slip_m": round(slip, 4),
            "hand_force_n": round(float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).max()), 3),
            "leg_joint_effort_nm": [round(float(v), 3) for v in robot.data.actuator_force[0, :10]],
            "leg_torque_saturated": bool((effort >= 1.56).any()),
            "reward": round(float(reward[0]), 4),
        })
        if bool(done[0]):
            break

    def window(a: float, b: float) -> dict | None:
        selected = [r for r in rows if a <= r["time_s"] < b]
        if not selected:
            return None
        return {
            "both_feet_grounded_fraction": round(sum(all(r["feet_grounded"]) for r in selected) / len(selected), 3),
            "single_support_fraction": round(sum(r["feet_grounded"][stance_idx] and not r["feet_grounded"][swing_idx] and r["swing_height_m"] > 0.03 for r in selected) / len(selected), 3),
            "mean_swing_height_m": round(sum(r["swing_height_m"] for r in selected) / len(selected), 4),
            "mean_stance_force_n": round(sum(r["foot_force_n"][stance_idx] for r in selected) / len(selected), 2),
            "mean_swing_force_n": round(sum(r["foot_force_n"][swing_idx] for r in selected) / len(selected), 2),
            "mean_com_from_stance_xy_m": [round(sum(r["whole_com_from_stance_xy_m"][i] for r in selected) / len(selected), 4) for i in (0, 1)],
            "max_tilt_deg": round(max(r["tilt_deg"] for r in selected), 2),
            "hand_contact_frames": sum(r["hand_force_n"] >= 1.0 for r in selected),
            "torque_saturated_fraction": round(sum(r["leg_torque_saturated"] for r in selected) / len(selected), 3),
        }

    report = {
        "task": task,
        "body_xml": str(args.body_xml or args.cove50_body_xml or ape_walk.task.ELBOW_MODEL),
        "body_sha256": hashlib.sha256((args.body_xml or args.cove50_body_xml or ape_walk.task.ELBOW_MODEL).read_bytes()).hexdigest(),
        "swing_contact_reward_weight": cfg.rewards["swing_foot_contact"].weight,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "zero_actions": args.zero_actions,
        "side": side,
        "swing_action_bias_rad": [args.swing_hip_pitch_bias,
                                  args.swing_knee_bias,
                                  args.swing_ankle_pitch_bias],
        "start_in_pose": args.start_in_pose,
        "hold_checkpoint": str(args.hold_checkpoint) if args.hold_checkpoint else None,
        "handoff_at_s": args.handoff_at if args.hold_checkpoint else None,
        "handoff_blend_s": args.handoff_blend_s if args.hold_checkpoint else None,
        "seed": args.seed,
        "duration_s": rows[-1]["time_s"],
        "fell_or_terminated": rows[-1]["time_s"] < 9.99,
        "stance_slip_m": rows[-1]["stance_slip_m"],
        "stand_before": None if args.start_in_pose else window(1.0, 2.0),
        "lift_hold": window(5.0, 6.0),
        "stand_after": None if args.start_in_pose else window(9.0, 10.0),
        "timeseries": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.state_out:
        args.state_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.state_out, qpos=np.asarray(qpos_frames),
                            action=np.asarray(action_frames), dt=dt)
    print(json.dumps({k: v for k, v in report.items() if k != "timeseries"}, indent=2))
    if args.video:
        import imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=25, macro_block_size=16)
    env.close()


if __name__ == "__main__":
    main()
