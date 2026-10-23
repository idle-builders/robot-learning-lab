"""Fresh native MJLab replay of commanded left/right ape support transfer."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

import ape_walk.task
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, nargs="?")
    parser.add_argument("--side", choices=("left", "right", "center"), required=True)
    parser.add_argument("--magnitude", type=float, default=0.2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--zero-actions", action="store_true")
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    if not args.zero_actions and args.checkpoint is None:
        parser.error("checkpoint required unless --zero-actions is set")

    task = (ape_walk.task.ELBOW_TRANSFER_PARTIAL_TASK_ID if args.partial
            else ape_walk.task.ELBOW_TRANSFER_TASK_ID)
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.episode_length_s = 8.5
    cfg.auto_reset = False
    cfg.commands["twist"].resampling_time_range = (100.0, 100.0)
    if args.video:
        cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_BODY
        cfg.viewer.entity_name = "robot"
        cfg.viewer.body_name = "pelvis"
        cfg.viewer.distance = 1.0
        cfg.viewer.width = 640
        cfg.viewer.height = 480
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device="cuda:0",
                              render_mode="rgb_array" if args.video else None),
        clip_actions=load_rl_cfg(task).clip_actions,
    )
    robot = env.unwrapped.scene["robot"]
    if args.zero_actions:
        def policy(_):
            return torch.zeros_like(robot.data.joint_pos)
    else:
        runner = load_runner_cls(task)(env, asdict(load_rl_cfg(task)), device="cuda:0")
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
    sensors = env.unwrapped.scene.sensors
    command = env.unwrapped.command_manager.get_term("twist")
    wanted = {"left": args.magnitude, "right": -args.magnitude, "center": 0.0}[args.side]
    dt = env.unwrapped.step_dt
    records = []
    frames = []
    support_slip = 0.0
    for step in range(round(8.0 / dt)):
        command.vel_command_b[:, 0] = 0.0
        command.vel_command_b[:, 1] = wanted
        with torch.inference_mode():
            action = policy(env.get_observations())
            _, reward, done, _ = env.step(action)
        if args.video and step % 2 == 0:
            frames.append(env.unwrapped.render())
        feet = sensors["feet_ground_contact"].data.found[0].cpu().tolist()
        forces = sensors["feet_ground_contact"].data.force[0].norm(dim=-1).cpu().tolist()
        sole_xy = sensors["foot_height_scan"].data.frame_pos_w[0, :, :2].cpu().tolist()
        com_xy = robot.data.root_com_pos_w[0, :2].cpu().tolist()
        whole_com_xy = env.unwrapped.sim.data.subtree_com[
            0, robot.indexing.root_body_id, :2
        ].cpu().tolist()
        mid_xy = [(sole_xy[0][j] + sole_xy[1][j]) / 2 for j in range(2)]
        selected = 0 if wanted >= 0 else 1
        support_share = forces[selected] / max(sum(forces), 1e-6)
        foot_vel = sensors[f"robot/{args.side if args.side != 'center' else 'left'}_sole_center_global_linvel"].data[0, :2].norm().item()
        if feet[selected]:
            support_slip += foot_vel * dt
        quat = robot.data.root_link_pose_w[0, 3:7].cpu().tolist()
        tilt = math.degrees(math.acos(max(-1.0, min(1.0,
            1 - 2 * (quat[1] ** 2 + quat[2] ** 2)))))
        effort = robot.data.actuator_force[0, :10].abs()
        records.append({
            "time_s": round((step + 1) * dt, 3),
            "command_side": args.side,
            "command_heading_error_rad": round(float(command.vel_command_b[0, 2]), 3),
            "pelvis_z_m": round(float(robot.data.root_link_pos_w[0, 2]), 4),
            "root_xy_m": [round(x, 4) for x in robot.data.root_link_pos_w[0, :2].cpu().tolist()],
            "com_from_sole_mid_xy_m": [round(com_xy[j] - mid_xy[j], 4) for j in range(2)],
            "whole_com_from_sole_mid_xy_m": [
                round(whole_com_xy[j] - mid_xy[j], 4) for j in range(2)
            ],
            "sole_xy_m": [[round(x, 4) for x in row] for row in sole_xy],
            "feet_grounded": [bool(x) for x in feet],
            "foot_force_magnitude_n": [round(x, 2) for x in forces],
            "requested_support_share": round(support_share, 3),
            "support_slip_m": round(support_slip, 4),
            "hand_force_n": round(float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).max()), 3),
            "tilt_deg": round(tilt, 2),
            "max_leg_torque_nm": round(float(effort.max()), 3),
            "leg_joint_effort_nm": [round(x, 3) for x in robot.data.actuator_force[0, :10].cpu().tolist()],
            "leg_raw_action": [round(x, 3) for x in action[0, :10].cpu().tolist()],
            "leg_torque_saturated": bool((effort >= 1.56).any()),
            "step_reward": round(float(reward[0]), 4),
        })
        if bool(done[0]):
            break

    def window(start: float, end: float) -> dict:
        rows = [r for r in records if start <= r["time_s"] < end]
        return {
            "mean_support_share": round(sum(r["requested_support_share"] for r in rows) / len(rows), 3),
            "mean_com_y_from_sole_mid_m": round(sum(r["com_from_sole_mid_xy_m"][1] for r in rows) / len(rows), 4),
            "mean_whole_com_y_from_sole_mid_m": round(
                sum(r["whole_com_from_sole_mid_xy_m"][1] for r in rows) / len(rows), 4
            ),
            "both_feet_grounded_fraction": round(sum(all(r["feet_grounded"]) for r in rows) / len(rows), 3),
            "max_tilt_deg": round(max(r["tilt_deg"] for r in rows), 2),
            "saturated_leg_fraction": round(sum(r["leg_torque_saturated"] for r in rows) / len(rows), 3),
        } if rows else {}

    report = {
        "task": task, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "zero_actions": args.zero_actions, "side": args.side,
        "command_magnitude": args.magnitude, "seed": args.seed,
        "duration_s": records[-1]["time_s"], "ended_early": records[-1]["time_s"] < 7.99,
        "net_xy_m": [round(records[-1]["root_xy_m"][i] - records[0]["root_xy_m"][i], 4) for i in range(2)],
        "initial": window(0.5, 1.5), "late": window(6.5, 7.5),
        "peak_hand_force_n": max(r["hand_force_n"] for r in records),
        "support_slip_m": round(support_slip, 4), "timeseries": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "timeseries"}, indent=2))
    if args.video:
        import imageio
        imageio.mimwrite(args.video, frames, fps=25, macro_block_size=16)
    env.close()


if __name__ == "__main__":
    main()
