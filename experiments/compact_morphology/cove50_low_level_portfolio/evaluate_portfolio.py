"""Native one-scene portfolio replay through the explicit low-level router.

Saved gait and posture actors run in one scene without a physics reset.
This is a composition diagnostic, not a trained combined actor.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

import ape_walk.task as task
from controller import LowLevelCommand, PortfolioRouter
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--squat-actor", type=Path, required=True)
    parser.add_argument("--arm-actor", type=Path, required=True)
    parser.add_argument("--gait-actor", type=Path, required=True)
    parser.add_argument("--stop-actor", type=Path,
                        help="Saved forward actor controlling zero-command settling")
    parser.add_argument("--walk-velocity", type=float, default=0.25)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--native-evaluation", action="store_true",
                        help="Explicitly enable a frozen CUDA evaluation")
    parser.add_argument("--seed", type=int, default=54122)
    parser.add_argument("--rail-v", type=float, default=12.0)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1200,
                        help="Unscored exact-task sentinel may stop after one step")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 1200 or not 0.15 <= args.walk_velocity <= 0.4 or abs(args.yaw_rate) > 0.4:
        parser.error("expected 1..1200 steps, 0.15..0.4 m/s and yaw rate within +/-0.4 rad/s")
    if not args.stop_actor:
        parser.error("portfolio sequence requires a saved stop actor")
    if args.device != "cpu" and not args.native_evaluation:
        parser.error("CUDA requires an explicit frozen native evaluation")
    if args.out.exists() or (args.trace and args.trace.exists()) or (args.video and args.video.exists()):
        parser.error("original output already exists")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from experiments.compact_morphology.cove50_low_level.multicommand_bc import load_student
    from experiments.compact_morphology.cove50_policy_transfer.train_posture_arm import (
        expose_squat_blend, mask_squat_action_history, retarget,
    )
    from ape_walk.physics.frame50 import (
        clip_frame50_shoulder_targets, configure_frame50_motors,
        reveal_frame50_proxy_geometry_for_video,
    )

    body = args.body.resolve()
    body_sha = hashlib.sha256(body.read_bytes()).hexdigest()
    if body_sha != "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c":
        raise ValueError("compiled Direct Frame body differs from saved actor contract")
    task_id = task.ELBOW_SQUAT_HEADING_OBS_TASK_ID
    cfg = load_env_cfg(task_id, play=True)
    retarget(cfg, body, mujoco.MjModel.from_xml_path(str(body)), "squat", variance=False)
    expose_squat_blend(cfg)
    mask_squat_action_history(cfg)
    configure_frame50_motors(cfg, body, rail_v=args.rail_v, delay_s=args.delay_ms / 1000)
    clip_frame50_shoulder_targets(cfg, body, margin_rad=0.05)
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        task.ContactSensorCfg(
            name="handoff_self_contact",
            primary=task.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=task.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"), reduce="none", num_slots=1,
        ),
        task.ContactSensorCfg(
            name="handoff_body_ground_contact",
            primary=task.ContactMatch(mode="geom", pattern=(
                r"^pelvis_frame_budget$", r"^torso_frame_compute_budget$",
                r"^(left|right)_(thigh|shin|upper_arm|forearm)_frame_budget$",
            ), entity="robot"),
            secondary=task.ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="none", num_slots=1,
        ),
    )
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.auto_reset = False
    cfg.episode_length_s = 24.5
    cfg.commands["twist"].resampling_time_range = (100.0, 100.0)
    reset = cfg.events["set_ground_state"].params
    reset["standing_prob"], reset["sitting_prob"] = 1.0, 0.0
    if args.video:
        cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
        cfg.viewer.lookat = (0.0, 0.0, 0.22)
        cfg.viewer.distance = 0.7
        cfg.viewer.azimuth = 90.0
        cfg.viewer.elevation = -12.0
        cfg.viewer.width = 640
        cfg.viewer.height = 480
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device=args.device,
                          render_mode="rgb_array" if args.video else None),
        clip_actions=load_rl_cfg(task_id).clip_actions,
    )
    try:
        robot = env.unwrapped.scene["robot"]
        if args.video:
            reveal_frame50_proxy_geometry_for_video(env.unwrapped.sim.mj_model)
        action_term = env.unwrapped.action_manager.get_term("joint_pos")
        names = list(action_term.target_names)
        expected_legs = [f"{side}_{joint}" for side in ("left", "right")
                         for joint in task.LEG_JOINTS]
        if names[:10] != expected_legs or len(names) != 21:
            raise ValueError("gait/posture joint ordering differs")
        shoulders = [names.index(f"{side}_shoulder_pitch") for side in ("left", "right")]
        router = PortfolioRouter(names, action_term.scale[0].cpu().numpy())
        squat_policy = load_student(args.squat_actor, args.device, names,
                                    action_term.scale, action_term.offset, "squat")
        arm_policy = load_student(args.arm_actor, args.device, names,
                                  action_term.scale, action_term.offset, "both_arms")
        gait_task = task.ELBOW_TASK_ID
        gait_cfg = load_env_cfg(gait_task, play=True)
        if (not gait_cfg.actions["joint_pos"].use_default_offset
                or not cfg.actions["joint_pos"].use_default_offset
                or not torch.all(action_term.scale[:, :10] > 0)):
            raise ValueError("physical target conversion has changed offsets or scale")
        gait_runner = load_runner_cls(gait_task)(
            env, asdict(load_rl_cfg(gait_task)), device=args.device)
        gait_runner.load(str(args.gait_actor), load_cfg={"actor": True},
                         strict=True, map_location=args.device)
        gait_policy = gait_runner.get_inference_policy(device=args.device)
        stop_policy = None
        if args.stop_actor:
            stop_runner = load_runner_cls(gait_task)(
                env, asdict(load_rl_cfg(gait_task)), device=args.device)
            stop_runner.load(str(args.stop_actor), load_cfg={"actor": True},
                             strict=True, map_location=args.device)
            stop_policy = stop_runner.get_inference_policy(device=args.device)
        command = env.unwrapped.command_manager.get_term("twist")
        dt = float(env.unwrapped.step_dt)
        if abs(dt - 0.02) > 1e-8:
            raise ValueError(f"expected 50 Hz, got {dt}")
        site_ids = [robot.site_names.index(f"{side}_hand_center")
                    for side in ("left", "right")]
        sensors = env.unwrapped.scene.sensors
        leg_rated = torch.tensor([a.effort_limit for a in
                                  cfg.scene.entities["robot"].articulation.actuators[:10]],
                                 device=args.device)
        shoulder_rated = [float(cfg.scene.entities["robot"].articulation.actuators[i].effort_limit)
                          for i in shoulders]
        initial_qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy().copy()
        initial_qvel = env.unwrapped.sim.data.qvel[0].cpu().numpy().copy()
        trace = {key: [] for key in ("pre_qpos", "pre_qvel", "post_qpos", "post_qvel",
                                      "actor_obs", "arm_actor_obs", "squat_action",
                                      "arm_action", "gait_action", "applied_action",
                                      "processed_target")}
        rows, frames = [], []
        ended_early = False
        for step in range(args.max_steps):
            t = step * dt
            walking = t < 2.0
            stopping = stop_policy is not None and 2.0 <= t < 4.0
            squat = 6.0 <= t < 10.0
            arm_flag = float(16.0 <= t < 20.0)
            requested = LowLevelCommand(
                mode="walk" if walking else "stop" if stopping else "posture" if t < 16 else "arms",
                forward_m_s=args.walk_velocity if walking else 0.0,
                yaw_rad_s=args.yaw_rate if walking else 0.0,
                squat=squat, raise_arms=bool(arm_flag),
            )
            command.vel_command_b[:, 0] = (args.walk_velocity if walking else float(squat))
            command.vel_command_b[:, 2] = args.yaw_rate if walking else 0.0
            with torch.inference_mode():
                obs = env.get_observations()
                if obs["actor"].shape[1] != 72:
                    raise ValueError("actor observation width changed")
                arm_obs = torch.as_tensor(router.arm_observation(obs["actor"][0].cpu().numpy()),
                                          device=args.device).unsqueeze(0)
                pre_qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy().copy()
                pre_qvel = env.unwrapped.sim.data.qvel[0].cpu().numpy().copy()
                squat_action = squat_policy(obs)
                arm_action = arm_policy({"actor": arm_obs})
                gait_action = torch.zeros_like(squat_action)
                gait_actor_command = None
                if walking or stopping:
                    gait_obs = obs.clone()
                    gait_obs["actor"] = torch.as_tensor(
                        router.gait_observation(obs["actor"][0].cpu().numpy(), requested),
                        device=args.device).unsqueeze(0)
                    gait_actor_command = float(gait_obs["actor"][0, 69])
                    gait_action = (gait_policy if walking else stop_policy)(gait_obs)
                filtered = torch.as_tensor(router.route(
                    requested, squat_action[0].cpu().numpy(), arm_action[0].cpu().numpy(),
                    gait_action[0].cpu().numpy() if walking or stopping else None,
                ), device=args.device).unsqueeze(0)
                _, _, done, _ = env.step(filtered)
            for key, value in (
                ("pre_qpos", pre_qpos), ("pre_qvel", pre_qvel),
                ("post_qpos", env.unwrapped.sim.data.qpos[0].cpu().numpy()),
                ("post_qvel", env.unwrapped.sim.data.qvel[0].cpu().numpy()),
                ("actor_obs", obs["actor"][0].cpu().numpy()),
                ("arm_actor_obs", arm_obs[0].cpu().numpy()),
                ("squat_action", squat_action[0].cpu().numpy()),
                ("arm_action", arm_action[0].cpu().numpy()),
                ("gait_action", gait_action[0].cpu().numpy()),
                ("applied_action", filtered[0].cpu().numpy()),
                ("processed_target", action_term._processed_actions[0].cpu().numpy()),
            ):
                trace[key].append(np.asarray(value).copy())
            if args.video and step % 2 == 0:
                frames.append(env.unwrapped.render())
            hands = robot.data.site_pos_w[0, site_ids, 2].cpu().numpy().copy()
            quat = robot.data.root_link_quat_w[0]
            upright = float(torch.clamp(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2), -1.0, 1.0))
            leg_force = robot.data.actuator_force[0, :10].abs()
            rows.append({
                "t": round(t + dt, 3),
                "phase": "walk" if walking else "stop" if stopping
                         else "posture" if t < 16 else "arms",
                "squat_requested": squat, "arm_requested": bool(arm_flag),
                "pelvis_z": float(robot.data.root_link_pos_w[0, 2]),
                "pelvis_x": float(robot.data.root_link_pos_w[0, 0]),
                "pelvis_y": float(robot.data.root_link_pos_w[0, 1]),
                "hand_z": hands.tolist(),
                "tilt_deg": math.degrees(math.acos(upright)),
                "feet_both": bool(sensors["feet_ground_contact"].data.found[0].all()),
                "hand_force_n": float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).max()),
                "body_ground": bool(sensors["handoff_body_ground_contact"].data.found[0].any()),
                "self_contact": bool(sensors["handoff_self_contact"].data.found[0].any()),
                "leg_duty": bool((leg_force >= leg_rated * 0.995).any()),
                "shoulder_duty": any(
                    abs(float(robot.data.actuator_force[0, i])) >= shoulder_rated[j] * 0.995
                    for j, i in enumerate(shoulders)
                ),
                "shoulder_force_nm": [float(robot.data.actuator_force[0, i]) for i in shoulders],
                "actor_command": float(obs["actor"][0, 69]),
                "gait_actor_command": gait_actor_command,
                "arm_actor_command": float(arm_obs[0, 69]),
            })
            if bool(done[0]):
                ended_early = True
                break
        def phase(lo, hi):
            subset = [r for r in rows if lo <= r["t"] < hi]
            return {
                "frames": len(subset),
                "pelvis_z": float(np.mean([r["pelvis_z"] for r in subset])),
                "hand_z": np.mean([r["hand_z"] for r in subset], axis=0).tolist(),
                "feet_both_fraction": float(np.mean([r["feet_both"] for r in subset])),
                "body_ground_fraction": float(np.mean([r["body_ground"] for r in subset])),
                "leg_duty_fraction": float(np.mean([r["leg_duty"] for r in subset])),
                "shoulder_duty_fraction": float(np.mean([r["shoulder_duty"] for r in subset])),
                "max_hand_force_n": max(r["hand_force_n"] for r in subset),
                "max_tilt_deg": max(r["tilt_deg"] for r in subset),
            }
        windows = {name: phase(*span) for name, span in {
            "walk_end": (1, 2), "settled": (3, 4),
            "stand_before": (5, 6), "squat_hold": (9, 10),
            "stand_after": (13, 14), "arm_before": (15, 16),
            "arm_raised": (19, 20), "arm_lowered": (23, 24),
        }.items() if rows[-1]["t"] >= span[1]}
        squat_pass = (all(k in windows for k in ("stand_before", "squat_hold", "stand_after"))
                      and windows["stand_before"]["pelvis_z"] - windows["squat_hold"]["pelvis_z"] >= 0.04
                      and abs(windows["stand_before"]["pelvis_z"] - windows["stand_after"]["pelvis_z"]) <= 0.015)
        arm_pass = (all(k in windows for k in ("arm_before", "arm_raised", "arm_lowered"))
                    and min(np.subtract(windows["arm_raised"]["hand_z"], windows["arm_before"]["hand_z"])) >= 0.15
                    and max(abs(x) for x in np.subtract(windows["arm_lowered"]["hand_z"],
                                                      windows["arm_before"]["hand_z"])) <= 0.04)
        no_contact = all(r["hand_force_n"] < 1 and not r["body_ground"]
                         and not r["self_contact"] for r in rows)
        support = all(w["feet_both_fraction"] >= 0.95 for name, w in windows.items()
                      if name not in ("walk_end",))
        rated_duty = all(w["leg_duty_fraction"] <= 0.1 and
                         w["shoulder_duty_fraction"] <= 0.1
                         for name, w in windows.items() if name != "walk_end")
        continuity_gap = max(float(np.max(abs(trace["post_qpos"][i-1] - trace["pre_qpos"][i])))
                             for i in (100, 200, 800)) if len(rows) > 800 else None
        q = initial_qpos[3:7]
        spawn_yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]),
                               1 - 2 * (q[2] ** 2 + q[3] ** 2))
        walk_distance = (float(np.dot(
            [rows[99]["pelvis_x"] - rows[0]["pelvis_x"],
             rows[99]["pelvis_y"] - rows[0]["pelvis_y"]],
            [math.cos(spawn_yaw), math.sin(spawn_yaw)]))
            if len(rows) > 100 else 0.0)
        stop_tail_xy = (float(np.hypot(
            rows[199]["pelvis_x"] - rows[149]["pelvis_x"],
            rows[199]["pelvis_y"] - rows[149]["pelvis_y"]))
            if len(rows) > 200 else None)
        report = {
            "schema": "robot-learning-lab.modular-low-level-portfolio.v1",
            "claim_limit": "source-labelled composition diagnostic; CPU run is not CUDA score or learned combined actor",
            "body": str(body), "body_sha256": body_sha,
            "checkpoint_sha256": {"squat": hashlib.sha256(args.squat_actor.read_bytes()).hexdigest(),
                                  "arms": hashlib.sha256(args.arm_actor.read_bytes()).hexdigest(),
                                  "gait": hashlib.sha256(args.gait_actor.read_bytes()).hexdigest(),
                                  "stop": (hashlib.sha256(args.stop_actor.read_bytes()).hexdigest()
                                           if args.stop_actor else None)},
            "device": args.device, "seed": args.seed, "rail_v": args.rail_v,
            "delay_ms": args.delay_ms, "dt_s": dt, "steps": len(rows),
            "walk_velocity_m_s": args.walk_velocity, "walk_distance_m": walk_distance,
            "walk_yaw_rate_rad_s": args.yaw_rate,
            "router_sha256": hashlib.sha256(Path(__file__).with_name("controller.py").read_bytes()).hexdigest(),
            "stop_tail_xy_m": stop_tail_xy,
            "spawn_yaw_rad": spawn_yaw,
            "ended_early": ended_early, "initial_qpos": initial_qpos.tolist(),
            "initial_qvel": initial_qvel.tolist(), "switch_qpos_gap": continuity_gap,
            "shoulder_action_indices": shoulders, "windows": windows,
            "squat_depth_return_pass": bool(squat_pass),
            "arm_rise_return_pass": bool(arm_pass),
            "full_continuity_gate_pass": bool(not ended_early and len(rows) == 1200
                                             and walk_distance >= 0.06
                                             and stop_tail_xy is not None and stop_tail_xy <= 0.03
                                             and squat_pass and arm_pass and no_contact and support
                                             and rated_duty
                                             and continuity_gap is not None and continuity_gap < 1e-9),
            "no_hand_or_body_contact": no_contact,
            "support_windows_pass": support,
            "rated_duty_windows_pass": rated_duty,
            "timeseries": rows,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.trace, initial_qpos=initial_qpos, initial_qvel=initial_qvel,
                                target_names=np.asarray(names), dt_s=np.asarray(dt),
                                **{k: np.asarray(v) for k, v in trace.items()})
        if args.video:
            import imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(args.video, frames, fps=25, macro_block_size=16)
        print(json.dumps({k: v for k, v in report.items() if k != "timeseries"}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
