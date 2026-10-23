"""Fresh no-reset native score for a reloaded command-conditioned V4 BC actor."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

import ape_walk.task as task
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--student-actor", type=Path)
    parser.add_argument("--teacher-recovery-step", type=int,
                        help="Unscored continuous teacher takeover from a student-visited state")
    parser.add_argument("--gait-actor", type=Path)
    parser.add_argument("--stop-actor", type=Path)
    parser.add_argument("--squat-actor", type=Path)
    parser.add_argument("--arm-actor", type=Path)
    parser.add_argument("--teacher-parity-report", type=Path,
                        help="Opened source report for one unscored native first-step adapter check")
    parser.add_argument("--teacher-parity-trace", type=Path)
    parser.add_argument("--teacher-screen-report", type=Path)
    parser.add_argument("--teacher-bank-episode", type=Path)
    parser.add_argument("--walk-velocity", type=float, default=0.25)
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
    if not 1 <= args.max_steps <= 1200 or not 0.15 <= args.walk_velocity <= 0.4:
        parser.error("expected 1..1200 steps and 0.15..0.4 m/s")
    if args.device != "cpu" and not args.native_evaluation:
        parser.error("CUDA requires an explicit frozen native evaluation")
    parity = any(value is not None for value in (args.teacher_parity_report,
                  args.teacher_parity_trace, args.teacher_screen_report, args.teacher_bank_episode))
    if parity and (any(value is None for value in (args.teacher_parity_report,
                   args.teacher_parity_trace, args.teacher_screen_report, args.teacher_bank_episode))
                   or args.device != "cuda:0" or not args.native_evaluation or args.max_steps != 1):
        parser.error("Parity requires four opened source files and one explicit CUDA native step")
    if not parity and args.student_actor is None:
        parser.error("Shared actor checkpoint required for candidate evaluation")
    recovery_actors = {"gait": args.gait_actor, "stop": args.stop_actor,
                       "squat": args.squat_actor, "arms": args.arm_actor}
    if args.teacher_recovery_step is not None:
        if (not 0 < args.teacher_recovery_step < args.max_steps
                or any(path is None for path in recovery_actors.values())):
            parser.error("Recovery requires an interior switch step and all four exact teachers")
    elif any(path is not None for path in recovery_actors.values()):
        parser.error("Teacher actors are only used for explicit recovery")
    if args.out.exists() or (args.trace and args.trace.exists()) or (args.video and args.video.exists()):
        parser.error("original output already exists")
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from experiments.compact_morphology.cove50_low_level_handoffs.train_v4_shared_bc import SharedActor
    from experiments.compact_morphology.cove50_low_level_handoffs.v4_action_map import arrays as action_arrays
    from experiments.compact_morphology.cove50_policy_transfer.train_posture_arm import (
        expose_squat_blend, mask_squat_action_history, retarget,
    )
    if args.teacher_recovery_step is not None:
        from experiments.compact_morphology.cove50_low_level.multicommand_bc import load_student
        expected_shas = {
            "gait": "0e8c6862108bd880e330fd8c07d19bf012d4a034f8be6a2ef808de190e5f301f",
            "stop": "1d01d490e3754bf9cb14e19588d784cb7bd1c9d6e82ae0b216c3ce06590e27a2",
            "squat": "0b20cc71d765ce6913adb1f70587fcbf59442bcff25b7645acd64a9e643028f4",
            "arms": "d3e718e3632b4e002adf08f16028700e16be18c4392702d975c9c38f24005774",
        }
        if {name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in recovery_actors.items()} != expected_shas:
            raise ValueError("Teacher recovery actor hashes differ")
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
        scale, offset, lower, upper = action_arrays(names)
        native_clip = action_term._clip[0].detach().cpu().numpy()
        if (load_rl_cfg(task_id).clip_actions is not None
                or not np.allclose(action_term.scale[0].cpu().numpy(), scale, atol=1e-7)
                or not np.allclose(action_term.offset[0].cpu().numpy(), offset, atol=1e-7)
                or not np.allclose(native_clip[:, 0], lower, atol=1e-7)
                or not np.allclose(native_clip[:, 1], upper, atol=1e-7)):
            raise ValueError("Live native action scale, offset or clip differs from source-checked bank")
        if parity:
            source_report = json.loads(args.teacher_parity_report.read_text())
            screen = json.loads(args.teacher_screen_report.read_text())
            screen_row = next((item for item in screen["rows"]
                               if item["case"]["seed"] == args.seed), None)
            plan = json.loads((ROOT / "artifacts/ape-low-level-training-20261015/broad-handoff-v4-l4-screen-method.json").read_text())
            if (screen["plan_sha256"] != hashlib.sha256((ROOT / "artifacts/ape-low-level-training-20261015/broad-handoff-v4-l4-screen-method.json").read_bytes()).hexdigest()
                    or screen_row is None or not screen_row["pass"]
                    or screen_row["case"] != next(item for item in plan["screen"] if item["seed"] == args.seed)
                    or screen_row["report_sha256"] != hashlib.sha256(args.teacher_parity_report.read_bytes()).hexdigest()
                    or screen_row["trace_sha256"] != hashlib.sha256(args.teacher_parity_trace.read_bytes()).hexdigest()
                    or any(source_report["checkpoint_sha256"][name] != spec["sha256"]
                           for name, spec in plan["actors"].items())):
                raise ValueError("Opened V4 source report/trace/checkpoints are not screen bound")
            with np.load(args.teacher_parity_trace, allow_pickle=False) as source:
                saved_pre = source["pre_qpos"][0].copy()
                saved_obs = source["actor_obs"][0].copy()
                saved_target = source["processed_target"][0].copy()
                saved_post = source["post_qpos"][0].copy()
                saved_names = source["target_names"].tolist()
            with np.load(args.teacher_bank_episode, allow_pickle=False) as bank_episode:
                saved_feature = bank_episode["features"][0].copy()
                bank_seed = int(bank_episode["seed"])
            if (source_report["seed"] != args.seed or source_report["body_sha256"] != body_sha
                    or source_report["device"] != "cuda:0" or bank_seed != args.seed
                    or source_report["rail_v"] != args.rail_v
                    or source_report["delay_ms"] != args.delay_ms
                    or source_report["walk_velocity_m_s"] != args.walk_velocity
                    or saved_names != names):
                raise ValueError("Opened teacher source differs from parity request")
            command = env.unwrapped.command_manager.get_term("twist")
            command.vel_command_b[:, 0] = args.walk_velocity
            command.vel_command_b[:, 2] = 0.0
            with torch.inference_mode():
                obs = env.get_observations()["actor"][0].cpu().numpy().copy()
                pre = env.unwrapped.sim.data.qpos[0].cpu().numpy().copy()
                live_feature = np.concatenate((obs,
                    np.asarray([args.walk_velocity, 0, 0, 0, 0, 0], dtype=np.float32),
                    np.asarray([1, 0, 0, 0], dtype=np.float32),
                    np.zeros(21, dtype=np.float32)))
                desired = torch.as_tensor(saved_target, device=args.device)[None, :]
                safe_scale = torch.where(action_term.scale.abs() > 1e-8,
                                         action_term.scale, torch.ones_like(action_term.scale))
                raw = torch.where(action_term.scale.abs() > 1e-8,
                                  (desired - action_term.offset) / safe_scale,
                                  torch.zeros_like(desired))
                env.step(raw)
                processed = action_term._processed_actions[0].cpu().numpy().copy()
                post = env.unwrapped.sim.data.qpos[0].cpu().numpy().copy()
            gaps = {"pre_qpos": float(np.max(np.abs(pre - saved_pre))),
                    "actor_obs": float(np.max(np.abs(obs - saved_obs))),
                    "full_103_feature": float(np.max(np.abs(live_feature - saved_feature))),
                    "processed_target": float(np.max(np.abs(processed - saved_target))),
                    "post_qpos": float(np.max(np.abs(post - saved_post)))}
            passed = all(value <= 1e-4 for value in gaps.values())
            parity_report = {
                "schema": "robot-learning-lab.broad-handoff-v4-first-step-adapter-parity.v1",
                "classification": "unscored opened-teacher CUDA first-step adapter check",
                "source_report_sha256": hashlib.sha256(args.teacher_parity_report.read_bytes()).hexdigest(),
                "source_trace_sha256": hashlib.sha256(args.teacher_parity_trace.read_bytes()).hexdigest(),
                "body_sha256": body_sha, "seed": args.seed, "device": args.device,
                "gaps": gaps, "pass": passed,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(parity_report, indent=2) + "\n")
            print(json.dumps(parity_report, indent=2))
            if not passed:
                raise RuntimeError("Teacher adapter first-step parity failed")
            return
        shoulders = [names.index(f"{side}_shoulder_pitch") for side in ("left", "right")]
        checkpoint = torch.load(args.student_actor, map_location=args.device, weights_only=True)
        if (checkpoint["schema"] not in (
                "robot-learning-lab.broad-handoff-v4-l4-shared-bc.v1",
                "robot-learning-lab.broad-handoff-v5-shared-bc.v1")
                or checkpoint["body_sha256"] != body_sha
                or checkpoint["target_names"] != names
                or checkpoint["feature_width"] != 103
                or not cfg.actions["joint_pos"].use_default_offset):
            raise ValueError("Shared actor body, target order or action map differs")
        shared = SharedActor().to(args.device)
        shared.load_state_dict(checkpoint["model"])
        shared.eval()
        if args.teacher_recovery_step is not None:
            squat_policy = load_student(args.squat_actor, args.device, names,
                                        action_term.scale, action_term.offset, "squat")
            arm_policy = load_student(args.arm_actor, args.device, names,
                                      action_term.scale, action_term.offset, "both_arms")
            gait_task = task.ELBOW_TASK_ID
            gait_runner = load_runner_cls(gait_task)(
                env, __import__("dataclasses").asdict(load_rl_cfg(gait_task)), device=args.device)
            gait_runner.load(str(args.gait_actor), load_cfg={"actor": True},
                             strict=True, map_location=args.device)
            gait_policy = gait_runner.get_inference_policy(device=args.device)
            stop_runner = load_runner_cls(gait_task)(
                env, __import__("dataclasses").asdict(load_rl_cfg(gait_task)), device=args.device)
            stop_runner.load(str(args.stop_actor), load_cfg={"actor": True},
                             strict=True, map_location=args.device)
            stop_policy = stop_runner.get_inference_policy(device=args.device)
            last_gait = torch.zeros((1, 21), device=args.device)
            teacher_filtered = None
        input_mean = checkpoint["input_mean"].to(args.device)
        input_std = checkpoint["input_std"].to(args.device)
        target_mean = checkpoint["target_mean"].to(args.device)
        target_std = checkpoint["target_std"].to(args.device)
        target_min = checkpoint["target_min"].to(args.device)
        target_max = checkpoint["target_max"].to(args.device)
        margin = 0.05 * (target_max - target_min).clamp_min(0.02)
        safe_scale = torch.where(action_term.scale.abs() > 1e-8, action_term.scale,
                                 torch.ones_like(action_term.scale))
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
                                      "actor_obs", "shared_features", "predicted_target",
                                      "applied_action", "processed_target")}
        rows, frames = [], []
        max_target_mapping_gap = 0.0
        max_requested_target_clipping_gap = 0.0
        previous_action = torch.zeros((1, 21), device=args.device)
        previous_arm_flag = 0.0
        ended_early = False
        for step in range(args.max_steps):
            t = step * dt
            walking = t < 2.0
            stopping = 2.0 <= t < 4.0
            squat = 6.0 <= t < 10.0
            arm_flag = float(16.0 <= t < 20.0)
            command.vel_command_b[:, 0] = (args.walk_velocity if walking else float(squat))
            command.vel_command_b[:, 2] = 0.0
            with torch.inference_mode():
                obs = env.get_observations()
                if obs["actor"].shape[1] != 72:
                    raise ValueError("actor observation width changed")
                pre_qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy().copy()
                pre_qvel = env.unwrapped.sim.data.qvel[0].cpu().numpy().copy()
                phase = "walk" if walking else "stop" if stopping else "posture" if t < 16 else "arms"
                commands = torch.tensor([[args.walk_velocity if walking else 0.0, 0.0,
                                          float(squat), arm_flag, previous_arm_flag,
                                          float(stopping)]], device=args.device)
                modes = torch.tensor([[float(phase == name) for name in
                                       ("walk", "stop", "posture", "arms")]], device=args.device)
                features = torch.cat((obs["actor"], commands, modes, previous_action), dim=1)
                target = shared((features - input_mean) / input_std) * target_std + target_mean
                if not torch.isfinite(target).all():
                    raise RuntimeError("Student predicted nonfinite physical target")
                target = torch.maximum(target_min - margin, torch.minimum(target_max + margin, target))
                effective_target = torch.clamp(target, min=action_term._clip[:, :, 0],
                                               max=action_term._clip[:, :, 1])
                effective_target = torch.where(action_term.scale.abs() > 1e-8,
                                               effective_target, action_term.offset)
                max_requested_target_clipping_gap = max(max_requested_target_clipping_gap,
                    float(torch.max(torch.abs(target - effective_target))))
                raw = (effective_target - action_term.offset) / safe_scale
                raw = torch.where(action_term.scale.abs() > 1e-8, raw, torch.zeros_like(raw))
                if args.teacher_recovery_step is not None and step >= args.teacher_recovery_step:
                    arm_obs = obs["actor"].clone()
                    arm_obs[:, 69] = previous_arm_flag
                    merged = squat_policy(obs).clamp(-1.0, 1.0).clone()
                    if walking or stopping:
                        gait_obs = obs.clone()
                        gait_obs["actor"][:, 48:69] = last_gait
                        gait_obs["actor"][:, 69:72] = 0.0
                        gait_obs["actor"][:, 69] = args.walk_velocity if walking else 0.0
                        gait_action = (gait_policy if walking else stop_policy)(gait_obs)
                        last_gait = gait_action.clone()
                        merged[:] = 0.0
                        merged[:, :10] = gait_action[:, :10] * 0.3 / action_term.scale[:, :10]
                    if t >= 16.0:
                        arm_action = arm_policy({"actor": arm_obs})
                        merged[:, :10] = 0.0
                        merged[:, shoulders] = arm_action[:, shoulders]
                    if teacher_filtered is None:
                        teacher_filtered = previous_action.clone()
                    teacher_filtered = (merged.clone() if walking or stopping
                                        else 0.2 * merged + 0.8 * teacher_filtered)
                    raw = teacher_filtered
                    effective_target = torch.clamp(raw * action_term.scale + action_term.offset,
                                                   min=action_term._clip[:, :, 0],
                                                   max=action_term._clip[:, :, 1])
                if not torch.isfinite(raw).all():
                    raise RuntimeError("Student produced nonfinite native action")
                _, _, done, _ = env.step(raw)
                max_target_mapping_gap = max(max_target_mapping_gap, float(
                    torch.max(torch.abs(effective_target - action_term._processed_actions))))
                if max_target_mapping_gap > 1e-5 and args.teacher_recovery_step is None:
                    raise RuntimeError("Live processed physical target differs from canonical request")
                previous_action = raw.clone()
                previous_arm_flag = arm_flag
            for key, value in (
                ("pre_qpos", pre_qpos), ("pre_qvel", pre_qvel),
                ("post_qpos", env.unwrapped.sim.data.qpos[0].cpu().numpy()),
                ("post_qvel", env.unwrapped.sim.data.qvel[0].cpu().numpy()),
                ("actor_obs", obs["actor"][0].cpu().numpy()),
                ("shared_features", features[0].cpu().numpy()),
                ("predicted_target", target[0].cpu().numpy()),
                ("applied_action", raw[0].cpu().numpy()),
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
                "phase": phase,
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
                "gait_actor_command": args.walk_velocity if walking else 0.0 if stopping else None,
                "arm_actor_command": float(commands[0, 4]),
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
            "schema": "robot-learning-lab.broad-handoff-shared-bc-evaluation.v2",
            "checkpoint_schema": checkpoint["schema"],
            "classification": ("unscored continuous teacher takeover from learner-visited state"
                               if args.teacher_recovery_step is not None else "fresh learned actor rollout"),
            "teacher_recovery_step": args.teacher_recovery_step,
            "teacher_checkpoint_sha256": ({name: hashlib.sha256(path.read_bytes()).hexdigest()
                                           for name, path in recovery_actors.items()}
                                          if args.teacher_recovery_step is not None else None),
            "claim_limit": "Fresh source-labelled learned BC rollout; no redesigned-body or physical transfer claim",
            "body": str(body), "body_sha256": body_sha,
            "student_checkpoint_sha256": hashlib.sha256(args.student_actor.read_bytes()).hexdigest(),
            "max_target_mapping_gap": max_target_mapping_gap,
            "max_requested_target_clipping_gap": max_requested_target_clipping_gap,
            "device": args.device, "seed": args.seed, "rail_v": args.rail_v,
            "delay_ms": args.delay_ms, "dt_s": dt, "steps": len(rows),
            "walk_velocity_m_s": args.walk_velocity, "walk_distance_m": walk_distance,
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
