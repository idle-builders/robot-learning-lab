"""Probe a phase-aware COM correction on selected native MJLab DC motors.

Retain the full per-leg actuator effort for the foot-release transition.
This feedback controller is an authored diagnostic, not a learned policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

from ape_walk import task as ape
from ape_walk.physics.frame50 import (clip_frame50_leg_targets,
                                      configure_frame50_motors,
                                      reveal_frame50_proxy_geometry_for_video)


def smooth(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--body-xml", type=Path, required=True)
    p.add_argument("--design-pointer", type=Path, required=True)
    p.add_argument("--stance", choices=("left", "right"), required=True)
    p.add_argument("--shift-roll", type=float, default=0.24)
    p.add_argument("--clear-swing-arm-roll", type=float, default=0.0,
                   help="Outward shoulder-roll magnitude ramped before the main shift")
    p.add_argument("--shift-s", type=float, default=2.0)
    p.add_argument("--dwell-s", type=float, default=1.0)
    p.add_argument("--lift-s", type=float, default=0.0,
                   help="Enable a swing-foot lift and return after the shift dwell")
    p.add_argument("--hold-s", type=float, default=2.0)
    p.add_argument("--lower-s", type=float, default=2.0)
    p.add_argument("--center-s", type=float, default=2.0)
    p.add_argument("--swing-hip-pitch", type=float, default=-0.7)
    p.add_argument("--swing-knee", type=float, default=1.2)
    p.add_argument("--swing-ankle-pitch", type=float, default=-0.329)
    p.add_argument("--swing-hip-roll-extra", type=float, default=0.0,
                   help="Additional outward swing hip roll during the lift phase")
    p.add_argument("--stance-hip-roll-delta", type=float, default=0.0,
                   help="Signed-outward stance hip-roll change during lift; negative reduces the prelift shift")
    p.add_argument("--stance-com-feedback-gain", type=float, default=0.0,
                   help="Stance hip-roll correction in rad per metre of lateral COM error")
    p.add_argument("--stance-com-feedback-limit", type=float, default=0.18)
    p.add_argument("--stance-com-target-offset", type=float, default=0.016)
    p.add_argument("--rail-v", type=float, default=12.0)
    p.add_argument("--hip-roll-effort-scale", type=float, default=1.0,
                   help="Hypothetical hip-roll rated/stall envelope multiplier")
    p.add_argument("--slide-friction", type=float, default=1.0)
    p.add_argument("--delay-ms", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=7144)
    p.add_argument("--reset-jitter", choices=("none", "tiny"), default="none")
    p.add_argument("--video", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.shift_s <= 0 or args.dwell_s <= 0 or args.lift_s < 0:
        p.error("Shift and dwell must be positive; lift must be nonnegative")
    if args.lift_s and min(args.hold_s, args.lower_s, args.center_s) <= 0:
        p.error("Enabled lift requires positive hold, lower, and center durations")
    if args.stance_com_feedback_gain < 0 or args.stance_com_feedback_limit <= 0:
        p.error("COM feedback gain must be nonnegative and its limit positive")
    if args.hip_roll_effort_scale <= 0:
        p.error("Hip-roll effort scale must be positive")
    total_s = args.shift_s + args.dwell_s
    if args.lift_s:
        total_s += args.lift_s + args.hold_s + args.lower_s + args.center_s
    pointer = json.loads(args.design_pointer.read_text())
    if pointer["id"] != "cove-50-direct-frame":
        raise ValueError("Selected design pointer changed")
    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    names = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    if model.nu != 21 or model.nq != 28 or "head_pitch" in names:
        raise ValueError("Expected the selected 21-axis Cove body")
    if any(abs(float(model.qpos0[model.joint(name).qposadr[0]])) > 1e-8 for name in names):
        raise ValueError("Shift probe expects selected qpos0 joint home")

    cfg = load_env_cfg(ape.ELBOW_TASK_ID, play=True)
    configure_frame50_motors(cfg, args.body_xml,
                             rail_v=args.rail_v, delay_s=args.delay_ms / 1000.0)
    if args.hip_roll_effort_scale != 1.0:
        articulation = cfg.scene.entities["robot"].articulation
        articulation.actuators = tuple(
            replace(act, effort_limit=act.effort_limit * args.hip_roll_effort_scale,
                    saturation_effort=act.saturation_effort * args.hip_roll_effort_scale)
            if names[i].endswith("_hip_roll") else act
            for i, act in enumerate(articulation.actuators))
    clip_frame50_leg_targets(cfg, args.body_xml, margin_rad=0.02)
    cfg.scene.entities["robot"].collisions[0].friction = (args.slide_friction,)
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, float(model.qpos0[2]))
    cfg.scene.entities["robot"].init_state.joint_pos = {r".*": 0.0}
    cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
    swing_side = "left" if args.stance == "right" else "right"
    if args.clear_swing_arm_roll:
        cfg.actions["joint_pos"].scale[f"^{swing_side}_shoulder_roll$"] = 0.3
    cfg.commands["twist"].ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands["twist"].rel_standing_envs = 1.0
    jitter = args.reset_jitter == "tiny"
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (-0.001, 0.001) if jitter else (0.0, 0.0),
        "y": (-0.001, 0.001) if jitter else (0.0, 0.0),
        "z": (0.0, 0.001) if jitter else (0.0, 0.0),
        "roll": (-0.005, 0.005) if jitter else (0.0, 0.0),
        "pitch": (-0.005, 0.005) if jitter else (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.events["reset_robot_joints"].params.update({
        "position_range": (-0.005, 0.005) if jitter else (0.0, 0.0),
        "velocity_range": (-0.01, 0.01) if jitter else (0.0, 0.0),
    })
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
                mode="geom", pattern=r"^(?!(?:left_sole|right_sole)$).*", entity="robot"),
            secondary=ape.ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
        ),
    )
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.auto_reset = False
    cfg.episode_length_s = total_s + 0.5
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
    robot = env.unwrapped.scene["robot"]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    if list(action_term.target_names) != names:
        raise ValueError("Native action order differs from selected model")
    sensors = env.unwrapped.scene.sensors
    site_ids = [robot.site_names.index(f"{side}_sole_center") for side in ("left", "right")]
    stance_slot = 0 if args.stance == "left" else 1
    start_stance = robot.data.site_pos_w[0, site_ids[stance_slot], :2].cpu().numpy().copy()
    sign = -1.0 if args.stance == "left" else 1.0
    hip_roll_ids = [names.index(f"{side}_hip_roll") for side in ("left", "right")]
    stance_roll_id = names.index(f"{args.stance}_hip_roll")
    arm_roll_id = names.index(f"{swing_side}_shoulder_roll")
    swing_roll_id = names.index(f"{swing_side}_hip_roll")
    swing_pitch_id = names.index(f"{swing_side}_hip_pitch")
    swing_knee_id = names.index(f"{swing_side}_knee")
    swing_ankle_id = names.index(f"{swing_side}_ankle_pitch")
    rated = torch.tensor([act.effort_limit for act in
                          cfg.scene.entities["robot"].articulation.actuators[:10]],
                         device="cuda:0")
    first_state_hash = hashlib.sha256()
    for tensor in (env.unwrapped.sim.data.qpos, env.unwrapped.sim.data.qvel):
        first_state_hash.update(tensor.cpu().numpy().tobytes())
    audit = mujoco.MjData(model)
    root_id = model.body("pelvis").id
    stance_site_id = model.site(f"{args.stance}_sole_center").id
    rows, frames, states, velocities, targets = [], [], [], [], []
    audited_pairs = Counter()
    dt = env.unwrapped.step_dt
    for step in range(round(total_s / dt)):
        t = step * dt
        dwell_end = args.shift_s + args.dwell_s
        lift_end = dwell_end + args.lift_s
        hold_end = lift_end + args.hold_s
        lower_end = hold_end + args.lower_s
        if t < args.shift_s:
            phase, roll_fraction, lift_fraction = "shift", smooth(t / args.shift_s), 0.0
        elif t < dwell_end or not args.lift_s:
            phase, roll_fraction, lift_fraction = "dwell", 1.0, 0.0
        elif t < lift_end:
            phase, roll_fraction, lift_fraction = "lift", 1.0, smooth((t - dwell_end) / args.lift_s)
        elif t < hold_end:
            phase, roll_fraction, lift_fraction = "hold", 1.0, 1.0
        elif t < lower_end:
            phase, roll_fraction, lift_fraction = "lower", 1.0, 1.0 - smooth((t - hold_end) / args.lower_s)
        else:
            phase = "center"
            roll_fraction, lift_fraction = 1.0 - smooth((t - lower_end) / args.center_s), 0.0
        target = torch.zeros_like(action_term.offset)
        target[:, hip_roll_ids] = sign * args.shift_roll * roll_fraction
        if args.clear_swing_arm_roll:
            target[:, arm_roll_id] = (sign * args.clear_swing_arm_roll *
                                      (roll_fraction if phase == "center"
                                       else smooth(t / 0.75)))
        if args.lift_s:
            target[:, stance_roll_id] += sign * args.stance_hip_roll_delta * lift_fraction
            target[:, swing_roll_id] += sign * args.swing_hip_roll_extra * lift_fraction
            target[:, swing_pitch_id] = args.swing_hip_pitch * lift_fraction
            target[:, swing_knee_id] = args.swing_knee * lift_fraction
            target[:, swing_ankle_id] = args.swing_ankle_pitch * lift_fraction
        stance_roll_correction = 0.0
        if args.stance_com_feedback_gain and phase in ("lift", "hold", "lower"):
            com_y = float(env.unwrapped.sim.data.subtree_com[0, robot.indexing.root_body_id, 1])
            sole_y = float(robot.data.site_pos_w[0, site_ids[stance_slot], 1])
            target_y = -sign * args.stance_com_target_offset
            stance_roll_correction = float(np.clip(
                args.stance_com_feedback_gain * (com_y - sole_y - target_y),
                -args.stance_com_feedback_limit, args.stance_com_feedback_limit))
            target[:, stance_roll_id] += stance_roll_correction
        action = torch.where(action_term.scale.abs() > 1e-8,
                             (target - action_term.offset) / action_term.scale,
                             torch.zeros_like(target))
        with torch.inference_mode():
            _, _, done, _ = env.step(action)
        native_qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy().copy()
        native_qvel = env.unwrapped.sim.data.qvel[0].cpu().numpy().copy()
        audit.qpos[:] = native_qpos
        audit.qvel[:] = native_qvel
        mujoco.mj_forward(model, audit)
        com_local = (audit.site_xmat[stance_site_id].reshape(3, 3).T @
                     (audit.subtree_com[root_id] - audit.site_xpos[stance_site_id]))[:2]
        # Support is judged in the horizontal ground plane. Rotating the
        # COM's 0.23 m height through the tilted sole frame adds a spurious
        # lateral offset of about 5 cm at this shift angle.
        com_world_xy = (audit.subtree_com[root_id] - audit.site_xpos[stance_site_id])[:2]
        for contact in audit.contact[:audit.ncon]:
            pair = tuple(sorted((model.geom(contact.geom1).name,
                                 model.geom(contact.geom2).name)))
            if "floor" not in pair:
                audited_pairs[pair] += 1
        soles = robot.data.site_pos_w[0, site_ids].cpu().numpy()
        foot_forces = sensors["feet_ground_contact"].data.force[0].norm(dim=-1).cpu().numpy().reshape(-1)
        load_fraction = (float(foot_forces[stance_slot] / foot_forces.sum())
                         if foot_forces.sum() > 1.0 else 0.0)
        quat = robot.data.root_link_pose_w[0, 3:7].cpu().numpy()
        tilt = math.degrees(math.acos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1)))
        effort = robot.data.actuator_force[0, :10].abs()
        rows.append({
            "time_s": round((step + 1) * dt, 3), "phase": phase,
            "pelvis_z_m": round(float(robot.data.root_link_pos_w[0, 2]), 5),
            "tilt_deg": round(tilt, 3),
            "feet_grounded": [bool(v) for v in sensors["feet_ground_contact"].data.found[0].cpu().tolist()],
            "swing_sole_z_m": round(float(soles[1 - stance_slot, 2]), 5),
            "foot_force_n": np.round(foot_forces, 3).tolist(),
            "stance_foot_load_fraction": round(load_fraction, 4),
            "stance_roll_feedback_rad": round(stance_roll_correction, 5),
            "stance_slip_m": round(float(np.linalg.norm(soles[stance_slot, :2] - start_stance)), 5),
            "com_minus_stance_world_xy_m": np.round(com_world_xy, 5).tolist(),
            "com_in_stance_sole_xy_m": np.round(com_local, 5).tolist(),
            "rated_leg_joints": [names[i] for i in range(10) if bool(effort[i] >= .995 * rated[i])],
            "leg_effort_nm": np.round(effort.cpu().numpy(), 4).tolist(),
            "nonsole_floor_contact": bool(sensors["nonsole_ground_contact"].data.found[0].any()),
            "self_contact": bool(sensors["self_body_contact"].data.found[0].any()),
            "hand_force_n": round(float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).max()), 3),
        })
        states.append(native_qpos)
        velocities.append(native_qvel)
        targets.append(target[0].cpu().numpy().copy())
        if args.video and step % 2 == 0:
            frames.append(env.unwrapped.render())
        if bool(done[0]):
            break
    if not rows:
        raise RuntimeError("No native steps executed")
    dwell_rows = [row for row in rows if row["phase"] == "dwell" and
                  row["time_s"] >= args.shift_s + args.dwell_s - 0.4 - 1e-6]
    prelift_rows = [row for row in rows if row["phase"] in ("shift", "dwell")]
    dwell_mean_load = (sum(row["stance_foot_load_fraction"] for row in dwell_rows) /
                       len(dwell_rows)) if dwell_rows else None
    rated_fraction = sum(bool(row["rated_leg_joints"]) for row in rows) / len(rows)
    prelift_rated_fraction = (sum(bool(row["rated_leg_joints"]) for row in prelift_rows) /
                              len(prelift_rows)) if prelift_rows else 1.0
    prelift_complete = any(row["time_s"] >= args.shift_s + args.dwell_s - dt / 2
                           for row in prelift_rows)
    complete = rows[-1]["time_s"] >= total_s - dt / 2
    gate = prelift_complete and len(dwell_rows) >= round(0.4 / dt) - 1 and prelift_rated_fraction <= .10
    gate = gate and all(
        row["feet_grounded"] == [True, True] and row["pelvis_z_m"] >= .20
        and row["tilt_deg"] <= 15.0 and abs(row["com_minus_stance_world_xy_m"][1]) <= .035
        and row["stance_slip_m"] <= .015 and not row["nonsole_floor_contact"]
        and not row["self_contact"] and row["hand_force_n"] < 1.0
        for row in dwell_rows
    )
    gate = gate and not any(row["nonsole_floor_contact"] or row["self_contact"] or
                            row["hand_force_n"] >= 1.0 for row in prelift_rows)
    support_pair = [args.stance == "left", args.stance == "right"]
    longest = current = 0
    for row in rows:
        current = (current + 1 if row["feet_grounded"] == support_pair and
                   row["swing_sole_z_m"] > .025 else 0)
        longest = max(longest, current)
    final = rows[-1]
    final_two_foot_stand = (complete and final["feet_grounded"] == [True, True] and
                            final["pelvis_z_m"] >= .20 and final["tilt_deg"] <= 15.0)
    lift_pass = (args.lift_s > 0 and gate and complete and longest * dt >= 2.0 and
                 final_two_foot_stand and max(row["stance_slip_m"] for row in rows) <= .015 and
                 rated_fraction <= .10 and not any(
                     row["nonsole_floor_contact"] or row["self_contact"] or row["hand_force_n"] >= 1.0
                     for row in rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.out.with_suffix(".npz")
    np.savez_compressed(trace_path, qpos=np.asarray(states), qvel=np.asarray(velocities),
                        joint_target=np.asarray(targets), joint_names=np.asarray(names),
                        dt_s=dt)
    report = {
        "schema": "robot-learning-lab.cove50-native-phase-feedback.v1",
        "method": (("Authored shift/lift/hold/lower/center targets through MJLab DC motors" if args.lift_s else
                    "Authored hip-roll target through MJLab DC motors") +
                   (" with hypothetical scaled hip-roll torque envelope" if args.hip_roll_effort_scale != 1.0
                    else " with selected catalog envelope") + "; not a learned policy"),
        "design_pointer_sha256": sha256(args.design_pointer),
        "body_sha256": sha256(args.body_xml),
        "stance": args.stance, "seed": args.seed, "reset_jitter": args.reset_jitter,
        "rail_v": args.rail_v, "slide_friction": args.slide_friction,
        "hip_roll_effort_scale_hypothetical": args.hip_roll_effort_scale,
        "delay_ms": args.delay_ms, "shift_roll_rad": args.shift_roll,
        "clear_swing_arm_roll_rad": args.clear_swing_arm_roll,
        "shift_s": args.shift_s, "dwell_s": args.dwell_s,
        "lift_s": args.lift_s,
        "hold_s": args.hold_s if args.lift_s else None,
        "lower_s": args.lower_s if args.lift_s else None,
        "center_s": args.center_s if args.lift_s else None,
        "swing_joint_targets_rad": {
            "stance_hip_roll_delta": args.stance_hip_roll_delta,
            "hip_roll_extra": args.swing_hip_roll_extra,
            "hip_pitch": args.swing_hip_pitch, "knee": args.swing_knee,
            "ankle_pitch": args.swing_ankle_pitch,
        } if args.lift_s else None,
        "stance_com_feedback_gain_rad_per_m": args.stance_com_feedback_gain,
        "stance_com_feedback_limit_rad": args.stance_com_feedback_limit,
        "stance_com_target_offset_m": args.stance_com_target_offset,
        "leg_joint_names": names[:10],
        "leg_rated_effort_nm": np.round(rated.cpu().numpy(), 6).tolist(),
        "initial_state_sha256": first_state_hash.hexdigest(),
        "duration_s": rows[-1]["time_s"], "completed_schedule": complete,
        "completed_prelift": prelift_complete,
        "entry_screen_pass": gate,
        "high_load_entry_screen_pass": gate and dwell_mean_load is not None and dwell_mean_load >= .80,
        "longest_single_support_s": round(longest * dt, 3) if args.lift_s else None,
        "final_two_foot_stand": final_two_foot_stand if args.lift_s else None,
        "lift_skill_screen_pass": lift_pass if args.lift_s else None,
        "rated_leg_fraction": round(rated_fraction, 4),
        "prelift_rated_leg_fraction": round(prelift_rated_fraction, 4),
        "max_stance_slip_m": max(row["stance_slip_m"] for row in rows),
        "max_swing_sole_z_m": max(row["swing_sole_z_m"] for row in rows),
        "min_pelvis_z_m": min(row["pelvis_z_m"] for row in rows),
        "final_com_minus_stance_world_xy_m": rows[-1]["com_minus_stance_world_xy_m"],
        "final_com_in_stance_sole_frame_xy_m": rows[-1]["com_in_stance_sole_xy_m"],
        "dwell_two_foot_frames": sum(row["feet_grounded"] == [True, True] for row in dwell_rows),
        "dwell_frames": len(dwell_rows),
        "dwell_mean_stance_foot_load_fraction": round(dwell_mean_load, 4)
            if dwell_mean_load is not None else None,
        "self_contact_frames": sum(row["self_contact"] for row in rows),
        "nonsole_floor_contact_frames": sum(row["nonsole_floor_contact"] for row in rows),
        "hand_contact_frames": sum(row["hand_force_n"] >= 1.0 for row in rows),
        "cpu_audited_self_contact_pairs": [
            {"pair": list(pair), "sampled_frames": count}
            for pair, count in audited_pairs.most_common()
        ],
        "state_trace": str(trace_path), "state_trace_sha256": sha256(trace_path),
        "video": str(args.video) if args.video else None,
        "last": rows[-1], "timeseries": rows,
    }
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, frames, fps=round(1 / (2 * dt)))
        report["video_sha256"] = sha256(args.video)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "timeseries"}, indent=2))


if __name__ == "__main__":
    main()
