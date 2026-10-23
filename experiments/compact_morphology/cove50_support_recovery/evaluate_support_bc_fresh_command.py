"""Native Stage-A teacher or active-only BC actor replay with varied upright reset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from ape_walk import task as ape
from ape_walk.physics.frame50 import (clip_frame50_leg_targets,
                                      configure_frame50_motors,
                                      give_soles_contact_priority,
                                      require_direct_frame,
                                      reveal_frame50_proxy_geometry_for_video)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smooth_phase(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def authored_teacher_target(t: float, stance: str, profile: str,
                            names: list[str], device: str,
                            shift_roll: float, release_amplitude: float) -> torch.Tensor:
    """Two bounded source-labelled feasibility routes, never policy labels unless passed."""
    sign = 1.0 if stance == "right" else -1.0
    swing = "left" if stance == "right" else "right"
    if t < 2.0:
        shift = smooth_phase(t - 1.0)
        release = 0.0
    elif profile == "load_transfer":
        shift = 1.0 if t < 5.0 else 1.0 - smooth_phase((t - 5.0) / 2.0)
        release = 0.0
    elif profile.startswith("staged_"):
        shift = 1.0 if t < 6.0 else 1.0 - smooth_phase((t - 6.0) / 2.0)
        release = (0.0 if t < 3.0 else
                   smooth_phase((t - 3.0) / .5) if t < 3.5 else
                   1.0 if t < 5.5 else
                   1.0 - smooth_phase((t - 5.5) / .5) if t < 6.0 else 0.0)
    elif t < 3.0:
        shift = 1.0
        release = smooth_phase(t - 2.0)
    elif t < 5.0:
        shift = release = 1.0
    elif t < 6.0:
        shift = 1.0
        release = 1.0 - smooth_phase(t - 5.0)
    else:
        shift = 1.0 - smooth_phase((t - 6.0) / 2.0)
        release = 0.0
    target = torch.zeros((1, len(names)), device=device)
    for side in ("left", "right"):
        target[0, names.index(f"{side}_hip_roll")] = sign * shift_roll * shift
    target[0, names.index(f"{swing}_shoulder_roll")] = sign * .20 * shift
    if profile in ("pitch_release", "staged_pitch_release"):
        target[0, names.index(f"{swing}_hip_pitch")] = -release_amplitude * release
    elif profile in ("lateral_release", "staged_lateral_release"):
        target[0, names.index(f"{swing}_hip_roll")] += sign * release_amplitude * release
    elif profile == "load_transfer":
        pass
    else:
        raise ValueError(profile)
    return target


def sole_patch_xy(model: mujoco.MjModel, data: mujoco.MjData,
                  qpos: np.ndarray, geom_id: int) -> np.ndarray:
    """Four world-frame corners of the planted sole's lower face."""
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    sx, sy, sz = model.geom_size[geom_id]
    local = np.asarray([(x, y, -sz) for x in (-sx, sx) for y in (-sy, sy)])
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    return (local @ rotation.T + data.geom_xpos[geom_id])[:, :2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--design-pointer", type=Path, required=True)
    parser.add_argument("--right-pose-json", type=Path, required=True)
    parser.add_argument("--left-pose-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bc-checkpoint", type=Path,
                        help="Active-only observed BC pilot checkpoint; not RSL PPO")
    parser.add_argument("--zero-action", action="store_true")
    parser.add_argument("--authored-teacher", choices=("pitch_release", "lateral_release", "load_transfer",
                                                      "staged_pitch_release", "staged_lateral_release"),
                        help="Feasibility probe only: exact scripted target at the native 50 Hz interface")
    parser.add_argument("--teacher-shift-roll", type=float, default=0.36)
    parser.add_argument("--teacher-release-amplitude", type=float,
                        help="Release radians; defaults to 0.20 lateral or 0.60 pitch")
    parser.add_argument("--stance", choices=("right", "left"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rail-v", type=float, default=12.0)
    parser.add_argument("--grip", type=float, default=1.0)
    parser.add_argument("--contact-model", choices=("repaired", "legacy"),
                        default="repaired",
                        help="Repaired sole-only grip/priority for new scores; legacy only reproduces old setup")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-steps", type=int, default=0,
                        help="Run only a few local setup steps; output cannot be a skill score")
    parser.add_argument("--delay-ms", type=float, default=0.0)
    parser.add_argument("--reset-tilt-noise-deg", type=float, default=0.0)
    parser.add_argument("--reset-joint-noise-std", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    if sum((args.zero_action, bool(args.checkpoint), bool(args.bc_checkpoint),
            bool(args.authored_teacher))) != 1:
        parser.error("Provide exactly one controller")
    if not 0 <= args.reset_tilt_noise_deg <= 2 or not 0 <= args.reset_joint_noise_std <= .02:
        parser.error("Reset disturbance exceeds the bounded pilot interval")
    if args.preflight_steps < 0:
        parser.error("--preflight-steps must be nonnegative")
    if args.teacher_shift_roll < 0 or args.teacher_shift_roll > .6:
        parser.error("Teacher shift roll must be within [0, 0.6] rad")
    if args.teacher_release_amplitude is None:
        args.teacher_release_amplitude = (.60 if args.authored_teacher in
                                          ("pitch_release", "staged_pitch_release") else .20)
    pointer = json.loads(args.design_pointer.read_text())
    if pointer["id"] != "cove-50-direct-frame":
        raise ValueError("Selected design pointer changed")
    model = require_direct_frame(args.body_xml) if args.contact_model == "repaired" else \
        mujoco.MjModel.from_xml_path(str(args.body_xml))
    names = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    if model.nu != 21 or model.nq != 28 or "head_pitch" in names:
        raise ValueError("Expected selected Cove 21-axis action contract")
    specs = {side: json.loads(path.read_text()) for side, path in
             (("right", args.right_pose_json), ("left", args.left_pose_json))}
    pose = {side: [specs[side]["joints"][name] for name in names]
            for side in ("right", "left")}
    z = specs["right"]["root_xyz_roll_pitch"][2]
    roll = specs["right"]["root_xyz_roll_pitch"][3]
    pitch = specs["right"]["root_xyz_roll_pitch"][4]
    source = ape.ELBOW_FLAMINGO_CYCLE_ARMS_TASK_ID
    cfg = load_env_cfg(source, play=True)
    configure_frame50_motors(cfg, args.body_xml,
                             rail_v=args.rail_v, delay_s=args.delay_ms / 1000.0)
    cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
    clip_frame50_leg_targets(cfg, args.body_xml, margin_rad=0.02)
    if args.contact_model == "repaired":
        give_soles_contact_priority(cfg)
        cfg.scene.entities["robot"].collisions[0].friction = {
            r"^(left|right)_sole$": (args.grip,)
        }
    else:
        cfg.scene.entities["robot"].collisions[0].friction = (args.grip,)
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, float(model.qpos0[2]))
    cfg.rewards["pose_track"].params.update({"pose_right": pose["right"],
                                              "pose_left": pose["left"]})
    cfg.rewards["gravity_track"].params.update({
        "gravity_right": (math.sin(pitch), -math.sin(roll) * math.cos(pitch),
                          -math.cos(roll) * math.cos(pitch)),
        "gravity_left": (math.sin(pitch), math.sin(roll) * math.cos(pitch),
                         -math.cos(roll) * math.cos(pitch)),
    })
    cfg.rewards["swing_foot_clear"].params["target"] = 0.07
    cfg.rewards["joint_torques_l2"].weight = -0.05
    cfg.events["set_flamingo_cycle_state"].params.update({
        "pose_right": pose["right"], "pose_left": pose["left"],
        "base_roll": roll, "base_pitch": pitch,
        "z_min": z, "z_max": z + .002,
        "standing_z_min": float(model.qpos0[2]),
        "standing_z_max": float(model.qpos0[2]),
        "tilt_noise": math.radians(args.reset_tilt_noise_deg),
        "joint_noise_std": args.reset_joint_noise_std,
        "in_pose_prob": 0.0,
    })
    cfg.commands["twist"].resampling_time_range = (100.0, 100.0)
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
    cfg.episode_length_s = 8.5
    if args.video:
        cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
        cfg.viewer.lookat = (0.0, 0.0, 0.24)
        cfg.viewer.distance = 0.75
        cfg.viewer.azimuth = 45.0
        cfg.viewer.elevation = -14.0
        cfg.viewer.width = 640
        cfg.viewer.height = 480
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device=args.device,
                          render_mode="rgb_array" if args.video else None),
        clip_actions=100.0,
    )
    if args.video:
        reveal_frame50_proxy_geometry_for_video(env.unwrapped.sim.mj_model)
    policy = None
    if args.checkpoint:
        runner = load_runner_cls(source)(env, asdict(load_rl_cfg(source)), device=args.device)
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
    if args.bc_checkpoint:
        from fit_support_load_bc import SupportLoadBc
        fit_report = json.loads(args.bc_checkpoint.with_name("report.json").read_text())
        if fit_report["checkpoint_sha256"] != sha(args.bc_checkpoint) or \
           fit_report["body_sha256"] != sha(args.body_xml):
            raise ValueError("BC checkpoint or body source mismatch")
        saved = torch.load(args.bc_checkpoint, map_location=args.device, weights_only=True)
        if saved["dataset_report_sha256"] != fit_report["source_sha256"]:
            raise ValueError("BC dataset source mismatch")
        bc = SupportLoadBc(len(saved["input_indices"]), saved["hidden_dim"]).to(args.device)
        bc.load_state_dict(saved["state_dict"])
        bc.eval()
        indices = tuple(saved["input_indices"])
        outputs = tuple(saved["action_indices"])
        mean = saved["mean"].to(args.device)
        std = saved["std"].to(args.device)

        def policy(observation):
            x = observation["actor"][:, indices]
            x = ((x - mean) / std).clamp(-10, 10)
            # The teacher's zero joint target is not a zero native action:
            # action offsets hold the crouched arm/leg pose. Keep that fixed
            # teacher baseline and predict only the four varying channels.
            action = torch.where(action_term.scale.abs() > 1e-8,
                                 -action_term.offset / action_term.scale,
                                 torch.zeros_like(action_term.offset)).repeat(len(x), 1)
            action[:, outputs] = bc(x)
            return action
    robot = env.unwrapped.scene["robot"]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    if list(action_term.target_names) != names:
        raise ValueError("Native action target order differs from selected body")
    sensors = env.unwrapped.scene.sensors
    term = env.unwrapped.command_manager.get_term("twist")
    side = 1.0 if args.stance == "right" else -1.0
    term._stance_side[0] = side
    term._alpha[0] = 0.0
    term.vel_command_b[0] = torch.tensor((0.0, 0.0, 0.0), device=args.device)
    term._fresh[0] = False
    site_ids = [robot.site_names.index(f"{side}_sole_center") for side in ("left", "right")]
    stance_slot = 1 if args.stance == "right" else 0
    start_stance = robot.data.site_pos_w[0, site_ids[stance_slot], :2].cpu().numpy().copy()
    rated = torch.tensor([act.effort_limit for act in
                          cfg.scene.entities["robot"].articulation.actuators[:10]],
                         device=args.device)
    arm_rated = torch.tensor([act.effort_limit for act in
                              cfg.scene.entities["robot"].articulation.actuators[10:20]],
                             device=args.device)
    first_state = hashlib.sha256()
    for tensor in (env.unwrapped.sim.data.qpos, env.unwrapped.sim.data.qvel):
        first_state.update(tensor.cpu().numpy().tobytes())
    initial_qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy().copy()
    dt = env.unwrapped.step_dt
    rows, states, velocities, observations, actions, targets, frames = [], [], [], [], [], [], []
    # The displayed time series is rounded for readability. Keep separate
    # full-precision values for the prospective Stage-A physical gate.
    gate_rows = []
    for step in range(args.preflight_steps or round(8.0 / dt)):
        t = step * dt
        flag = 1.0 if 1.0 <= t < (6.0 if args.authored_teacher and
                                args.authored_teacher.startswith("staged_") else 5.0) else 0.0
        term.vel_command_b[0] = torch.tensor((flag, side if flag else 0.0, 0.0),
                                             device=args.device)
        term._stance_side[0] = side
        with torch.inference_mode():
            # This evaluator sets the command directly before inference. The
            # wrapper returns the previous step's cached observation, so read
            # the actor group afresh without advancing any history buffer.
            observation = {"actor": env.unwrapped.observation_manager.compute_group(
                "actor", update_history=False)}
            actor_observation = observation["actor"].clone()
            if args.authored_teacher:
                target = authored_teacher_target(t, args.stance, args.authored_teacher,
                                                 names, args.device,
                                                 args.teacher_shift_roll,
                                                 args.teacher_release_amplitude)
                action = torch.where(action_term.scale.abs() > 1e-8,
                                     (target - action_term.offset) / action_term.scale,
                                     torch.zeros_like(target))
            else:
                action = (policy(observation) if policy is not None else
                          torch.zeros((1, len(names)), device=args.device))
                target = action_term.offset + action_term.scale * action
            _, _, done, _ = env.step(action)
        feet = [bool(x) for x in sensors["feet_ground_contact"].data.found[0].cpu().tolist()]
        soles = robot.data.site_pos_w[0, site_ids].cpu().numpy()
        effort = robot.data.actuator_force[0, :10].abs()
        arm_effort = robot.data.actuator_force[0, 10:20].abs()
        quat = robot.data.root_link_pose_w[0, 3:7].cpu().numpy()
        tilt = math.degrees(math.acos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1)))
        foot_force = sensors["feet_ground_contact"].data.force[0].norm(dim=-1)
        foot_force_sum = float(foot_force.sum())
        load_fraction = (float(foot_force[stance_slot] / foot_force.sum())
                         if foot_force_sum > 1 else 0.0)
        pelvis_z = float(robot.data.root_link_pos_w[0, 2])
        hand_force = float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).max())
        stance_slip = float(np.linalg.norm(soles[stance_slot, :2] - start_stance))
        nonsole = bool(sensors["nonsole_ground_contact"].data.found[0].any())
        self_contact = bool(sensors["self_body_contact"].data.found[0].any())
        gate_rows.append({
            "feet_grounded": feet, "stance_load_fraction": load_fraction,
            "pelvis_z_m": pelvis_z, "tilt_deg": tilt,
            "swing_sole_z_m": float(soles[1 - stance_slot, 2]),
            "stance_slip_m": stance_slip, "hand_force_n": hand_force,
            "nonsole_floor_contact": nonsole, "self_contact": self_contact,
        })
        rows.append({
            "time_s": round((step + 1) * dt, 3),
            "command_flag": flag, "command_side": side if flag else 0.0,
            "command_alpha": round(float(term.alpha[0]), 4),
            "pelvis_z_m": round(pelvis_z, 5),
            "tilt_deg": round(tilt, 3), "feet_grounded": feet,
            "swing_sole_z_m": round(float(soles[1 - stance_slot, 2]), 5),
            "stance_slip_m": round(stance_slip, 5),
            "stance_load_fraction": round(load_fraction, 4),
            "rated_leg_joints": [names[i] for i in range(10)
                                  if bool(effort[i] >= .995 * rated[i])],
            "rated_arm_joints": [names[10 + i] for i in range(10)
                                  if bool(arm_effort[i] >= .995 * arm_rated[i])],
            "nonsole_floor_contact": nonsole,
            "self_contact": self_contact,
            "hand_force_n": round(hand_force, 3),
        })
        states.append(env.unwrapped.sim.data.qpos[0].cpu().numpy().copy())
        velocities.append(env.unwrapped.sim.data.qvel[0].cpu().numpy().copy())
        observations.append(actor_observation[0].cpu().numpy().copy())
        actions.append(action[0].cpu().numpy().copy())
        targets.append(target[0].cpu().numpy().copy())
        if args.video and step % 2 == 0:
            frames.append(env.unwrapped.render())
        if bool(done[0]):
            break
    support_pair = [args.stance == "left", args.stance == "right"]
    longest = current = 0
    for row in gate_rows:
        current = current + 1 if (row["feet_grounded"] == support_pair and
                                  row["swing_sole_z_m"] > .025) else 0
        longest = max(longest, current)
    assembled_model = env.unwrapped.sim.mj_model
    sole_geom = mujoco.mj_name2id(
        assembled_model, mujoco.mjtObj.mjOBJ_GEOM,
        f"robot/{args.stance}_sole",
    )
    if sole_geom < 0 or assembled_model.geom_type[sole_geom] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("Expected a box sole in the assembled support scene")
    patch_data = mujoco.MjData(assembled_model)
    start_patch = sole_patch_xy(assembled_model, patch_data, initial_qpos, sole_geom)
    for row, gate_row, qpos in zip(rows, gate_rows, states, strict=True):
        patch = sole_patch_xy(assembled_model, patch_data, qpos, sole_geom)
        patch_slip = float(np.linalg.norm(patch - start_patch, axis=1).max())
        gate_row["stance_patch_slip_m"] = patch_slip
        row["stance_patch_slip_m"] = round(patch_slip, 5)
    rated_leg_fraction = sum(bool(row["rated_leg_joints"]) for row in rows) / len(rows)
    rated_arm_fraction = sum(bool(row["rated_arm_joints"]) for row in rows) / len(rows)
    no_bad_contact = not any(row["nonsole_floor_contact"] or row["self_contact"] or
                             row["hand_force_n"] >= 1 for row in gate_rows)
    final = rows[-1]
    final_raw = gate_rows[-1]
    complete = len(gate_rows) == round(8.0 / dt)
    final_stand = complete and final_raw["feet_grounded"] == [True, True] and \
        final_raw["pelvis_z_m"] >= .20 and final_raw["tilt_deg"] <= 15
    slip_key = "stance_patch_slip_m" if args.contact_model == "repaired" else "stance_slip_m"
    passed = (complete and final_stand and longest * dt >= 2.0 and
              max(row[slip_key] for row in gate_rows) <= .015 and
              rated_leg_fraction <= .10 and rated_arm_fraction <= .10 and no_bad_contact)
    # Step indices 99..248 are the exact 2.00..4.98 s post-step window at
    # 50 Hz. An incomplete window is a failure even if every observed row passes.
    load_start = round(2.0 / dt) - 1
    load_end = round(5.0 / dt) - 1
    load_sample_count = round(3.0 / dt)
    load_rows = gate_rows[load_start:load_end]
    failure_reasons = []
    if not complete:
        failure_reasons.append("incomplete_400_step_schedule")
    if not final_stand:
        failure_reasons.append("no_final_upright_two_foot_stand")
    if len(load_rows) != load_sample_count:
        failure_reasons.append("incomplete_150_frame_load_window")
    if any(row["feet_grounded"] != [True, True] for row in load_rows):
        failure_reasons.append("lost_double_sole_contact_in_load_window")
    if any(row["stance_load_fraction"] < .75 for row in load_rows):
        failure_reasons.append("requested_foot_load_below_75_percent")
    if any(row["pelvis_z_m"] < .20 or row["tilt_deg"] > 15 for row in load_rows):
        failure_reasons.append("not_upright_in_load_window")
    if max(row["stance_patch_slip_m"] for row in gate_rows) > .015:
        failure_reasons.append("whole_sole_patch_slip_over_15_mm")
    if rated_leg_fraction > .10:
        failure_reasons.append("rated_leg_duty_over_10_percent")
    if rated_arm_fraction > .10:
        failure_reasons.append("rated_arm_duty_over_10_percent")
    if not no_bad_contact:
        failure_reasons.append("forbidden_hand_nonsole_or_self_contact")
    load_transfer_pass = not failure_reasons
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace = args.out.with_suffix(".npz")
    np.savez_compressed(trace, qpos=np.asarray(states), qvel=np.asarray(velocities),
                        actor_observation=np.asarray(observations),
                        action=np.asarray(actions), joint_target=np.asarray(targets),
                        joint_names=np.asarray(names), dt_s=dt)
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, frames, fps=round(1 / (2 * dt)), macro_block_size=1)
    report = {
        "schema": "robot-learning-lab.cove50-support-cycle-native-replay.v3",
        "method": "Fresh source-labelled teacher or saved actor through native MJLab DC motors, from upright two-foot reset; commanded [stand, side] -> [support request, side] -> [stand, zero] with no physics reset",
        "design_pointer_sha256": sha(args.design_pointer),
        "body_sha256": sha(args.body_xml),
        "right_pose_sha256": sha(args.right_pose_json),
        "left_pose_sha256": sha(args.left_pose_json),
        "checkpoint_sha256": sha(args.checkpoint or args.bc_checkpoint)
        if args.checkpoint or args.bc_checkpoint else None,
        "controller": ("zero_action" if args.zero_action else
                       "authored_teacher" if args.authored_teacher else
                       "saved_bc" if args.bc_checkpoint else "saved_ppo"),
        "authored_teacher_profile": args.authored_teacher,
        "teacher_shift_roll_rad": args.teacher_shift_roll if args.authored_teacher else None,
        "teacher_release_amplitude_rad": args.teacher_release_amplitude if args.authored_teacher else None,
        "evaluator_sha256": sha(Path(__file__)),
        "stance": args.stance, "seed": args.seed,
        "reset_tilt_noise_deg": args.reset_tilt_noise_deg,
        "reset_joint_noise_std": args.reset_joint_noise_std,
        "rail_v": args.rail_v, "grip": args.grip, "delay_ms": args.delay_ms,
        "contact_model": args.contact_model,
        "slip_gate_measure": slip_key,
        "runtime_device": args.device,
        "runtime_versions": {"mujoco": mujoco.__version__, "torch": torch.__version__},
        "control_step_s": dt,
        "action_joint_names": names,
        "actor_observation_dim": int(np.asarray(observations).shape[1]),
        "preflight_steps": args.preflight_steps,
        "claim_limit": ("Short setup execution only; not a CUDA skill score."
                        if args.preflight_steps else "Native full-episode replay on the reported runtime."),
        "initial_state_sha256": first_state.hexdigest(),
        "duration_s": final["time_s"], "completed_schedule": complete,
        "longest_single_support_s": round(longest * dt, 3),
        "final_two_foot_stand": final_stand,
        "max_stance_slip_m": max(row["stance_slip_m"] for row in rows),
        "max_stance_patch_slip_m": max(row["stance_patch_slip_m"] for row in rows),
        "rated_leg_fraction": round(rated_leg_fraction, 4),
        "rated_arm_fraction": round(rated_arm_fraction, 4),
        "bad_contact_frames": sum(row["nonsole_floor_contact"] or row["self_contact"] or
                                  row["hand_force_n"] >= 1 for row in rows),
        "full_gate_pass": passed,
        "load_transfer_gate": {
            "criterion": "All 150 full-precision 2–5 s frames: double-foot load >=75% on requested side, upright, <=15 mm patch slip, <=10% rated leg/arm frames, no bad contact, and complete two-foot return",
            "load_window_s": [2.0, 5.0],
            "observed_frames": len(load_rows),
            "required_frames": load_sample_count,
            "min_stance_load_fraction": min((r["stance_load_fraction"] for r in load_rows), default=None),
            "max_stance_patch_slip_m": max(r["stance_patch_slip_m"] for r in gate_rows),
            "max_hand_force_n": max(r["hand_force_n"] for r in gate_rows),
            "gate_precision": "unrounded native measurements; display timeseries is rounded",
            "failure_reasons": failure_reasons,
            "passed": load_transfer_pass,
        },
        "trace": str(trace), "trace_sha256": sha(trace),
        "video": str(args.video) if args.video else None,
        "timeseries": rows,
    }
    if args.video:
        report["video_sha256"] = sha(args.video)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"stance": args.stance, "seed": args.seed,
                      "duration_s": final["time_s"], "support_s": round(longest * dt, 3),
                      "slip_mm": round(max(row[slip_key] for row in rows) * 1000, 2),
                      "rated_leg_fraction": report["rated_leg_fraction"],
                      "pass": passed}), flush=True)


if __name__ == "__main__":
    main()
