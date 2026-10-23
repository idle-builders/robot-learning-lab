"""Replay an authored side-reach lesson in native MJLab frame50 motors.

This checks the actual task reset, torque-speed actuators, contacts and delayed
actions before spending GPU time training an imitation actor. It is not a
learned policy and starts from a prepositioned, attained upright stand.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg

from ape_walk import task as ape  # registers ape tasks before reach import
from ape_walk import reach
from ape_walk.physics.frame50 import configure_frame50_motors, motor_configs
from experiments.compact_morphology.cove50_policy_transfer.train_side_reach_bc import ReachActor
from experiments.compact_morphology.cove50_policy_transfer.train_side_reach_arm_residual import ArmResidualActor


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--case", default="side_lean_repositioned")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-out", type=Path,
                        help="Save pre-action native observations and teacher actions for imitation")
    parser.add_argument("--policy-checkpoint", type=Path,
                        help="Reload a saved BC actor in a fresh process for closed-loop native replay")
    parser.add_argument("--label-feedback-checkpoint", type=Path,
                        help="Label visited states with the frozen live-feedback teacher")
    parser.add_argument("--teacher-execution-prob", type=float, default=0.0,
                        help="Probability of executing the label teacher rather than the BC actor")
    parser.add_argument("--teacher-after-s", type=float, default=-1.0,
                        help="Execute the label teacher from this time onward; negative disables the switch")
    parser.add_argument("--policy-arm-warmup-s", type=float, default=0.0,
                        help="Blend a factorized arm actor in from authored targets after reset")
    parser.add_argument("--xy-correction-reference", type=Path,
                        help="Nominal native qpos trace for small target-relative arm Jacobian corrections")
    parser.add_argument("--toy-step-time-s", type=float, default=-1,
                        help="Move the free toy once during replay; negative disables the step")
    parser.add_argument("--toy-step-x-m", type=float, default=0.0)
    parser.add_argument("--toy-step-y-m", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2144)
    parser.add_argument("--rail-v", type=float, default=12.0)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    parser.add_argument("--start-x-offset-m", type=float, default=0.07)
    parser.add_argument("--start-y-offset-m", type=float, default=0.0)
    parser.add_argument("--toy-x-offset-m", type=float, default=0.0)
    parser.add_argument("--toy-y-offset-m", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--transition-s", type=float, default=4.0)
    parser.add_argument("--right-hip-offset-rad", type=float, default=0.02)
    parser.add_argument("--right-ankle-offset-rad", type=float, default=0.01)
    parser.add_argument("--right-shoulder-pitch-offset-rad", type=float, default=0.06)
    parser.add_argument("--right-gripper-offset-rad", type=float, default=0.06)
    parser.add_argument("--right-shoulder-roll-offset-rad", type=float, default=0.0)
    parser.add_argument("--joint-target-offset", nargs=2, action="append", default=[],
                        metavar=("JOINT", "RAD"))
    args = parser.parse_args()
    if args.hand == "left" and (args.xy_correction_reference or
                                args.label_feedback_checkpoint or args.teacher_execution_prob):
        raise ValueError("Left mirror pilot supports the frozen feedback controller only")
    if not 0 <= args.teacher_execution_prob <= 1:
        raise ValueError("Teacher execution probability must be in [0, 1]")
    if args.teacher_execution_prob and args.teacher_after_s >= 0:
        raise ValueError("Random teacher execution and a timed teacher switch are mutually exclusive")
    if (args.teacher_execution_prob or args.teacher_after_s >= 0) and (
            not args.policy_checkpoint or not args.label_feedback_checkpoint):
        raise ValueError("Mixed execution requires both BC actor and feedback label checkpoint")
    if args.policy_arm_warmup_s < 0:
        raise ValueError("Arm warmup must be nonnegative")
    geometry = json.loads(args.geometry.read_text())
    candidate = next(row for row in geometry["results"] if row["case"] == args.case)
    body = args.body_xml.resolve()
    if mujoco.MjModel.from_xml_path(str(body)).nu != 21:
        raise ValueError("Expected selected 21-axis Cove body")
    cfg = load_env_cfg(reach.TASK_ID, play=True)
    if args.hand == "left":
        for observation_group in ("actor", "critic"):
            cfg.observations[observation_group].terms["ee_to_cube"].params["asset_cfg"].site_names = (
                "left_hand_center",)
        cfg.rewards["lift"].params["asset_cfg"].site_names = ("left_hand_center",)
        for name in ("fingertip_friction_slide", "fingertip_friction_spin",
                     "fingertip_friction_roll"):
            cfg.events[name].params["asset_cfg"].geom_names = r"^(left_hand|left_finger_tip)$"
    robot_cfg = cfg.scene.entities["robot"]
    robot_cfg.init_state.pos = (
        candidate["pelvis_xyz_m"][0] + args.start_x_offset_m,
        (1 if args.hand == "right" else -1) *
        (candidate["pelvis_xyz_m"][1] + args.start_y_offset_m),
        0.2280034,
    )
    pose = robot_cfg.init_state.joint_pos
    pose[r".*hip_pitch.*"] = -0.198
    pose[r".*knee.*"] = 0.409
    pose[r".*ankle_pitch.*"] = -0.2
    action = cfg.actions["joint_pos"]
    action.scale.pop("^head_pitch$", None)
    for side in ("left", "right"):
        for axis in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch"):
            # The authored transition needs up to 1.42 rad knee travel. Keep
            # its raw policy actions inside the native [-1, 1] runner bound.
            action.scale[f"^{side}_{axis}$"] = 1.6 if axis in ("hip_pitch", "knee") else 0.5
    other_hand = "left" if args.hand == "right" else "right"
    action.scale[f"^{other_hand}_shoulder_pitch$"] = 0.5
    action.scale[f"^{args.hand}_shoulder_yaw$"] = 0.7
    action.scale[f"^{args.hand}_shoulder_roll$"] = 0.5
    action.scale[f"^{args.hand}_shoulder_pitch$"] = 1.2
    action.scale[f"^{args.hand}_elbow_pitch$"] = 1.2
    action.scale[f"^{args.hand}_gripper$"] = 0.5
    if args.hand == "left":
        for axis in ("shoulder_yaw", "shoulder_roll", "elbow_pitch", "gripper"):
            action.scale[f"^right_{axis}$"] = 0.0
    gait_obs = ape.env_cfg(elbowed=True).observations["actor"].terms
    actor_terms = cfg.observations["actor"].terms
    actor_terms.pop("cube_to_goal")
    actor_terms["base_ang_vel"] = deepcopy(gait_obs["base_ang_vel"])
    actor_terms["projected_gravity"] = deepcopy(gait_obs["projected_gravity"])
    configure_frame50_motors(cfg, body, rail_v=args.rail_v,
                             delay_s=args.delay_ms / 1000)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.auto_reset = False
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10
    command_cfg = cfg.commands["lift_height"]
    command_cfg.resampling_time_range = (100.0, 100.0)
    toy_x, toy_y, _ = geometry["toy_center_m"]
    toy_x += args.toy_x_offset_m
    toy_y += args.toy_y_offset_m
    if args.hand == "left":
        toy_y *= -1
    command_cfg.object_pose_range.x = (toy_x, toy_x)
    command_cfg.object_pose_range.y = (toy_y, toy_y)
    command_cfg.object_pose_range.z = (0.01, 0.01)
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (
        ContactSensorCfg(
            name=f"{args.hand}_finger_tip_cube_contact",
            primary=ContactMatch(mode="geom", pattern=f"{args.hand}_finger_tip", entity="robot"),
            secondary=ContactMatch(mode="geom", pattern="cube_geom", entity="cube"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
        ),
    )
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    policy = None
    residual_weight = residual_center = residual_reference = None
    residual_bound = None
    feedback_integral = None
    feedback_p = feedback_i = None
    initial_ee_to_cube_xy = env.obs_buf["actor"][0, 42:44].clone()
    feature_mean = feature_std = None
    mask_action_history = False
    policy_joint_names = None
    arm_residual_model = arm_residual_reference = arm_residual_baseline = None
    arm_residual_bound = arm_residual_xy_scale = None
    arm_residual_action_history = False
    arm_residual_error_integral = False
    arm_error_integral = None
    if args.policy_checkpoint:
        saved = torch.load(args.policy_checkpoint, map_location="cpu", weights_only=True)
        nominal_body = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
        if args.hand == "left" and (saved.get("hand") != "left" or
                                     saved.get("selected_body_sha256") != hashlib.sha256(nominal_body.read_bytes()).hexdigest() or
                                     saved.get("selected_pointer_sha256") != hashlib.sha256((ROOT / "robot/ape-current-design.json").read_bytes()).hexdigest()):
            raise ValueError("Left reach requires a checkpoint mirrored for this selected body")
        if saved.get("method") == "bounded_learned_arm_residual_on_authored_path":
            arm_residual_model = ArmResidualActor(saved.get("input_dim", 5)).to("cuda:0")
            arm_residual_model.load_state_dict(saved["state_dict"])
            arm_residual_model.eval()
            arm_residual_reference = saved["reference_xy"].to("cuda:0")
            arm_residual_baseline = saved["baseline_action"].to("cuda:0")
            arm_residual_bound = float(saved["residual_bound"])
            arm_residual_xy_scale = float(saved["xy_scale_m"])
            arm_residual_action_history = bool(saved.get("include_action_history", False))
            arm_residual_error_integral = bool(saved.get("include_error_integral", False))
            if arm_residual_error_integral:
                arm_error_integral = torch.zeros(2, device="cuda:0")
            policy_joint_names = saved["policy_joint_names"]
        elif saved.get("method") == "bounded_goal_residual_on_native_teacher":
            residual_weight = saved["weight"].to("cuda:0")
            residual_center = saved["center_ee_to_cube_xy_m"].to("cuda:0")
            residual_bound = float(saved["residual_action_bound"])
        elif saved.get("method") == "bounded_observed_goal_pi_residual_on_native_teacher":
            residual_weight = saved["weight"].to("cuda:0")
            residual_reference = saved["center_ee_to_cube_xy_trajectory_m"].to("cuda:0")
            residual_bound = float(saved["residual_action_bound"])
            feedback_p = float(saved["proportional_gain"])
            feedback_i = float(saved["integral_gain_per_s"])
            feedback_integral = torch.zeros(
                env.action_manager.get_term("joint_pos").action_dim, device="cuda:0")
        else:
            policy = ReachActor(saved["input_dim"], saved["action_dim"]).to("cuda:0")
            policy.load_state_dict(saved["state_dict"])
            policy.eval()
            feature_mean = saved["feature_mean"].to("cuda:0")
            feature_std = saved["feature_std"].to("cuda:0")
            mask_action_history = bool(saved.get("mask_action_history", False))
            policy_joint_names = saved.get("policy_joint_names")
    label_weight = label_reference = label_integral = None
    label_bound = label_p = label_i = None
    label_rng = np.random.default_rng(args.seed + 9164)
    if args.label_feedback_checkpoint:
        labeled = torch.load(args.label_feedback_checkpoint, map_location="cpu", weights_only=True)
        if labeled.get("method") != "bounded_observed_goal_pi_residual_on_native_teacher":
            raise ValueError("Expected frozen live-feedback residual label teacher")
        label_weight = labeled["weight"].to("cuda:0")
        label_reference = labeled["center_ee_to_cube_xy_trajectory_m"].to("cuda:0")
        label_integral = torch.zeros(env.action_manager.get_term("joint_pos").action_dim,
                                     device="cuda:0")
        label_bound = float(labeled["residual_action_bound"])
        label_p = float(labeled["proportional_gain"])
        label_i = float(labeled["integral_gain_per_s"])
    robot = env.scene["robot"]
    cube = env.scene["cube"]
    action_term = env.action_manager.get_term("joint_pos")
    joint_names = list(robot.joint_names)
    target_names = list(action_term._target_names)
    start = robot.data.default_joint_pos[0].clone()
    desired = start.clone()
    for side_index, side in enumerate(("left", "right")):
        source_side_index = side_index if args.hand == "right" else 1 - side_index
        source_side = side if args.hand == "right" else ("right" if side == "left" else "left")
        mirror_sign = 1 if args.hand == "right" else -1
        desired[joint_names.index(f"{side}_hip_roll")] = (
            mirror_sign * candidate["leg_pose_rad"]["hip_roll_left_right"][source_side_index])
        for axis, value in zip(("hip_pitch", "knee", "ankle_pitch"),
                               candidate["leg_pose_rad"][source_side]):
            desired[joint_names.index(f"{side}_{axis}")] = value
    desired[joint_names.index(f"{other_hand}_shoulder_pitch")] = -1.0
    for name, value in candidate["arm_pose_rad"].items():
        target_name = name if args.hand == "right" else name.replace("right_", "left_", 1)
        sign = -1 if args.hand == "left" and name.endswith(("_yaw", "_roll")) else 1
        desired[joint_names.index(target_name)] = sign * value
    for name, value in (
        (f"{args.hand}_hip_pitch", args.right_hip_offset_rad),
        (f"{args.hand}_ankle_pitch", args.right_ankle_offset_rad),
        (f"{args.hand}_shoulder_pitch", args.right_shoulder_pitch_offset_rad),
        (f"{args.hand}_gripper", args.right_gripper_offset_rad),
        (f"{args.hand}_shoulder_roll", (1 if args.hand == "right" else -1) * args.right_shoulder_roll_offset_rad),
    ):
        desired[joint_names.index(name)] += value
    for name, value in args.joint_target_offset:
        desired[joint_names.index(name)] += float(value)
    arm_xy_correction = {}
    if args.xy_correction_reference:
        reference = np.load(args.xy_correction_reference)
        model = mujoco.MjModel.from_xml_path(str(body))
        data = mujoco.MjData(model)
        data.qpos[:] = reference["qpos"][250, :model.nq]
        mujoco.mj_forward(model, data)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_center")
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac, np.zeros_like(jac), site)
        arm_names = ("right_shoulder_yaw", "right_shoulder_roll",
                     "right_shoulder_pitch", "right_elbow_pitch")
        columns = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                   for name in arm_names]
        relative_shift_xy = np.array([
            args.toy_x_offset_m - (args.start_x_offset_m - 0.07),
            args.toy_y_offset_m - args.start_y_offset_m,
        ])
        correction = np.linalg.pinv(jac[:2, columns]) @ relative_shift_xy
        if np.max(np.abs(correction)) > 0.08:
            raise ValueError("Target-relative arm correction exceeds 0.08 rad bound")
        for name, value in zip(arm_names, correction):
            desired[joint_names.index(name)] += float(value)
            arm_xy_correction[name] = round(float(value), 6)
    rated = torch.tensor([act.effort_limit for act in motor_configs(body, rail_v=args.rail_v)[:10]],
                         device="cuda:0")
    initial_root = robot.data.root_link_pos_w[0].cpu().tolist()
    initial_cube = cube.data.root_link_pos_w[0].cpu().tolist()
    rows = []
    teacher_obs = []
    teacher_action = []
    teacher_qpos = []
    teacher_toy_relative_root = []
    teacher_contact = []
    teacher_phase = []
    feedback_teacher_actions = []
    feedback_teacher_executed = []
    feedback_residual_trace = []
    toy_step_applied = False
    max_abs_action = 0.0
    contact_run = best_contact_run = 0
    dt = env.step_dt
    for step in range(round(args.duration_s / dt)):
        if not toy_step_applied and args.toy_step_time_s >= 0 and step * dt >= args.toy_step_time_s:
            pose = cube.data.root_link_pose_w.clone()
            pose[0, 0] += args.toy_step_x_m
            pose[0, 1] += (1 if args.hand == "right" else -1) * args.toy_step_y_m
            cube.write_root_link_pose_to_sim(pose)
            toy_step_applied = True
        progress = min(1.0, step * dt / args.transition_s)
        blend = progress * progress * (3 - 2 * progress)
        target = start + blend * (desired - start)
        actions = torch.zeros((1, action_term.action_dim), device="cuda:0")
        for i, name in enumerate(target_names):
            scale = float(action_term._scale[0, i])
            if scale != 0:
                actions[0, i] = (target[joint_names.index(name)] - action_term._offset[0, i]) / scale
        authored_actions = actions.clone()
        if arm_residual_model is not None:
            indices = [target_names.index(name) for name in policy_joint_names]
            reference_index = min(step, len(arm_residual_reference) - 1)
            if (authored_actions[0, indices] - arm_residual_baseline[reference_index, indices]).abs().max() > 1e-4:
                raise ValueError("Arm residual checkpoint does not match the authored action path")
            entry_error_xy = (initial_ee_to_cube_xy - arm_residual_reference[0]) / arm_residual_xy_scale
            live_error_xy = (env.obs_buf["actor"][0, 42:44] -
                             arm_residual_reference[reference_index]) / arm_residual_xy_scale
            feature_parts = [entry_error_xy, live_error_xy]
            if arm_residual_error_integral:
                arm_error_integral = arm_error_integral + live_error_xy * dt
                feature_parts.append(arm_error_integral)
            if arm_residual_action_history:
                prior_baseline = (torch.zeros_like(arm_residual_baseline[0, indices]) if step == 0
                                  else arm_residual_baseline[step - 1, indices])
                previous_residual = (env.obs_buf["actor"][0, 45 + np.asarray(indices)] -
                                     prior_baseline) / arm_residual_bound
                feature_parts.append(previous_residual)
            feature_parts.append(torch.tensor([blend], device="cuda:0"))
            features = torch.cat(feature_parts)[None, :]
            with torch.inference_mode():
                correction = arm_residual_model(features)[0] * arm_residual_bound * blend
            actions[0, indices] = (actions[0, indices] + correction).clamp(-1, 1)
        label_actions = None
        if label_weight is not None:
            live_xy = env.obs_buf["actor"][0, 42:44]
            reference_xy = label_reference[min(step, len(label_reference) - 1)]
            label_direction = torch.mv(label_weight, live_xy - reference_xy)
            label_integral = (label_integral + label_i * dt * label_direction).clamp(
                -label_bound, label_bound)
            label_residual = (label_p * label_direction + label_integral).clamp(
                -label_bound, label_bound) * blend
            label_actions = (authored_actions + label_residual[None, :]).clamp(-1, 1)
        if policy is not None:
            obs = env.obs_buf["actor"].clone()
            if mask_action_history:
                obs[:, 45:66] = 0
            features = torch.cat((obs,
                                  torch.tensor([[blend]], device="cuda:0")), dim=1)
            with torch.inference_mode():
                predicted = policy(((features - feature_mean) / feature_std).clamp(-5, 5))
            if policy_joint_names:
                indices = [target_names.index(name) for name in policy_joint_names]
                if predicted.shape[1] != len(indices):
                    raise ValueError("Policy output/joint contract mismatch")
                warmup = (min(1.0, step * dt / args.policy_arm_warmup_s)
                          if args.policy_arm_warmup_s else 1.0)
                actions[:, indices] = actions[:, indices] * (1 - warmup) + predicted * warmup
            else:
                actions = predicted
        if residual_weight is not None:
            if residual_reference is None:
                shifted_goal = (initial_ee_to_cube_xy - residual_center) * blend
                residual = torch.mv(residual_weight, shifted_goal).clamp(-residual_bound, residual_bound)
            else:
                live_xy = env.obs_buf["actor"][0, 42:44]
                reference_xy = residual_reference[min(step, len(residual_reference) - 1)]
                goal_error = live_xy - reference_xy
                correction_direction = torch.mv(residual_weight, goal_error)
                feedback_integral = (feedback_integral + feedback_i * dt * correction_direction).clamp(
                    -residual_bound, residual_bound)
                residual = (feedback_p * correction_direction + feedback_integral).clamp(
                    -residual_bound, residual_bound) * blend
            actions += residual[None, :]
            feedback_residual_trace.append(residual.cpu().numpy().copy())
        use_label = (label_actions is not None and
                     (step * dt >= args.teacher_after_s if args.teacher_after_s >= 0 else
                      args.teacher_execution_prob > 0 and
                      label_rng.random() < args.teacher_execution_prob))
        if use_label:
            actions = label_actions
        max_abs_action = max(max_abs_action, float(actions.abs().max()))
        if max_abs_action > 1.001:
            raise ValueError(f"Teacher raw action {max_abs_action:.3f} exceeds policy bound")
        if args.dataset_out:
            # The unwrapped MJLab environment keeps the observation returned by
            # reset/step in obs_buf; get_observations belongs to the RL wrapper.
            teacher_obs.append(env.obs_buf["actor"][0].cpu().numpy().copy())
            teacher_action.append(actions[0].cpu().numpy().copy())
            teacher_qpos.append(env.sim.data.qpos[0].cpu().numpy().copy())
            teacher_toy_relative_root.append((cube.data.root_link_pos_w[0] -
                                             robot.data.root_link_pos_w[0]).cpu().numpy().copy())
            teacher_phase.append(blend)
            if label_actions is not None:
                feedback_teacher_actions.append(label_actions[0].cpu().numpy().copy())
                feedback_teacher_executed.append(use_label)
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated
        finger_touch = bool(env.scene.sensors[f"{args.hand}_finger_tip_cube_contact"].data.found[0].any())
        if args.dataset_out:
            teacher_contact.append(finger_touch)
        contact_run = contact_run + 1 if finger_touch else 0
        best_contact_run = max(best_contact_run, contact_run)
        root = robot.data.root_link_pos_w[0]
        quat = robot.data.root_link_pose_w[0, 3:7]
        tilt = math.degrees(2 * math.acos(min(1.0, abs(float(quat[0])))))
        efforts = robot.data.actuator_force[0, :10].abs()
        rows.append({
            "time_s": round((step + 1) * dt, 3),
            "teacher_executed": use_label,
            "root_xyz_m": [round(float(x), 5) for x in root.tolist()],
            "root_tilt_deg": round(tilt, 2),
            "cube_xyz_m": [round(float(x), 5) for x in cube.data.root_link_pos_w[0].tolist()],
            "finger_toy_contact": finger_touch,
            "feet_grounded": [bool(x) for x in env.scene.sensors["feet_ground_contact"].data.found[0].tolist()],
            "hand_ground_force_n": round(float(env.scene.sensors["hand_ground_contact"].data.force[0].norm(dim=-1).max()), 3),
            "leg_at_rated": bool((efforts >= rated * 0.995).any()),
            "leg_force_n_m": [round(float(x), 3) for x in efforts.tolist()],
        })
        if bool(done[0]):
            break
    report = {
        "schema": "robot-learning-lab.cove50-side-reach-native.v1",
        "selected_design_pointer_sha256": hashlib.sha256((ROOT / "robot/ape-current-design.json").read_bytes()).hexdigest(),
        "geometry_report": str(args.geometry),
        "geometry_sha256": hashlib.sha256(args.geometry.read_bytes()).hexdigest(),
        "body_xml": str(body),
        "body_sha256": hashlib.sha256(body.read_bytes()).hexdigest(),
        "seed": args.seed,
        "hand": args.hand,
        "rail_v": args.rail_v,
        "delay_ms": args.delay_ms,
        "start_xy_offset_m": [args.start_x_offset_m, args.start_y_offset_m],
        "toy_xy_offset_m": [args.toy_x_offset_m, args.toy_y_offset_m],
        "joint_target_offsets_rad": {
            **dict((name, float(value)) for name, value in args.joint_target_offset),
            f"{args.hand}_hip_pitch_base": args.right_hip_offset_rad,
            f"{args.hand}_ankle_pitch_base": args.right_ankle_offset_rad,
            f"{args.hand}_shoulder_pitch_base": args.right_shoulder_pitch_offset_rad,
            f"{args.hand}_gripper_base": args.right_gripper_offset_rad,
            f"{args.hand}_shoulder_roll_base": args.right_shoulder_roll_offset_rad,
        },
        "arm_xy_correction_rad": arm_xy_correction,
        "toy_step": {"time_s": args.toy_step_time_s, "x_m": args.toy_step_x_m,
                     "y_m": args.toy_step_y_m, "applied": toy_step_applied},
        "xy_correction_reference_sha256": (
            hashlib.sha256(args.xy_correction_reference.read_bytes()).hexdigest()
            if args.xy_correction_reference else None),
        "initial_root_xyz_m": initial_root,
        "initial_cube_xyz_m": initial_cube,
        "duration_s": rows[-1]["time_s"],
        "survived": len(rows) == round(args.duration_s / dt),
        "final_root_xyz_m": rows[-1]["root_xyz_m"],
        "final_cube_xyz_m": rows[-1]["cube_xyz_m"],
        "longest_continuous_finger_toy_contact_s": round(best_contact_run * dt, 3),
        "both_feet_grounded_fraction": round(sum(all(row["feet_grounded"]) for row in rows) / len(rows), 4),
        "rated_leg_fraction": round(sum(row["leg_at_rated"] for row in rows) / len(rows), 4),
        "max_hand_ground_force_n": max(row["hand_ground_force_n"] for row in rows),
        "qualification": ("Actor-to-feedback-teacher handoff diagnostic; contact after switch is not autonomous actor qualification."
                          if args.teacher_after_s >= 0 else
                          "Mixed feedback-label/BC collection, not an autonomous policy qualification."
                          if args.teacher_execution_prob else
                          "Fresh saved controller replay from a prepositioned attained stand; "
                          "no locomotion handoff or perception qualification."
                          if args.policy_checkpoint else
                          "Scripted native MJLab motor replay, not a saved learned policy or a locomotion handoff to the repositioned start."),
        "method": ("actor_to_feedback_teacher_native_handoff" if args.teacher_after_s >= 0 else
                   "feedback_labeled_mixed_bc_native_replay" if args.teacher_execution_prob else
                   "saved_bounded_learned_arm_residual_native_replay" if arm_residual_model is not None else
                   "saved_observed_goal_pi_residual_native_replay" if residual_reference is not None else
                   "saved_goal_residual_native_replay" if residual_weight is not None else
                   "saved_bc_actor_native_replay" if policy is not None else
                   "authored_native_teacher"),
        "policy_checkpoint_sha256": (hashlib.sha256(args.policy_checkpoint.read_bytes()).hexdigest()
                                      if args.policy_checkpoint else None),
        "policy_joint_names": policy_joint_names,
        "policy_arm_warmup_s": args.policy_arm_warmup_s,
        "label_feedback_checkpoint_sha256": (
            hashlib.sha256(args.label_feedback_checkpoint.read_bytes()).hexdigest()
            if args.label_feedback_checkpoint else None),
        "teacher_execution_prob": args.teacher_execution_prob,
        "teacher_after_s": args.teacher_after_s if args.teacher_after_s >= 0 else None,
        "teacher_executed_fraction": (round(sum(feedback_teacher_executed) / len(feedback_teacher_executed), 4)
                                      if feedback_teacher_executed else None),
        "dataset": str(args.dataset_out) if args.dataset_out else None,
        "actor_observation_terms": list(cfg.observations["actor"].terms),
        "teacher_max_abs_raw_action": round(max_abs_action, 5),
        "timeseries": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.dataset_out:
        args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            actor_observation=np.asarray(teacher_obs),
            executed_action=np.asarray(teacher_action),
            qpos=np.asarray(teacher_qpos),
            toy_relative_root_xyz_m=np.asarray(teacher_toy_relative_root),
            finger_toy_contact=np.asarray(teacher_contact),
            smoothstep_phase=np.asarray(teacher_phase),
            action_names=np.asarray(target_names),
            actor_observation_terms=np.asarray(list(cfg.observations["actor"].terms)),
            step_dt_s=dt,
        )
        if feedback_residual_trace:
            payload["feedback_residual"] = np.asarray(feedback_residual_trace)
        if feedback_teacher_actions:
            payload["feedback_teacher_action"] = np.asarray(feedback_teacher_actions)
            payload["feedback_teacher_executed"] = np.asarray(feedback_teacher_executed)
        if not args.policy_checkpoint:
            payload["teacher_action"] = payload["executed_action"]
        np.savez_compressed(args.dataset_out, **payload)
    print(json.dumps({key: value for key, value in report.items() if key != "timeseries"}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
