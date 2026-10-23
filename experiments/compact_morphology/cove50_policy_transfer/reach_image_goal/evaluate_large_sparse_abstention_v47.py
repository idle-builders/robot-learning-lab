"""Evaluate an observed large-sparse-correction abstention candidate.

The native robot dynamics use the rejected historical Direct Frame. A second, massless
MuJoCo model renders a counterfactual camera from the current native qpos;
only pixels and capture-time camera pose supply the controller's toy goal.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.manipulation.config.yam.env_cfgs import get_cube_spec
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from ape_walk import task as ape
from ape_walk.physics.frame50 import (clip_frame50_leg_targets,
                                      configure_frame50_motors, motor_configs)
from experiments.compact_morphology.cove50_policy_transfer.train_posture_arm import (
    SQUAT_ACTION_SCALE, SQUAT_COUNTERBALANCE_ARMS, expose_squat_blend,
    mask_squat_action_history, retarget,
)
from experiments.compact_morphology.cove50_policy_transfer.train_side_reach_arm_residual import (
    ArmResidualActor,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.selective_template_rgb_goal_v47 import (
    SelectiveTemplateRgbGoal,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.virtual_rgb_goal import (
    orange_bbox,
)


ROOT = Path(__file__).resolve().parents[4]
TASK = ape.ELBOW_SQUAT_HEADING_OBS_TASK_ID
POSTURE_S = 6.0
PREPOSITION_S = 2.5
REACH_S = 8.0
REACH_ROOT_TO_CUBE_B = (-0.07887782, -0.13679722)
STAND_DRIFT_B_X = 0.043  # Paired prior squat seeds 2906/7911: 0.042-0.044 m.
RIGHT_REACH_SCALE = {
    **{f"{side}_{axis}": (1.6 if axis in ("hip_pitch", "knee") else .5)
       for side in ("left", "right")
       for axis in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")},
    "left_shoulder_pitch": .5,
    "right_shoulder_yaw": .7,
    "right_shoulder_roll": .5,
    "right_shoulder_pitch": 1.2,
    "right_elbow_pitch": 1.2,
    "right_gripper": .5,
}

# Frozen before the native v14 development matrix. Inputs are body-frame
# image-goal errors at the actual reach entry; no simulator toy state is read.
V10_X_MAX_M = -.0095
V10_Y_MAX_M = -.0025
LIVE_X_MIN_M = -.004
LIVE_X_MAX_M = 0.0
LIVE_CORE_Y_MAX_M = -.004
LIVE_ADDED_X_MAX_M = -.001
LIVE_Y_MAX_M = -.002
HEIGHT_X_MAX_M = -.012
HEIGHT_Y_MAX_M = .002
BIAS_X_RANGE_M = (-.007, -.0055)
BIAS_Y_RANGE_M = (-.001, .001)
SPECIAL_X_RANGE_M = (0.0, .002)
SPECIAL_Y_RANGE_M = (.001, .004)
BIAS_BODY_Y_M = -.002
SCOPED_Y_X_MIN_M = .0015
SCOPED_Y_HIGH_Y_MIN_M = .009
SCOPED_Y_HIGH_Y_X_MIN_M = -.001
# Opened V37 forced routes supply the candidate actions for three remaining
# gaps. These bounded rules use only RGB goal error and capture camera pose.
LOCKED_GAP_X_M = (-.004, -.002)
LOCKED_GAP_Y_M = (-.006, -.003)
LOCKED_GAP_CAMERA_Y_MAX_M = -.09
SPECIAL_GAP_X_M = (-.0025, .0015)
SPECIAL_GAP_Y_M = (.002, .006)
SPECIAL_GAP_CAMERA_X_MAX_M = 0.0
LIVE_GAP_X_M = (-.004, -.002)
LIVE_GAP_Y_M = (.002, .006)
LIVE_GAP_CAMERA_X_MIN_M = .04
V10_TERMINAL_ARM_OFFSET_RAD = {
    "right_shoulder_yaw": -.132,
    "right_shoulder_roll": -.042,
    "right_shoulder_pitch": -.106,
    "right_elbow_pitch": .105,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoothstep(fraction: float) -> float:
    x = min(1.0, max(0.0, fraction))
    return x * x * (3 - 2 * x)


def reference_joint_pose(body: Path, trace: Path, joint_names: list[str]) -> tuple[torch.Tensor, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(body))
    with np.load(trace) as data:
        qpos = data["qpos"][0]
        recorded_names = data["action_names"].tolist()
    if recorded_names != joint_names:
        raise ValueError("Saved reach action order differs from this MJLab scene")
    if len(qpos) != model.nq + 7:
        raise ValueError("Saved reach trace must include one free cube after the robot")
    values = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        values.append(float(qpos[model.jnt_qposadr[joint_id]]))
    return torch.tensor(values, device="cuda:0"), qpos


def desired_reach_pose(start: torch.Tensor, names: list[str], geometry: dict,
                       pitch_offset: float) -> torch.Tensor:
    candidate = next(row for row in geometry["results"] if row["case"] == "side_lean_repositioned")
    desired = start.clone()
    for side_index, side in enumerate(("left", "right")):
        desired[names.index(f"{side}_hip_roll")] = candidate["leg_pose_rad"]["hip_roll_left_right"][side_index]
        for axis, value in zip(("hip_pitch", "knee", "ankle_pitch"),
                               candidate["leg_pose_rad"][side]):
            desired[names.index(f"{side}_{axis}")] = value
    desired[names.index("left_shoulder_pitch")] = -1.0
    for name, value in candidate["arm_pose_rad"].items():
        desired[names.index(name)] = value
    for name, value in (("right_hip_pitch", .02),
                        ("right_ankle_pitch", .01),
                        ("right_shoulder_pitch", pitch_offset),
                        ("right_gripper", .06)):
        desired[names.index(name)] += value
    return desired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--calibration-model", type=Path, required=True)
    parser.add_argument("--squat-checkpoint", type=Path, required=True)
    parser.add_argument("--reach-checkpoint", type=Path, required=True)
    parser.add_argument("--alternate-reach-checkpoint", type=Path, required=True)
    parser.add_argument("--specialist-reach-checkpoint", type=Path, required=True)
    parser.add_argument("--reach-controller", choices=("feedback", "learned-residual"),
                        default="feedback")
    parser.add_argument("--reach-reference", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--entry", choices=("reference-start", "continuous"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rail-v", type=float, default=12.0)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    parser.add_argument("--toy-offset-x-m", type=float, default=0.0)
    parser.add_argument("--toy-offset-y-m", type=float, default=0.0)
    parser.add_argument("--pitch-offset-rad", type=float, default=.105)
    parser.add_argument("--roll-clearance-rad", type=float, default=0.0)
    parser.add_argument("--terminal-roll-offset-rad", type=float, default=0.0)
    parser.add_argument("--terminal-roll-mode", choices=("fixed", "entry-error-x"), default="fixed")
    parser.add_argument("--terminal-roll-entry-x-start-m", type=float, default=-.0095)
    parser.add_argument("--terminal-roll-entry-x-full-m", type=float, default=-.0115)
    parser.add_argument("--reach-duration-s", type=float, default=REACH_S)
    parser.add_argument("--right-ankle-target-offset-rad", type=float, default=.01,
                        help="Diagnostic change to the authored reach target; default preserves the saved route")
    parser.add_argument("--gaze-head-yaw-rad", type=float,
                        help="Diagnostic head yaw target during the continuous handoff")
    parser.add_argument("--gaze-start-s", type=float, default=5.0)
    parser.add_argument("--gaze-ramp-s", type=float, default=2.0)
    parser.add_argument("--max-goal-age-s", type=float, default=9.0)
    parser.add_argument("--capture-stride-steps", type=int, default=5)
    parser.add_argument("--lock-goal-after-entry", action="store_true")
    parser.add_argument("--record-leg-efforts", action="store_true",
                        help="Retain per-joint leg actuator force for an opened failure diagnostic")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    if args.lock_goal_after_entry or args.reach_controller != "learned-residual":
        raise ValueError("V39 selects a learned actor and RGB goal mode at entry")
    if not 8.0 <= args.reach_duration_s <= 10.0 or args.reach_duration_s * 50 != round(args.reach_duration_s * 50):
        raise ValueError("Reach duration must be 8-10 s on the 50 Hz control grid")
    if args.terminal_roll_entry_x_full_m >= args.terminal_roll_entry_x_start_m:
        raise ValueError("Full terminal roll correction requires a more negative entry error")
    if args.gaze_head_yaw_rad is not None and (args.entry != "continuous" or
            not -1.5 <= args.gaze_head_yaw_rad <= 1.5 or args.gaze_ramp_s <= 0):
        raise ValueError("Gaze diagnostic requires continuous entry, legal yaw and positive ramp")
    if (args.entry != "continuous" or args.gaze_head_yaw_rad not in (-1.3, -1.5) or
            args.gaze_start_s != 0 or args.gaze_ramp_s != 7 or
            args.max_goal_age_s <= 0 or args.capture_stride_steps < 1):
        raise ValueError("Virtual RGB lesson requires the frozen continuous slow-gaze route")
    body = args.body_xml.resolve()
    model = mujoco.MjModel.from_xml_path(str(body))
    if model.nu != 21:
        raise ValueError("Expected an ordered 21-axis Cove body")
    geometry = json.loads(args.geometry.read_text())
    reach_saved = torch.load(args.reach_checkpoint, map_location="cpu", weights_only=True)
    alternate_saved = torch.load(args.alternate_reach_checkpoint,
                                 map_location="cpu", weights_only=True)
    specialist_saved = torch.load(args.specialist_reach_checkpoint,
                                  map_location="cpu", weights_only=True)
    if args.reach_controller == "feedback":
        if reach_saved.get("method") != "bounded_observed_goal_pi_residual_on_native_teacher":
            raise ValueError("Expected the frozen observed-goal PI hybrid")
    elif (reach_saved.get("method") not in (
              "bounded_learned_arm_residual_on_authored_path",
              "bounded_learned_arm_residual_on_handoff_path") or
          not reach_saved.get("include_error_integral") or
          not reach_saved.get("include_action_history") or
          reach_saved.get("input_dim") != 11):
        raise ValueError("Expected the saved four-arm residual with error integral and action history")
    if reach_saved.get("method") == "bounded_learned_arm_residual_on_handoff_path":
        if (abs(args.pitch_offset_rad - float(reach_saved["handoff_route_pitch_offset_rad"])) > 1e-9 or
                abs(args.roll_clearance_rad - float(reach_saved["handoff_route_roll_clearance_rad"])) > 1e-9):
            raise ValueError("Handoff-trained actor requires its recorded authored route")
    for key in ("method", "input_dim", "include_error_integral",
                "include_action_history", "policy_joint_names", "residual_bound",
                "xy_scale_m"):
        if alternate_saved.get(key) != reach_saved.get(key):
            raise ValueError(f"Alternate actor differs in policy contract: {key}")
    if not torch.equal(alternate_saved["reference_xy"], reach_saved["reference_xy"]):
        raise ValueError("Alternate actor has a different reach reference")
    for key in ("method", "input_dim", "include_error_integral",
                "include_action_history", "policy_joint_names", "residual_bound",
                "xy_scale_m"):
        if specialist_saved.get(key) != reach_saved.get(key):
            raise ValueError(f"Specialist actor differs in policy contract: {key}")
    if not torch.equal(specialist_saved["reference_xy"], reach_saved["reference_xy"]):
        raise ValueError("Specialist actor has a different reach reference")

    cfg = load_env_cfg(TASK, play=True)
    retarget(cfg, body, model, "squat", variance=False)
    expose_squat_blend(cfg)
    mask_squat_action_history(cfg)
    configure_frame50_motors(cfg, body, rail_v=args.rail_v, delay_s=args.delay_ms / 1000)
    clip_frame50_leg_targets(cfg, body, margin_rad=.02)
    # The saved squat actor commands only the legs and shoulder pitch. Enable
    # the other arm joints for a later reach target in the same action manager.
    action_scale = cfg.actions["joint_pos"].scale
    for side in ("left", "right"):
        action_scale[f"^{side}_shoulder_yaw$"] = .7
        action_scale[f"^{side}_shoulder_roll$"] = .8
        action_scale[f"^{side}_elbow_pitch$"] = 1.2
        action_scale[f"^{side}_gripper$"] = .5
    if args.gaze_head_yaw_rad is not None:
        action_scale["^head_yaw$"] = 1.5
    cfg.scene.entities["cube"] = EntityCfg(
        spec_fn=partial(get_cube_spec, cube_size=.01, mass=.00625))
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ContactSensorCfg(
            name="self_body_contact",
            primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"), reduce="none", num_slots=1),
        ContactSensorCfg(
            name="right_finger_tip_cube_contact",
            primary=ContactMatch(mode="geom", pattern="right_finger_tip", entity="robot"),
            secondary=ContactMatch(mode="geom", pattern="cube_geom", entity="cube"),
            fields=("found", "force"), reduce="netforce", num_slots=1),
        ContactSensorCfg(
            name="right_hand_cube_contact",
            primary=ContactMatch(mode="geom", pattern="right_hand", entity="robot"),
            secondary=ContactMatch(mode="geom", pattern="cube_geom", entity="cube"),
            fields=("found", "force"), reduce="netforce", num_slots=1),
    )
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.episode_length_s = POSTURE_S + PREPOSITION_S + args.reach_duration_s + 1.0
    cfg.auto_reset = False
    cfg.commands["twist"].resampling_time_range = (100.0, 100.0)
    reset = cfg.events["set_ground_state"].params
    reset["sitting_z_min"] = reset["sitting_z_max"] = .1712
    reset["sitting_joint_overrides"] = {
        2: -1.225, 3: 1.555, 4: -.287,
        7: -1.225, 8: 1.555, 9: -.287,
        **SQUAT_COUNTERBALANCE_ARMS,
    }
    reset["sitting_joint_noise_std"] = .005
    reset["sitting_tilt_max"] = math.radians(1)
    reset["sitting_prob"] = 1.0
    reset["standing_prob"] = 0.0

    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device="cuda:0"),
        clip_actions=load_rl_cfg(TASK).clip_actions,
    )
    robot = env.unwrapped.scene["robot"]
    cube = env.unwrapped.scene["cube"]
    names = list(robot.joint_names)
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    target_names = list(action_term._target_names)
    if target_names != names:
        raise ValueError("Reach residual and saved squat require the same 21-axis action order")
    head_index = target_names.index("head_yaw")
    reference, reference_qpos = reference_joint_pose(body, args.reach_reference, names)
    target = desired_reach_pose(reference, names, geometry, args.pitch_offset_rad)
    target[names.index("right_ankle_pitch")] += args.right_ankle_target_offset_rad - .01
    runner = load_runner_cls(TASK)(env, asdict(load_rl_cfg(TASK)), device="cuda:0")
    runner.load(str(args.squat_checkpoint), load_cfg={"actor": True}, strict=True,
                map_location="cuda:0")
    squat_policy = runner.get_inference_policy(device="cuda:0")
    command = env.unwrapped.command_manager.get_term("twist")
    dt = env.unwrapped.step_dt
    if abs(dt - .02) > 1e-9:
        raise ValueError("Expected 50 Hz controller")
    goal_source = SelectiveTemplateRgbGoal(calibration_model=args.calibration_model,
                                 selected_body_sha256=sha(body),
                                 max_age_s=args.max_goal_age_s,
                                 update_through_s=None)
    goal_source.template_entry_time_s = POSTURE_S + PREPOSITION_S
    capture_events = []
    missing_goal_at_s = None

    if args.entry == "reference-start":
        robot.write_root_link_pose_to_sim(torch.tensor(reference_qpos[:7][None, :], device="cuda:0"))
        robot.write_joint_state_to_sim(reference[None, :], torch.zeros_like(reference[None, :]))
        cube_pose = torch.tensor(reference_qpos[model.nq:model.nq + 7][None, :], device="cuda:0")
    else:
        root = robot.data.root_link_pose_w[0].clone()
        yaw = math.atan2(2 * float(root[3] * root[6] + root[4] * root[5]),
                         1 - 2 * float(root[5] ** 2 + root[6] ** 2))
        local_x = REACH_ROOT_TO_CUBE_B[0] + STAND_DRIFT_B_X + args.toy_offset_x_m
        local_y = REACH_ROOT_TO_CUBE_B[1] + args.toy_offset_y_m
        cube_pose = cube.data.root_link_pose_w.clone()
        cube_pose[0, 0] = root[0] + math.cos(yaw) * local_x - math.sin(yaw) * local_y
        cube_pose[0, 1] = root[1] + math.sin(yaw) * local_x + math.cos(yaw) * local_y
        cube_pose[0, 2] = .01
        cube_pose[0, 3:7] = torch.tensor((1., 0., 0., 0.), device="cuda:0")
    cube.write_root_link_pose_to_sim(cube_pose)
    cube.write_root_link_velocity_to_sim(torch.zeros((1, 6), device="cuda:0"))
    env.unwrapped.sim.forward()
    env.unwrapped.scene.update(0.0)

    rated = torch.tensor([act.effort_limit for act in motor_configs(body, rail_v=args.rail_v)[:10]],
                         device="cuda:0")
    if args.reach_controller == "feedback":
        weight = reach_saved["weight"].to("cuda:0")
        trajectory = reach_saved["center_ee_to_cube_xy_trajectory_m"].to("cuda:0")
        residual_bound = float(reach_saved["residual_action_bound"])
        feedback_p = float(reach_saved["proportional_gain"])
        feedback_i = float(reach_saved["integral_gain_per_s"])
        integral = torch.zeros(action_term.action_dim, device="cuda:0")
    else:
        expected = ("right_shoulder_yaw", "right_shoulder_roll",
                    "right_shoulder_pitch", "right_elbow_pitch")
        if tuple(reach_saved["policy_joint_names"]) != expected:
            raise ValueError("Learned residual joint names differ from the four-arm handoff")
        for name in expected:
            if name not in names or name not in target_names or name not in RIGHT_REACH_SCALE:
                raise ValueError(f"Missing physical target adapter for {name}")
        learned = ArmResidualActor(reach_saved["input_dim"]).to("cuda:0")
        learned.load_state_dict(reach_saved["state_dict"])
        learned.eval()
        alternate_learned = ArmResidualActor(alternate_saved["input_dim"]).to("cuda:0")
        alternate_learned.load_state_dict(alternate_saved["state_dict"])
        alternate_learned.eval()
        specialist_learned = ArmResidualActor(specialist_saved["input_dim"]).to("cuda:0")
        specialist_learned.load_state_dict(specialist_saved["state_dict"])
        specialist_learned.eval()
        active_learned = learned
        goal_bias_b = torch.zeros(2, device="cuda:0")
        selection_mode = None
        selection_features = None
        learned_reference = reach_saved["reference_xy"].to("cuda:0")
        learned_bound = float(reach_saved["residual_bound"])
        learned_xy_scale = float(reach_saved["xy_scale_m"])
        if learned_reference.shape != (400, 2) or learned_bound != .1 or learned_xy_scale != .01:
            raise ValueError("Learned reference or residual scale differs from the native reach contract")
        learned_indices = [target_names.index(name) for name in expected]
        learned_entry_error = None
        learned_error_integral = torch.zeros(2, device="cuda:0")
        previous_learned_residual = torch.zeros(4, device="cuda:0")
    arm_start = None
    reach_start = None
    route_terminal_roll_offset = None
    route_entry_error_x = None
    smoothed = None
    rows = []
    qpos = []
    actions_out = []
    first_action_hash = None
    boundary_hashes = {}
    max_unbounded_action = 0.0
    scan_extension_s = 0.0
    scan_trigger_s = None
    scan_probe_pixels = None
    scan_yaw_rad = None
    original_duration_s = (args.reach_duration_s if args.entry == "reference-start" else
                           POSTURE_S + PREPOSITION_S + args.reach_duration_s)
    steps = round((original_duration_s + (0.5 if args.entry == "continuous" else 0)) / dt)

    def raw_for_pose(pose: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        raw = torch.zeros((1, action_term.action_dim), device="cuda:0")
        for index, name in enumerate(target_names):
            scale = float(action_term._scale[0, index])
            if scale == 0:
                continue
            value = pose[names.index(name)]
            if residual is not None:
                value = value + RIGHT_REACH_SCALE.get(name, 0.0) * residual[index]
            raw[0, index] = (value - action_term._offset[0, index]) / scale
        return raw

    for step in range(steps):
        t = step * dt
        if t >= original_duration_s + scan_extension_s - 1e-9:
            break
        goal_age_s = None
        capture_event = None
        if (args.entry == "continuous" and
                step == round((POSTURE_S + PREPOSITION_S - dt) / dt) and
                args.gaze_head_yaw_rad is not None):
            # Observe the actual camera at the final preposition frame. This
            # peek does not update goal state or use simulator toy coordinates.
            observed_qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy().copy()
            goal_source.data.qpos[:] = observed_qpos
            goal_source.data.qvel[:] = 0
            mujoco.mj_forward(goal_source.model, goal_source.data)
            goal_source.renderer.update_scene(goal_source.data, camera="lower_rgb",
                                              scene_option=goal_source.option)
            bbox = orange_bbox(goal_source.renderer.render().copy())
            scan_probe_pixels = 0 if bbox is None else bbox["pixels"]
            if bbox is None:
                scan_extension_s = 0.5
                scan_trigger_s = round(t, 3)
                scan_yaw_rad = -1.5 if args.gaze_head_yaw_rad == -1.3 else -1.3
                goal_source.template_entry_time_s = POSTURE_S + PREPOSITION_S + scan_extension_s
        if args.entry == "reference-start":
            phase = "reach"
            reach_step = step
        elif t < POSTURE_S:
            phase = "posture"
            reach_step = None
        elif t < POSTURE_S + PREPOSITION_S + scan_extension_s:
            phase = "preposition"
            reach_step = None
        else:
            phase = "reach"
            reach_step = step - round((POSTURE_S + PREPOSITION_S + scan_extension_s) / dt)
        if phase not in boundary_hashes:
            h = hashlib.sha256()
            h.update(env.unwrapped.sim.data.qpos.cpu().numpy().tobytes())
            h.update(env.unwrapped.sim.data.qvel.cpu().numpy().tobytes())
            boundary_hashes[phase] = h.hexdigest()
        command.vel_command_b[:, 0] = 1.0 if phase == "posture" and t < 2.0 else 0.0
        if phase == "posture":
            with torch.inference_mode():
                actor_action = squat_policy(env.get_observations())
            bounded = actor_action.clone()
            inactive = set(range(bounded.shape[1])) - set(SQUAT_ACTION_SCALE)
            bounded[:, list(inactive)] = 0.0
            bounded.clamp_(-1, 1)
            smoothed = (bounded if smoothed is None else .2 * bounded + .8 * smoothed)
            raw = smoothed
        elif phase == "preposition":
            if arm_start is None:
                arm_start = robot.data.joint_pos[0].clone()
            blend = smoothstep((t - POSTURE_S) / PREPOSITION_S)
            raw = raw_for_pose(arm_start + blend * (reference - arm_start))
        else:
            if reach_start is None:
                reach_start = robot.data.joint_pos[0].clone()
            hand = robot.data.site_pos_w[:, robot.site_names.index("right_hand_center")]
            if reach_step % args.capture_stride_steps == 0:
                native_qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy().copy()
                capture_event, rgb = goal_source.capture(native_qpos, t)
                if capture_event["estimated_goal_world_xyz_m"] is not None:
                    capture_event["xy_error_truth_mm_scoring_only"] = round(float(
                        np.linalg.norm(np.asarray(capture_event["estimated_goal_world_xyz_m"][:2]) -
                                       cube.data.root_link_pos_w[0, :2].cpu().numpy()) * 1000), 3)
                capture_events.append(capture_event)
                if reach_step == 0 or abs(t - 10.0) < dt / 2:
                    image_path = args.out.with_name(args.out.stem +
                                                    ("-entry.png" if reach_step == 0 else "-10s.png"))
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    imageio.imwrite(image_path, rgb)
            goal_world, goal_age_s = goal_source.goal(t)
            if goal_world is None:
                missing_goal_at_s = round(t, 3)
                break
            goal_w = torch.tensor(goal_world[None, :], device="cuda:0", dtype=hand.dtype)
            distance_w = goal_w - hand
            live_xy = quat_apply(quat_inv(robot.data.root_link_quat_w), distance_w)[0, :2]
            if selection_mode is None:
                # A millimetre-scale template correction can cross a discrete
                # actor/height boundary. In the V39 corner branch, choose the
                # route from the already observed calibrated goal while the
                # arm still tracks the image-corrected goal. This separates
                # route identity from fine target refinement without truth.
                route_live_xy = live_xy
                route_goal_source = "selected_image_goal"
                if (capture_event["template_used_for_control"] and
                        capture_event["orange_pixels"] >= 64):
                    calibrated_goal = torch.tensor(
                        np.asarray(capture_event["pre_template_goal_world_xyz_m"])[None, :],
                        device="cuda:0", dtype=hand.dtype)
                    route_live_xy = quat_apply(
                        quat_inv(robot.data.root_link_quat_w), calibrated_goal - hand)[0, :2]
                    route_goal_source = "pre_template_calibrated_rgb"
                entry_error = (route_live_xy - learned_reference[0]).tolist()
                x, y = map(float, entry_error)
                camera_x, camera_y = map(float, capture_event["camera_world_xyz_m"][:2])
                if x < HEIGHT_X_MAX_M and y < HEIGHT_Y_MAX_M:
                    selection_mode = "v16_v10_height_locked"
                    active_learned = alternate_learned
                    goal_source.update_through_s = POSTURE_S + PREPOSITION_S + scan_extension_s
                elif x < V10_X_MAX_M and y < V10_Y_MAX_M:
                    selection_mode = "v16_v10_locked"
                    active_learned = alternate_learned
                    goal_source.update_through_s = POSTURE_S + PREPOSITION_S + scan_extension_s
                elif (LOCKED_GAP_X_M[0] <= x < LOCKED_GAP_X_M[1] and
                      LOCKED_GAP_Y_M[0] <= y < LOCKED_GAP_Y_M[1] and
                      camera_y < LOCKED_GAP_CAMERA_Y_MAX_M):
                    selection_mode = "v16_locked"
                    active_learned = learned
                    goal_source.update_through_s = POSTURE_S + PREPOSITION_S + scan_extension_s
                elif (SPECIAL_GAP_X_M[0] <= x < SPECIAL_GAP_X_M[1] and
                      SPECIAL_GAP_Y_M[0] <= y < SPECIAL_GAP_Y_M[1] and
                      camera_x < SPECIAL_GAP_CAMERA_X_MAX_M):
                    selection_mode = "v16_specialist_locked"
                    active_learned = specialist_learned
                    goal_source.update_through_s = POSTURE_S + PREPOSITION_S + scan_extension_s
                elif (LIVE_GAP_X_M[0] <= x < LIVE_GAP_X_M[1] and
                      LIVE_GAP_Y_M[0] <= y < LIVE_GAP_Y_M[1] and
                      camera_x > LIVE_GAP_CAMERA_X_MIN_M):
                    selection_mode = "v16_live"
                    active_learned = learned
                elif LIVE_X_MIN_M <= x < LIVE_X_MAX_M and (
                        y < LIVE_CORE_Y_MAX_M or
                        (x < LIVE_ADDED_X_MAX_M and y < LIVE_Y_MAX_M)):
                    selection_mode = "v16_live"
                    active_learned = learned
                elif BIAS_X_RANGE_M[0] <= x < BIAS_X_RANGE_M[1] and BIAS_Y_RANGE_M[0] <= y < BIAS_Y_RANGE_M[1]:
                    selection_mode = "v16_bias_y_locked"
                    active_learned = learned
                    goal_bias_b[1] = BIAS_BODY_Y_M
                    goal_source.update_through_s = POSTURE_S + PREPOSITION_S + scan_extension_s
                elif SPECIAL_X_RANGE_M[0] <= x < SPECIAL_X_RANGE_M[1] and SPECIAL_Y_RANGE_M[0] <= y < SPECIAL_Y_RANGE_M[1]:
                    selection_mode = "v16_specialist_locked"
                    active_learned = specialist_learned
                    goal_source.update_through_s = POSTURE_S + PREPOSITION_S + scan_extension_s
                elif x > SCOPED_Y_X_MIN_M or (x > SCOPED_Y_HIGH_Y_X_MIN_M and y > SCOPED_Y_HIGH_Y_MIN_M):
                    selection_mode = "v21_scoped_y_locked"
                    active_learned = learned
                    goal_bias_b[1] = BIAS_BODY_Y_M
                    goal_source.update_through_s = POSTURE_S + PREPOSITION_S + scan_extension_s
                else:
                    selection_mode = "v16_locked"
                    active_learned = learned
                    goal_source.update_through_s = POSTURE_S + PREPOSITION_S + scan_extension_s
                selection_features = {
                    "entry_error_xy_m": [round(x, 6), round(y, 6)],
                    "route_goal_source": route_goal_source,
                    "selected_image_error_xy_m": [round(float(v), 6) for v in
                                                   (live_xy - learned_reference[0]).tolist()],
                    "entry_camera_world_xyz_m": capture_event["camera_world_xyz_m"],
                }
            live_xy = live_xy + goal_bias_b
            if route_terminal_roll_offset is None:
                entry_reference = (trajectory[0] if args.reach_controller == "feedback"
                                   else learned_reference[0])
                route_entry_error_x = float(live_xy[0] - entry_reference[0])
                if args.terminal_roll_mode == "entry-error-x":
                    fraction = smoothstep((args.terminal_roll_entry_x_start_m - route_entry_error_x) /
                                          (args.terminal_roll_entry_x_start_m - args.terminal_roll_entry_x_full_m))
                    route_terminal_roll_offset = args.terminal_roll_offset_rad * fraction
                else:
                    route_terminal_roll_offset = args.terminal_roll_offset_rad
            blend = smoothstep(reach_step * dt / 4.0)
            pose = reach_start + blend * (target - reach_start)
            if selection_mode == "v16_v10_height_locked":
                for joint, offset_rad in V10_TERMINAL_ARM_OFFSET_RAD.items():
                    pose[names.index(joint)] += offset_rad * blend
            # Route the hand outward while it passes the right shin, then
            # return to the validated final contact pose by transition end.
            pulse = math.sin(math.pi * min(1.0, reach_step * dt / 4.0))
            pose[names.index("right_shoulder_roll")] += (args.roll_clearance_rad * pulse +
                                                         route_terminal_roll_offset * blend)
            if args.reach_controller == "feedback":
                error = live_xy - trajectory[min(reach_step, len(trajectory) - 1)]
                direction = torch.mv(weight, error)
                integral = (integral + feedback_i * dt * direction).clamp(-residual_bound, residual_bound)
                residual = (feedback_p * direction + integral).clamp(-residual_bound, residual_bound) * blend
                raw = raw_for_pose(pose, residual)
            else:
                live_error = (live_xy - learned_reference[min(reach_step, len(learned_reference) - 1)]) / learned_xy_scale
                if learned_entry_error is None:
                    learned_entry_error = live_error.clone()
                learned_error_integral = learned_error_integral + live_error * dt
                features = torch.cat((learned_entry_error, live_error,
                                      learned_error_integral, previous_learned_residual,
                                      torch.tensor([blend], device="cuda:0")))[None, :]
                with torch.inference_mode():
                    correction = active_learned(features)[0] * learned_bound * blend
                # The handoff action manager uses a different shoulder-roll
                # scale. Convert reach raw corrections through physical joint
                # targets; history stays relative to this active authored path.
                residual = torch.zeros(action_term.action_dim, device="cuda:0")
                residual[learned_indices] = correction
                raw = raw_for_pose(pose, residual)
                previous_learned_residual = correction / learned_bound
        gaze_target = None
        if args.gaze_head_yaw_rad is not None:
            gaze_target = args.gaze_head_yaw_rad * smoothstep(
                (t - args.gaze_start_s) / args.gaze_ramp_s)
            if scan_yaw_rad is not None:
                gaze_target = scan_yaw_rad
            raw[0, head_index] = ((gaze_target - action_term._offset[0, head_index]) /
                                  action_term._scale[0, head_index])
        max_unbounded_action = max(max_unbounded_action, float(raw.abs().max()))
        applied = raw.clamp(-1, 1)
        if first_action_hash is None:
            first_action_hash = hashlib.sha256(applied.cpu().numpy().tobytes()).hexdigest()
        qpos.append(env.unwrapped.sim.data.qpos[0].cpu().numpy().copy())
        actions_out.append(applied[0].cpu().numpy().copy())
        with torch.inference_mode():
            _, _, done, _ = env.step(applied)
        sensors = env.unwrapped.scene.sensors
        finger = bool(sensors["right_finger_tip_cube_contact"].data.found[0].any())
        palm = bool(sensors["right_hand_cube_contact"].data.found[0].any())
        effort = robot.data.actuator_force[0, :10].abs()
        rows.append({
            "time_s": round(t + dt, 3), "phase": phase,
            "pelvis_xyz_m": [round(float(x), 5) for x in robot.data.root_link_pos_w[0].tolist()],
            "cube_xyz_m": [round(float(x), 5) for x in cube.data.root_link_pos_w[0].tolist()],
            "hand_xyz_m": [round(float(x), 5) for x in robot.data.site_pos_w[0, robot.site_names.index("right_hand_center")].tolist()],
            "finger_contact": finger, "palm_contact": palm,
            "cube_to_hand_distance_m": round(float((cube.data.root_link_pos_w[0] -
                robot.data.site_pos_w[0, robot.site_names.index("right_hand_center")]).norm()), 5),
            "feet_grounded": [bool(x) for x in sensors["feet_ground_contact"].data.found[0].tolist()],
            "hand_floor_force_n": round(float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).max()), 3),
            "self_body_contact": bool(sensors["self_body_contact"].data.found[0].any()),
            "leg_at_rated": bool((effort >= rated * .995).any()),
            "max_leg_effort_nm": round(float(effort.max()), 3),
            **({"leg_effort_nm": [round(float(x), 4) for x in effort.tolist()]}
               if args.record_leg_efforts else {}),
            "max_abs_raw_action": round(float(raw.abs().max()), 4),
            **({"image_goal_age_s": round(float(goal_age_s), 3),
                "image_capture_accepted": capture_event["accepted"] if capture_event else None}
               if phase == "reach" else {}),
            **({"gaze_target_rad": round(gaze_target, 5),
                "head_yaw_rad": round(float(robot.data.joint_pos[0, head_index]), 5),
                "head_yaw_effort_nm": round(float(robot.data.actuator_force[0, head_index]), 5)}
               if gaze_target is not None else {}),
            **({"controller_target_error_xy_m": [round(float(x), 6) for x in
                                                   (live_xy - learned_reference[min(reach_step, len(learned_reference) - 1)]).tolist()],
                "learned_arm_residual_raw": [round(float(x), 6) for x in correction.tolist()],
                "learned_error_integral": [round(float(x), 6) for x in learned_error_integral.tolist()]}
               if phase == "reach" and args.reach_controller == "learned-residual" else {}),
        })
        if bool(done[0]):
            break

    summaries = {}
    for phase in ("posture", "preposition", "reach"):
        part = [row for row in rows if row["phase"] == phase]
        if not part:
            continue
        streak = best = 0
        for row in part:
            streak = streak + 1 if row["finger_contact"] else 0
            best = max(best, streak)
        summaries[phase] = {
            "duration_s": round(len(part) * dt, 3),
            "both_feet_fraction": round(sum(all(row["feet_grounded"]) for row in part) / len(part), 4),
            "rated_leg_fraction": round(sum(row["leg_at_rated"] for row in part) / len(part), 4),
            "max_hand_floor_force_n": max(row["hand_floor_force_n"] for row in part),
            "self_body_contact_fraction": round(sum(row["self_body_contact"] for row in part) / len(part), 4),
            "continuous_finger_contact_s": round(best * dt, 3),
            "both_jaws_contact_frames": sum(row["finger_contact"] and row["palm_contact"] for row in part),
            "min_cube_to_hand_distance_m": min(row["cube_to_hand_distance_m"] for row in part),
            "start_pelvis_xyz_m": part[0]["pelvis_xyz_m"],
            "end_pelvis_xyz_m": part[-1]["pelvis_xyz_m"],
        }
    report = {
        "schema": "robot-learning-lab.cove50-squat-reach-handoff.v1",
        "evaluator_sha256": sha(Path(__file__)),
        "entry": args.entry, "one_sim_without_skill_reset": args.entry == "continuous",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "body_xml": str(args.body_xml), "body_sha256": sha(args.body_xml),
        "squat_checkpoint_sha256": sha(args.squat_checkpoint),
        "reach_checkpoint_sha256": sha(args.reach_checkpoint),
        "alternate_reach_checkpoint_sha256": sha(args.alternate_reach_checkpoint),
        "specialist_reach_checkpoint_sha256": sha(args.specialist_reach_checkpoint),
        "selected_reach_checkpoint_sha256": (sha(args.alternate_reach_checkpoint)
                                               if selection_mode in ("v16_v10_height_locked", "v16_v10_locked") else
                                               sha(args.specialist_reach_checkpoint)
                                               if selection_mode == "v16_specialist_locked" else
                                               sha(args.reach_checkpoint)),
        "adaptive_variant": "v47_large_sparse_positive_camera_abstention_candidate",
        "reacquisition_probe_time_s": (round(POSTURE_S + PREPOSITION_S - dt, 3)
                                        if scan_probe_pixels is not None else None),
        "reacquisition_probe_orange_pixels": scan_probe_pixels,
        "reacquisition_trigger_s": scan_trigger_s,
        "reacquisition_head_yaw_rad": scan_yaw_rad,
        "reach_start_s": round(POSTURE_S + PREPOSITION_S + scan_extension_s, 3),
        "negative_camera_sparse_abstention_entry": bool(
            capture_events and capture_events[0].get("negative_camera_sparse_abstention")),
        "positive_camera_large_sparse_abstention_entry": bool(
            capture_events and capture_events[0].get("positive_camera_large_sparse_abstention")),
        "selection_mode": selection_mode,
        "goal_bias_body_xy_m": [0.0, BIAS_BODY_Y_M if selection_mode in ("v16_bias_y_locked", "v21_scoped_y_locked") else 0.0],
        "v10_terminal_arm_offset_rad": V10_TERMINAL_ARM_OFFSET_RAD,
        "v10_terminal_arm_offset_applied": selection_mode == "v16_v10_height_locked",
        "selection_features": selection_features,
        "selection_thresholds_m": {"v10_x_max": V10_X_MAX_M,
                                   "v10_y_max": V10_Y_MAX_M,
                                   "live_x_min": LIVE_X_MIN_M,
                                   "live_x_max": LIVE_X_MAX_M,
                                   "live_core_y_max": LIVE_CORE_Y_MAX_M,
                                   "live_added_x_max": LIVE_ADDED_X_MAX_M,
                                   "live_y_max": LIVE_Y_MAX_M,
                                   "height_x_max": HEIGHT_X_MAX_M,
                                   "height_y_max": HEIGHT_Y_MAX_M,
                                   "bias_x_range": BIAS_X_RANGE_M,
                                   "bias_y_range": BIAS_Y_RANGE_M,
                                   "special_x_range": SPECIAL_X_RANGE_M,
                                   "special_y_range": SPECIAL_Y_RANGE_M,
                                   "scoped_y_x_min": SCOPED_Y_X_MIN_M,
                                   "scoped_y_high_y_min": SCOPED_Y_HIGH_Y_MIN_M,
                                   "scoped_y_high_y_x_min": SCOPED_Y_HIGH_Y_X_MIN_M},
        "v38_observable_gap_thresholds": {
            "locked_error_x_m": LOCKED_GAP_X_M, "locked_error_y_m": LOCKED_GAP_Y_M,
            "locked_camera_y_max_m": LOCKED_GAP_CAMERA_Y_MAX_M,
            "specialist_error_x_m": SPECIAL_GAP_X_M, "specialist_error_y_m": SPECIAL_GAP_Y_M,
            "specialist_camera_x_max_m": SPECIAL_GAP_CAMERA_X_MAX_M,
            "live_error_x_m": LIVE_GAP_X_M, "live_error_y_m": LIVE_GAP_Y_M,
            "live_camera_x_min_m": LIVE_GAP_CAMERA_X_MIN_M},
        "reach_controller": args.reach_controller,
        "reach_action_adapter": ("Four reach raw-arm corrections map through the native reach joint "
                                 "scales to physical targets, then through the handoff action manager; "
                                 "previous-action memory is relative to the active authored path."
                                 if args.reach_controller == "learned-residual" else None),
        "reach_reference_sha256": sha(args.reach_reference),
        "geometry_sha256": sha(args.geometry),
        "seed": args.seed, "rail_v": args.rail_v, "delay_ms": args.delay_ms,
        "toy_offset_b_m": [args.toy_offset_x_m, args.toy_offset_y_m],
        "pitch_offset_rad": args.pitch_offset_rad,
        "roll_clearance_rad": args.roll_clearance_rad,
        "terminal_roll_offset_rad": args.terminal_roll_offset_rad,
        "terminal_roll_mode": args.terminal_roll_mode,
        "terminal_roll_entry_x_start_m": args.terminal_roll_entry_x_start_m,
        "terminal_roll_entry_x_full_m": args.terminal_roll_entry_x_full_m,
        "route_entry_error_x_m": route_entry_error_x,
        "applied_terminal_roll_offset_rad": route_terminal_roll_offset,
        "reach_duration_s": args.reach_duration_s,
        "right_ankle_target_offset_rad": args.right_ankle_target_offset_rad,
        "gaze_head_yaw_rad": args.gaze_head_yaw_rad,
        "gaze_start_s": args.gaze_start_s if args.gaze_head_yaw_rad is not None else None,
        "gaze_ramp_s": args.gaze_ramp_s if args.gaze_head_yaw_rad is not None else None,
        "toy_goal_source": "observable_negative_or_large_positive_camera_sparse_abstention_or_template_or_calibrated_virtual_rgb_exact_pose_known_cube",
        "calibration_model_sha256": goal_source.calibration_model_sha256,
        "template_source_sha256": sha(Path(__file__).with_name("template_virtual_rgb_goal_v30.py")),
        "selector_source_sha256": sha(Path(__file__).with_name("selective_template_rgb_goal_v47.py")),
        "body_role": "rejected_historical_direct_frame_sandbox",
        "virtual_camera_mass_and_dynamics": "separate renderer only; selected native body unchanged",
        "max_goal_age_s": args.max_goal_age_s,
        "capture_stride_steps": args.capture_stride_steps,
        "image_goal_capture_count": goal_source.capture_count,
        "image_goal_accepted_capture_count": goal_source.accepted_count,
        "image_goal_used_update_count": goal_source.used_update_count,
        "image_goal_lock_after_entry": selection_mode != "v16_live",
        "image_goal_missing_at_s": missing_goal_at_s,
        "image_goal_capture_events": capture_events,
        "toy_root_relative_setup_b_m": [REACH_ROOT_TO_CUBE_B[0] + STAND_DRIFT_B_X + args.toy_offset_x_m,
                                        REACH_ROOT_TO_CUBE_B[1] + args.toy_offset_y_m],
        "initial_cube_xyz_m": rows[0]["cube_xyz_m"],
        "boundary_state_sha256": boundary_hashes,
        "first_action_sha256": first_action_hash,
        "max_abs_unbounded_action": round(max_unbounded_action, 4),
        **({"leg_joint_order": names[:10],
            "leg_rated_limit_nm": [round(float(x), 4) for x in rated.tolist()]}
           if args.record_leg_efforts else {}),
        "duration_s": rows[-1]["time_s"],
        "ended_early": rows[-1]["time_s"] < original_duration_s + scan_extension_s - 1e-9,
        "phases": summaries,
        "qualification": ("Historical V42b opened diagnostic candidate on rejected Direct Frame. "
                          "The reach-entry toy goal uses calibrated RGB and applies rendered-template matching only for sufficiently visible, materially shifted observed orange pixels. Template matching compares observed orange pixels "
                          "with rendered hypotheses using exact virtual camera and robot pose and a "
                          "known upright 20 mm cube. No simulator toy coordinates enter inference. "
                          "The legal -1.3 or -1.5 rad gaze is supplied as a varied camera command; the observed far negative-X "
                          "region extends the saved V10 height route to Y below +2 mm. Later live "
                          "updates retain the older calibrated pixel goal. Native opened and disjoint "
                          "results decide robustness. "
                          "The motor arm preposition lasts 2.5 s; an absent final preposition RGB mask triggers a physical 0.2 rad alternate head turn and 0.5 s arm hold. Sparse template correction abstains to calibrated RGB in the V45 negative camera quadrant and when a positive-X, negative-Y camera sees a large >=6 mm correction from 24-63 toy pixels. These observable rules are candidates pending matched native retention and a disjoint score. "
                          "No selected-body, physical camera, grasp or possession qualification."),
        "timeseries": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.trace, qpos=np.asarray(qpos),
                        action=np.asarray(actions_out), step_dt_s=dt,
                        phase=np.asarray([row["phase"] for row in rows]))
    print(json.dumps({k: v for k, v in report.items() if k != "timeseries"}, indent=2))
    goal_source.close()
    env.close()


if __name__ == "__main__":
    main()
