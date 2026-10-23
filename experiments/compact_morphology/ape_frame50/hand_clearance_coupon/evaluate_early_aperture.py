"""Replay an early aperture transition with the intact native support commands."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares

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


ROOT = Path(__file__).resolve().parents[4]
TASK = ape.ELBOW_SQUAT_HEADING_OBS_TASK_ID
POSTURE_S = 6.0
PREPOSITION_S = 1.5
REACH_S = 2.0
ARM_JOINTS = ("right_shoulder_yaw", "right_shoulder_roll",
              "right_shoulder_pitch", "right_elbow_pitch")
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


def solve_aperture_entry(model: mujoco.MjModel, live_qpos: np.ndarray,
                         cube_xyz: np.ndarray, above_cube_mm: float = 18.0) -> dict:
    """Fit four arm joints with fixed root/legs and an open jaw."""
    data = mujoco.MjData(model)
    site_id = model.site("right_hand_center").id
    jaw_adr = model.joint("right_gripper").qposadr[0]
    arm_adr = np.array([model.joint(name).qposadr[0] for name in ARM_JOINTS])
    arm0 = live_qpos[arm_adr].copy()
    low = np.array([model.joint(name).range[0] for name in ARM_JOINTS])
    high = np.array([model.joint(name).range[1] for name in ARM_JOINTS])
    low = np.maximum(low, arm0 - .55)
    high = np.minimum(high, arm0 + .55)
    target = cube_xyz + np.array([0, 0, above_cube_mm / 1000])

    def site_at(values: np.ndarray) -> np.ndarray:
        data.qpos[:] = live_qpos[:model.nq]
        data.qpos[arm_adr] = values
        data.qpos[jaw_adr] = 0.0
        mujoco.mj_forward(model, data)
        return data.site_xpos[site_id].copy()

    def residual(values: np.ndarray) -> np.ndarray:
        return np.r_[(site_at(values) - target) * 1000, (values - arm0) / .4]

    fit = least_squares(residual, np.clip(arm0, low + 1e-8, high - 1e-8),
                        bounds=(low, high), max_nfev=120,
                        ftol=1e-9, xtol=1e-9, gtol=1e-9)
    actual = site_at(fit.x)
    return {
        "joint_targets": {name: float(value) for name, value in zip(ARM_JOINTS, fit.x)},
        "arm_delta_rad": {name: float(value) for name, value in zip(ARM_JOINTS, fit.x - arm0)},
        "target_site_xyz_m": target.tolist(),
        "fitted_site_xyz_m": actual.tolist(),
        "site_error_mm": float(np.linalg.norm((actual - target) * 1000)),
        "solver_success": bool(fit.success), "solver_nfev": int(fit.nfev),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--squat-checkpoint", type=Path, required=True)
    parser.add_argument("--reach-checkpoint", type=Path, required=True)
    parser.add_argument("--reach-controller", choices=("feedback", "learned-residual"),
                        default="feedback")
    parser.add_argument("--reach-reference", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--route-plan", type=Path, required=True)
    parser.add_argument("--goal-screen", type=Path, required=True)
    parser.add_argument("--reference-actions", type=Path, required=True)
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
    parser.add_argument("--record-leg-efforts", action="store_true",
                        help="Retain per-joint leg actuator force for an opened failure diagnostic")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    if args.reach_duration_s != REACH_S:
        raise ValueError("Early aperture transition starts exactly two seconds into reach")
    if args.entry != "continuous":
        raise ValueError("Aperture handoff must keep one continuous squat-to-close scene")
    route_plan = json.loads(args.route_plan.read_text())
    route_phases = route_plan["route"]
    route_names = [phase["phase"] for phase in route_phases]
    if route_names != ["transition", "open_hold", "close", "hold"]:
        raise ValueError("Unexpected early-aperture phase order")
    if any(phase["duration_s"] <= 0 or phase["duration_s"] * 50 != round(phase["duration_s"] * 50)
           for phase in route_phases):
        raise ValueError("Route durations must be positive on the 50 Hz grid")
    goal_screen = json.loads(args.goal_screen.read_text())
    if not goal_screen["screen_pass"]:
        raise ValueError("Early aperture geometry screen did not pass")
    route_goal = np.asarray(goal_screen["selected"]["arm_rad"])
    with np.load(args.reference_actions) as saved_reference:
        reference_trace_qpos = saved_reference["qpos"].copy()
        reference_actions = saved_reference["action"].copy()
        reference_dt = float(saved_reference["step_dt_s"])
    if reference_dt != .02 or reference_trace_qpos.shape[1] != 35 or reference_actions.shape[1] != 21:
        raise ValueError("Expected intact 50 Hz 21-axis reference action trace")
    if args.terminal_roll_entry_x_full_m >= args.terminal_roll_entry_x_start_m:
        raise ValueError("Full terminal roll correction requires a more negative entry error")
    body = args.body_xml.resolve()
    model = mujoco.MjModel.from_xml_path(str(body))
    if model.nu != 21:
        raise ValueError("Expected an ordered 21-axis Cove body")
    geometry = json.loads(args.geometry.read_text())
    reach_saved = torch.load(args.reach_checkpoint, map_location="cpu", weights_only=True)
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
    cfg.episode_length_s = (POSTURE_S + PREPOSITION_S + args.reach_duration_s +
                            sum(phase["duration_s"] for phase in route_phases) + 1.0)
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
    route_start = None
    route_source_check = None
    route_cube_xyz = None
    rows = []
    qpos = []
    actions_out = []
    first_action_hash = None
    boundary_hashes = {}
    max_unbounded_action = 0.0
    reach_end_s = POSTURE_S + PREPOSITION_S + args.reach_duration_s
    phase_bounds = {}
    previous_end = reach_end_s
    for phase_cfg in route_phases:
        phase_bounds[phase_cfg["phase"]] = (previous_end, previous_end + phase_cfg["duration_s"])
        previous_end += phase_cfg["duration_s"]
    steps = round(previous_end / dt)
    arm_indices = [names.index(name) for name in ARM_JOINTS]
    arm_qpos_adr = [model.joint(name).qposadr[0] for name in ARM_JOINTS]
    jaw_index = names.index("right_gripper")
    source_reference_qpos = reference_trace_qpos[round(reach_end_s / dt)]
    source_cube = source_reference_qpos[model.nq:model.nq + 3]
    if len(reference_actions) < steps or route_goal.shape != (4,):
        raise ValueError("Reference actions or goal missing for complete early route")
    replaced_action_indices = [target_names.index(name) for name in
                               (*ARM_JOINTS, "right_gripper")]

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
        if t < POSTURE_S:
            phase = "posture"
            reach_step = None
        elif t < POSTURE_S + PREPOSITION_S:
            phase = "preposition"
            reach_step = None
        elif t < reach_end_s:
            phase = "reach"
            reach_step = step - round((POSTURE_S + PREPOSITION_S) / dt)
        else:
            phase = next(name for name, (_, end) in phase_bounds.items() if t < end - 1e-9)
            reach_step = None
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
        elif phase == "reach":
            if reach_start is None:
                reach_start = robot.data.joint_pos[0].clone()
            hand = robot.data.site_pos_w[:, robot.site_names.index("right_hand_center")]
            distance_w = cube.data.root_link_pos_w - hand
            live_xy = quat_apply(quat_inv(robot.data.root_link_quat_w), distance_w)[0, :2]
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
                    correction = learned(features)[0] * learned_bound * blend
                # The handoff action manager uses a different shoulder-roll
                # scale. Convert reach raw corrections through physical joint
                # targets; history stays relative to this active authored path.
                residual = torch.zeros(action_term.action_dim, device="cuda:0")
                residual[learned_indices] = correction
                raw = raw_for_pose(pose, residual)
                previous_learned_residual = correction / learned_bound
        else:
            if route_start is None:
                route_start = robot.data.joint_pos[0].clone()
                route_cube_xyz = cube.data.root_link_pos_w[0].cpu().numpy().copy()
                live_qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy().copy()
                arm_error = np.abs(live_qpos[arm_qpos_adr] - source_reference_qpos[arm_qpos_adr])
                leg_error = np.abs(live_qpos[7:17] - source_reference_qpos[7:17])
                root_pos_error_mm = float(np.linalg.norm((live_qpos[:3] - source_reference_qpos[:3]) * 1000))
                cube_error_mm = float(np.linalg.norm((route_cube_xyz - source_cube) * 1000))
                route_source_check = {
                    "arm_max_abs_error_rad": float(arm_error.max()),
                    "leg_max_abs_error_rad": float(leg_error.max()),
                    "root_position_error_mm": root_pos_error_mm,
                    "cube_xyz_error_mm": cube_error_mm,
                    "source_jaw_rad": float(live_qpos[model.joint("right_gripper").qposadr[0]]),
                    "finger_contact": bool(env.unwrapped.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any()),
                    "palm_contact": bool(env.unwrapped.scene.sensors["right_hand_cube_contact"].data.found[0].any()),
                }
                if (arm_error.max() > .02 or leg_error.max() > .02 or
                    root_pos_error_mm > 3 or cube_error_mm > 3 or
                    route_source_check["finger_contact"] or route_source_check["palm_contact"]):
                    raise ValueError(f"Live source differs from screened precontact state: {route_source_check}")
            phase_start, phase_end = phase_bounds[phase]
            fraction = smoothstep((t - phase_start) / (phase_end - phase_start))
            pose = route_start.clone()
            def set_arm(values):
                for index, value in zip(arm_indices, values):
                    pose[index] = float(value)
            if phase == "transition":
                set_arm((1 - fraction) * route_start[arm_indices].cpu().numpy() +
                        fraction * route_goal)
                jaw_fraction = smoothstep((t - reach_end_s) / .5)
                pose[jaw_index] = route_start[jaw_index] * (1 - jaw_fraction)
            elif phase == "open_hold":
                set_arm(route_goal)
                pose[jaw_index] = 0.0
            else:
                set_arm(route_goal)
                pose[jaw_index] = .5 * fraction if phase == "close" else .5
            raw_pose = raw_for_pose(pose)
            raw = torch.tensor(reference_actions[step][None, :], device="cuda:0")
            raw[:, replaced_action_indices] = raw_pose[:, replaced_action_indices]
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
            "right_gripper_rad": round(float(robot.data.joint_pos[0, names.index("right_gripper")]), 5),
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
            **({"controller_target_error_xy_m": [round(float(x), 6) for x in
                                                   (live_xy - learned_reference[min(reach_step, len(learned_reference) - 1)]).tolist()],
                "learned_arm_residual_raw": [round(float(x), 6) for x in correction.tolist()],
                "learned_error_integral": [round(float(x), 6) for x in learned_error_integral.tolist()]}
               if phase == "reach" and args.reach_controller == "learned-residual" else {}),
        })
        if bool(done[0]):
            break

    summaries = {}
    for phase in ("posture", "preposition", "reach", *route_names):
        part = [row for row in rows if row["phase"] == phase]
        if not part:
            continue
        streak = best = both_streak = both_best = 0
        for row in part:
            streak = streak + 1 if row["finger_contact"] else 0
            best = max(best, streak)
            both_streak = both_streak + 1 if row["finger_contact"] and row["palm_contact"] else 0
            both_best = max(both_best, both_streak)
        cube_xyz = np.asarray([row["cube_xyz_m"] for row in part])
        cube_xy_drift_mm = float(np.max(np.linalg.norm(
            (cube_xyz[:, :2] - np.asarray(route_cube_xyz[:2])) * 1000, axis=1))) if route_cube_xyz is not None else None
        summaries[phase] = {
            "duration_s": round(len(part) * dt, 3),
            "both_feet_fraction": round(sum(all(row["feet_grounded"]) for row in part) / len(part), 4),
            "rated_leg_fraction": round(sum(row["leg_at_rated"] for row in part) / len(part), 4),
            "max_hand_floor_force_n": max(row["hand_floor_force_n"] for row in part),
            "self_body_contact_fraction": round(sum(row["self_body_contact"] for row in part) / len(part), 4),
            "continuous_finger_contact_s": round(best * dt, 3),
            "both_jaws_contact_frames": sum(row["finger_contact"] and row["palm_contact"] for row in part),
            "continuous_both_jaw_contact_s": round(both_best * dt, 3),
            "min_cube_to_hand_distance_m": min(row["cube_to_hand_distance_m"] for row in part),
            "max_cube_xy_drift_from_route_mm": (round(cube_xy_drift_mm, 3)
                                                 if cube_xy_drift_mm is not None else None),
            "cube_start_z_m": part[0]["cube_xyz_m"][2],
            "cube_end_z_m": part[-1]["cube_xyz_m"][2],
            "final_gripper_rad": part[-1]["right_gripper_rad"],
            "start_pelvis_xyz_m": part[0]["pelvis_xyz_m"],
            "end_pelvis_xyz_m": part[-1]["pelvis_xyz_m"],
        }
    report = {
        "schema": "robot-learning-lab.cove50-native-early-aperture.v3",
        "evaluator_sha256": sha(Path(__file__)),
        "entry": args.entry, "one_sim_without_skill_reset": args.entry == "continuous",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "body_xml": str(args.body_xml), "body_sha256": sha(args.body_xml),
        "squat_checkpoint_sha256": sha(args.squat_checkpoint),
        "reach_checkpoint_sha256": sha(args.reach_checkpoint),
        "reach_controller": args.reach_controller,
        "reach_action_adapter": ("Four reach raw-arm corrections map through the native reach joint "
                                 "scales to physical targets, then through the handoff action manager; "
                                 "previous-action memory is relative to the active authored path."
                                 if args.reach_controller == "learned-residual" else None),
        "reach_reference_sha256": sha(args.reach_reference),
        "geometry_sha256": sha(args.geometry),
        "route_plan_sha256": sha(args.route_plan),
        "goal_screen_sha256": sha(args.goal_screen),
        "reference_actions_sha256": sha(args.reference_actions),
        "route_source_check": route_source_check,
        "route_goal_arm_rad": route_goal.tolist(),
        "replayed_action_channels": [name for name in target_names if
                                     name not in (*ARM_JOINTS, "right_gripper")],
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
        "route_phase_durations_s": {phase["phase"]: phase["duration_s"] for phase in route_phases},
        "route_cube_xyz_m": (route_cube_xyz.tolist() if route_cube_xyz is not None else None),
        "right_ankle_target_offset_rad": args.right_ankle_target_offset_rad,
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
        "ended_early": len(rows) < steps,
        "phases": summaries,
        "qualification": ("One-sim squat, early reach, margin-screened arm transition and close "
                          "using the prior native leg/left-arm actions. This exact-start route is "
                          "an authored teacher diagnostic with simulator truth, not a learned "
                          "acquisition policy. No lift is commanded."),
        "timeseries": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.trace, qpos=np.asarray(qpos),
                        action=np.asarray(actions_out), step_dt_s=dt,
                        phase=np.asarray([row["phase"] for row in rows]))
    print(json.dumps({k: v for k, v in report.items() if k != "timeseries"}, indent=2))
    env.close()


if __name__ == "__main__":
    main()
