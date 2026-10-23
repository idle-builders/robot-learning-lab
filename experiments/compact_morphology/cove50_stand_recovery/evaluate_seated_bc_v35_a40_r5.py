"""Exact-task, unscored seated BC CUDA environment readiness visits."""

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
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
    parser.add_argument("--intervention-route", required=True,
                        choices=("bc", "environment_timing"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True,
                        choices=(35003, 35900))
    parser.add_argument("--preflight-sentinel", action="store_true",
                        help="Unscored seed35900 one-step exact CUDA BC entry check")
    parser.add_argument("--render-preflight", action="store_true",
                        help="Second seed35900 one-step visit with native render and MP4")
    parser.add_argument("--timing-probe", action="store_true",
                        help="Unscored seed35003 zero-action full native render with explicit resets")
    parser.add_argument("--device", default="cuda:0",
                        help="Native simulation device; CPU runs are diagnostics, not CUDA score substitutes")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--state-out", type=Path)
    parser.add_argument("--native-trajectory-out", type=Path,
                        help="Save full post-reset observations, 21 actions/targets and physical states")
    parser.add_argument("--body-xml", type=Path)
    parser.add_argument("--cove50-body-xml", type=Path,
                        help="Retarget the 21-axis Cove body, motor law, reset and height goals")
    parser.add_argument("--physics-model", choices=("baseline", "candidate"), default="baseline")
    parser.add_argument("--mode", choices=("cycle", "stand"), default="cycle")
    parser.add_argument("--task", choices=(
        ape_walk.task.ELBOW_SQUAT_TASK_ID,
        ape_walk.task.ELBOW_SQUAT_LOW_NOISE_TASK_ID,
        ape_walk.task.ELBOW_SQUAT_HEADING_TASK_ID,
        ape_walk.task.ELBOW_SQUAT_HEADING_OBS_TASK_ID,
    ))
    parser.add_argument("--zero-actions", action="store_true")
    parser.add_argument("--scripted-target-probe", action="store_true",
                        help="Probe the retargeted joint pose through native Cove motors; not a learned policy")
    parser.add_argument("--scripted-hip-pitch", type=float)
    parser.add_argument("--scripted-knee", type=float)
    parser.add_argument("--scripted-ankle-pitch", type=float)
    parser.add_argument("--motor-rail-v", type=float, default=12.0)
    parser.add_argument("--command-delay-ms", type=float, default=0.0)
    parser.add_argument("--leg-target-margin-rad", type=float)
    parser.add_argument("--action-smoothing-alpha", type=float)
    parser.add_argument("--squat-active-mask", action="store_true",
                        help="Zero channels unused by the Cove squat teacher")
    parser.add_argument("--action-abs-bound", type=float,
                        help="Bound each normalized actor action before motor targets")
    parser.add_argument("--expose-squat-blend", action="store_true")
    parser.add_argument("--mask-action-history", action="store_true")
    parser.add_argument("--start-squat", action="store_true")
    parser.add_argument("--settled-squat-reset", action="store_true",
                        help="Use a measured reached-squat pose with small reset disturbance")
    parser.add_argument("--recover-from-start", action="store_true",
                        help="Hold stand command throughout from a real seated reset")
    parser.add_argument("--posture-ramp-s", type=float,
                        help="Override the sit/stand command target ramp for a bounded diagnostic")
    parser.add_argument("--initial-blend", type=float,
                        help="Diagnostic only: align internal target blend to a seated reset")
    parser.add_argument("--seated-rise-stage", action="store_true",
                        help="Use the separate seated-only training entry and failure contract")
    parser.add_argument("--randomized-physics", action="store_true",
                        help="Exercise the seated-stage training randomization in a native diagnostic")
    parser.add_argument("--scripted-min-blend", type=float,
                        help="Diagnostic: park a scripted seated-rise target above this blend")
    parser.add_argument("--duration-s", type=float, default=10.0,
                        help="Shorten a diagnostic episode; default retains the 10 s score")
    parser.add_argument("--probe-joint", choices=("hip_pitch", "ankle_pitch"),
                        help="Apply a temporary equal left/right normalized-action bias")
    parser.add_argument("--probe-bias-action", type=float, default=0.0)
    parser.add_argument("--probe-start-s", type=float, default=0.2)
    parser.add_argument("--probe-end-s", type=float, default=0.5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    exact_body = root / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
    exact_checkpoint = (
        root / "artifacts/ape-cove50-stand-recovery/seated-bc-v35/model_bc_seed.pt"
        if args.intervention_route in ("bc", "environment_timing") else
        root / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25"
        / "checkpoints/model_25.pt"
    )
    if (args.device != "cuda:0"
            or (args.seed == 35900) != args.preflight_sentinel
            or (args.seed == 35003) != args.timing_probe
            or (args.preflight_sentinel and args.timing_probe)
            or (args.render_preflight and not args.preflight_sentinel)
            or (args.render_preflight and not args.video)
            or (args.preflight_sentinel and bool(args.video) != args.render_preflight)
            or (args.duration_s != (0.02 if args.preflight_sentinel else 10.0))
            or (args.intervention_route == "environment_timing") != args.timing_probe
            or (args.timing_probe and not args.video)
            or not args.cove50_body_xml or args.cove50_body_xml.resolve() != exact_body
            or not args.checkpoint or args.checkpoint.resolve() != exact_checkpoint
            or not (args.mode == "cycle" and args.start_squat and args.recover_from_start
                    and args.seated_rise_stage and args.randomized_physics)
            or args.task is not None or args.body_xml or args.physics_model != "baseline"
            or args.posture_ramp_s is not None or args.settled_squat_reset
            or args.leg_target_margin_rad is not None
            or args.expose_squat_blend or args.mask_action_history):
        parser.error("V35 permits only the exact randomized CUDA task: unscored seed35900 one-step or seed35003 full timing")
    if args.checkpoint is None:
        parser.error("V35 requires the route-bound saved checkpoint on every route")
    if args.zero_actions or args.scripted_target_probe or args.scripted_min_blend is not None:
        parser.error("V35 route selects the actor/script; generic zero/script/park flags are forbidden")
    if any(value is not None for value in (args.scripted_hip_pitch, args.scripted_knee,
                                           args.scripted_ankle_pitch)):
        parser.error("V35 uses only the source scripted target, without leg-angle overrides")
    if args.probe_joint or args.probe_bias_action or args.action_abs_bound is not None or args.action_smoothing_alpha is not None or args.squat_active_mask:
        parser.error("V35 allows only the three declared routes")
    if not (args.zero_actions or args.scripted_target_probe) and args.checkpoint is None:
        parser.error("checkpoint is required unless --zero-actions is set")
    if args.physics_model == "candidate" and args.body_xml:
        parser.error("--body-xml and --physics-model candidate are mutually exclusive")
    gpu_names = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip().splitlines()
    cuda_device_name = torch.cuda.get_device_name(0)
    if gpu_names != ["NVIDIA A40"] or cuda_device_name != "NVIDIA A40":
        raise RuntimeError("V35 A40 evaluator requires one native NVIDIA A40 in nvidia-smi and Torch")
    if args.cove50_body_xml and (args.body_xml or args.physics_model != "baseline"):
        parser.error("--cove50-body-xml requires the default baseline mode and no --body-xml")
    if args.recover_from_start and (not args.start_squat or args.mode != "cycle"):
        parser.error("--recover-from-start requires --start-squat in cycle mode")
    if args.posture_ramp_s is not None and args.posture_ramp_s <= 0:
        parser.error("--posture-ramp-s must be positive")
    if args.initial_blend is not None and not (
        0 <= args.initial_blend <= 1 and args.cove50_body_xml and args.scripted_target_probe
        and args.start_squat and args.recover_from_start
    ):
        parser.error("--initial-blend requires a Cove scripted seated recovery and a value in [0,1]")
    if args.seated_rise_stage and not (
        args.cove50_body_xml and args.start_squat and args.recover_from_start
        and args.mode == "cycle" and not args.settled_squat_reset
        and args.initial_blend is None
    ):
        parser.error("Seated rise requires Cove native seated recovery without phase override")
    if args.seated_rise_stage and args.task not in (
        None, ape_walk.task.ELBOW_SQUAT_HEADING_OBS_TASK_ID
    ):
        parser.error("Seated rise must use the trainer's heading-observed squat task")
    if args.scripted_min_blend is not None and not (
        args.seated_rise_stage and args.scripted_target_probe
        and 0 < args.scripted_min_blend < 1
    ):
        parser.error("A parked blend requires a scripted seated-rise stage and value in (0,1)")
    if args.randomized_physics and not args.seated_rise_stage:
        parser.error("Randomized-physics diagnostic requires the seated-rise stage")
    if args.randomized_physics and (
        args.motor_rail_v != 12.0 or args.command_delay_ms != 0.0
    ):
        parser.error("Randomized seated-stage replay must retain the trainer motor envelope")
    if args.duration_s <= 0 or args.duration_s > 10.0:
        parser.error("--duration-s must be in (0, 10]")
    if args.probe_bias_action and not args.probe_joint:
        parser.error("--probe-joint is required for a nonzero probe bias")
    if (not args.preflight_sentinel and
            not 0 <= args.probe_start_s < args.probe_end_s <= args.duration_s):
        parser.error("Probe window must be within the episode")
    if (any(value is not None for value in (args.scripted_hip_pitch, args.scripted_knee,
                                           args.scripted_ankle_pitch))
            and (not args.scripted_target_probe or any(value is None for value in
                (args.scripted_hip_pitch, args.scripted_knee, args.scripted_ankle_pitch)))):
        parser.error("A scripted leg override requires all three angles and --scripted-target-probe")

    task_id = (ape_walk.task.ELBOW_SQUAT_HEADING_OBS_TASK_ID
               if args.seated_rise_stage else
               ape_walk.task.ELBOW_STAND_TASK_ID if args.mode == "stand"
               else args.task or ape_walk.task.ELBOW_SQUAT_TASK_ID)
    # Seated replay uses the trainer config, including observation corruption.
    # The generic play config disables corruption and clears curriculum.
    env_cfg_play = not args.seated_rise_stage
    cfg = load_env_cfg(task_id, play=env_cfg_play)
    candidate_model = None
    if args.physics_model == "candidate":
        from ape_walk.physics.candidate import MODEL, configure_candidate
        cfg = configure_candidate(cfg)
        candidate_model = MODEL
    if args.body_xml:
        cfg.scene.entities["robot"].spec_fn = partial(ape_walk.task._ape_spec, args.body_xml.resolve())
        cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
    if args.cove50_body_xml:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from experiments.compact_morphology.cove50_policy_transfer.train_posture_arm import (
            retarget, SQUAT_LEGS, SQUAT_COUNTERBALANCE_ARMS, SQUAT_ACTION_SCALE,
            expose_squat_blend, mask_squat_action_history, build_seated_rise_env_cfg,
        )
        from ape_walk.physics.frame50 import (configure_frame50_motors,
                                              clip_frame50_leg_targets,
                                              reveal_frame50_proxy_geometry_for_video)
        body = args.cove50_body_xml.resolve()
        if args.seated_rise_stage:
            cfg = build_seated_rise_env_cfg(body, variance=args.randomized_physics)
        else:
            retarget(cfg, body, mujoco.MjModel.from_xml_path(str(body)),
                     "stand" if args.mode == "stand" else "squat",
                     variance=args.randomized_physics)
        if args.expose_squat_blend:
            expose_squat_blend(cfg)
        if args.mask_action_history:
            mask_squat_action_history(cfg)
        if not args.randomized_physics:
            configure_frame50_motors(cfg, body, rail_v=args.motor_rail_v,
                                     delay_s=args.command_delay_ms / 1000.0)
        if args.leg_target_margin_rad is not None:
            clip_frame50_leg_targets(cfg, body,
                                     margin_rad=args.leg_target_margin_rad)
    elif args.scripted_target_probe:
        parser.error("--scripted-target-probe requires --cove50-body-xml")
    if candidate_model or args.body_xml or args.cove50_body_xml:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (
            ape_walk.task.ContactSensorCfg(
                name="self_body_contact",
                primary=ape_walk.task.ContactMatch(
                    mode="subtree", pattern="pelvis", entity="robot"),
                secondary=ape_walk.task.ContactMatch(
                    mode="subtree", pattern="pelvis", entity="robot"),
                fields=("found", "force"), reduce="none", num_slots=1,
            ),
        )
    if args.cove50_body_xml and not args.seated_rise_stage:
        # The pelvis-subtree sensor above detects robot self contact. It does
        # not detect a fallen pelvis touching terrain. Keep that physical
        # failure separate from sole and hand contacts in the score trace.
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (
            ape_walk.task.ContactSensorCfg(
                name="nonsole_body_ground_contact",
                primary=ape_walk.task.ContactMatch(
                    mode="geom",
                    pattern=(
                        r"^pelvis_frame_budget$",
                        r"^torso_frame_compute_budget$",
                        r"^(left|right)_(thigh|shin|upper_arm|forearm)_frame_budget$",
                    ),
                    entity="robot",
                ),
                secondary=ape_walk.task.ContactMatch(mode="body", pattern="terrain"),
                fields=("found", "force"), reduce="none", num_slots=1,
            ),
        )
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.episode_length_s = 12.0 if args.seated_rise_stage else 10.5
    cfg.auto_reset = False
    cfg.commands["twist"].resampling_time_range = (100.0, 100.0)
    if args.posture_ramp_s is not None:
        cfg.commands["twist"].ramp_s = args.posture_ramp_s
    reset = cfg.events["set_ground_state"].params
    if args.settled_squat_reset:
        if not (args.start_squat and args.cove50_body_xml):
            parser.error("Settled squat reset requires --start-squat on Cove")
        reset["sitting_z_min"] = 0.1712
        reset["sitting_z_max"] = 0.1712
        reset["sitting_joint_overrides"] = {
            2: -1.225, 3: 1.555, 4: -0.287,
            7: -1.225, 8: 1.555, 9: -0.287,
            **SQUAT_COUNTERBALANCE_ARMS,
        }
        reset["sitting_joint_noise_std"] = 0.005
        reset["sitting_tilt_max"] = math.radians(1)
    reset["sitting_prob"] = 1.0 if args.start_squat else 0.0
    reset["standing_prob"] = 0.0 if args.start_squat else 1.0
    if args.video:
        cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
        cfg.viewer.lookat = (0.0, 0.0, 0.2)
        cfg.viewer.distance = 0.65
        cfg.viewer.azimuth = 90.0
        cfg.viewer.elevation = -12.0
        cfg.viewer.width = 640
        cfg.viewer.height = 480
    action_clip = load_rl_cfg(task_id).clip_actions
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device=args.device,
                              render_mode="rgb_array" if args.video else None),
        clip_actions=action_clip,
    )
    robot = env.unwrapped.scene["robot"]
    motor_delay_config = [
        {"min_lag_steps": act.cfg.delay_min_lag,
         "max_lag_steps": act.cfg.delay_max_lag}
        for act in robot.actuators
    ]
    observed_motor_delay_lags = set()
    if args.video and args.cove50_body_xml:
        reveal_frame50_proxy_geometry_for_video(env.unwrapped.sim.mj_model)
    hand_site_ids = [robot.site_names.index(name) for name in
                     ("left_hand_center", "right_hand_center")]
    sole_site_ids = [robot.site_names.index(name) for name in
                     ("left_sole_center", "right_sole_center")]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    leg_rated_effort = torch.tensor(
        ([act.effort_limit for act in cfg.scene.entities["robot"].articulation.actuators[:10]]
         if args.cove50_body_xml else [1.56] * 10),
        device=args.device,
    )
    targets = dict(SQUAT_LEGS | SQUAT_COUNTERBALANCE_ARMS)
    default_pos = robot.data.default_joint_pos

    def script_policy(_):
        action = torch.zeros_like(robot.data.joint_pos)
        blend = command.alpha[:, None]
        for index, target in targets.items():
            action[:, index] = (blend[:, 0] *
                                (target - default_pos[:, index]) / SQUAT_ACTION_SCALE[index])
        return action

    runner = load_runner_cls(task_id)(env, asdict(load_rl_cfg(task_id)), device=args.device)
    runner.load(str(args.checkpoint), load_cfg={"actor": True, "critic": True},
                strict=True, map_location=args.device)
    # V25's saved infos restores common_step_counter=624; the BC export has
    # infos={}. Use the same historical native replay counter on all routes.
    v25_checkpoint = (root / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25"
                      / "checkpoints/model_25.pt")
    evaluation_counter = int(torch.load(v25_checkpoint, map_location="cpu",
                                        weights_only=True)["infos"]["env_state"]
                             ["common_step_counter"])
    env.unwrapped.common_step_counter = evaluation_counter
    actor_policy = runner.get_inference_policy(device=args.device)
    critic = runner.alg.critic.eval()
    sensors = env.unwrapped.scene.sensors
    command = env.unwrapped.command_manager.get_term("twist")
    if args.initial_blend is not None:
        # Reset calls command.compute before the evaluator takes its first
        # action. Override both the current blend and its first-step re-init
        # to isolate the source-height versus actual seated-pose hypothesis.
        command._alpha[:] = args.initial_blend
        command._alpha_from_height = lambda: torch.full_like(command.alpha, args.initial_blend)
    reward_manager = env.unwrapped.reward_manager
    reward_term_names = list(reward_manager.active_terms)
    initial_state_hash = hashlib.sha256()
    for tensor in (env.unwrapped.sim.data.qpos, env.unwrapped.sim.data.qvel):
        initial_state_hash.update(tensor.cpu().numpy().tobytes())
    initial_sole_xy = robot.data.site_pos_w[0, sole_site_ids, :2].cpu().numpy().copy()
    initial_qpos = env.unwrapped.sim.data.qpos[0].cpu().numpy().copy()
    initial_qvel = env.unwrapped.sim.data.qvel[0].cpu().numpy().copy()
    initial_predicted_sole_gap_m = (
        float(env.unwrapped._seated_last_predicted_gap_m[0])
        if args.seated_rise_stage else None
    )
    initial_physics = None
    initial_motors = None
    if args.randomized_physics:
        sim = env.unwrapped.sim
        torso = int(robot.indexing.body_ids[robot.body_names.index("torso")])
        soles = [int(robot.indexing.geom_ids[robot.geom_names.index(f"{side}_sole")])
                 for side in ("left", "right")]
        initial_physics = {
            "torso_mass_kg": float(sim.model.body_mass[0, torso]),
            "sole_slide_friction": [
                float(sim.model.geom_friction[0, geom, 0]) for geom in soles
            ],
            "robot_body_names": list(robot.body_names),
            "robot_body_mass_kg": [
                float(sim.model.body_mass[0, int(body)])
                for body in robot.indexing.body_ids
            ],
            "robot_body_inertia_kg_m2": [
                sim.model.body_inertia[0, int(body)].cpu().tolist()
                for body in robot.indexing.body_ids
            ],
            "robot_body_ipos_m": [
                sim.model.body_ipos[0, int(body)].cpu().tolist()
                for body in robot.indexing.body_ids
            ],
            "robot_body_iquat_wxyz": [
                sim.model.body_iquat[0, int(body)].cpu().tolist()
                for body in robot.indexing.body_ids
            ],
        }
        initial_motors = {
            "target_names": [act.target_names[0] for act in robot.actuators],
            "rated_effort_nm": [float(act.force_limit[0, 0]) for act in robot.actuators],
            "effort_factor": [float(act.force_limit[0, 0] / act.default_force_limit[0, 0])
                              for act in robot.actuators],
            "stiffness_factor": [float(act.stiffness[0, 0] / act.default_stiffness[0, 0])
                                 for act in robot.actuators],
            "damping_factor": [float(act.damping[0, 0] / act.default_damping[0, 0])
                               for act in robot.actuators],
            "delay_lag_steps": [
                int(act._delay_buffer.current_lags[0])
                if act._delay_buffer is not None else None
                for act in robot.actuators
            ],
        }
    first_action_hash = None
    smoothed_action = None
    frames = []
    qpos_frames = []
    native_frames = {name: [] for name in (
        "actor_observation", "critic_observation", "critic_value",
        "actor_candidate_action", "script_candidate_action", "selected_action_is_script",
        "control_index", "episode_step_index", "pre_action_time_s", "previous_action_valid", "previous_action_time_s",
        "previous_action_age_s", "previous_raw_action", "previous_applied_action",
        "command_squat_pre", "command_x_pre", "rise_phase_pre",
        "raw_action", "applied_action", "processed_target", "qpos", "qvel", "reward_rate",
        "post_time_s", "post_sole_contact", "post_sole_force_n", "post_sole_site_xyz_m",
        "post_leg_torque_nm", "post_all_actuator_force_nm", "post_leg_rated_duty",
        "post_pelvis_pose_w", "post_pelvis_z_m", "post_hand_site_xyz_m",
        "post_tilt_deg", "post_hand_force_n", "post_body_contact", "post_done",
        "post_floor_nonsole_body_contact", "post_termination_flags",
        "post_command_x", "post_rise_phase", "post_step_reward",
    )} if args.native_trajectory_out else None
    records = []
    dt = env.unwrapped.step_dt
    end_termination_terms = None
    done_events = []
    reset_events = []
    episode_step = 0
    termination_terms = list(env.unwrapped.termination_manager.active_terms)
    previous_raw = torch.zeros_like(robot.data.joint_pos)
    previous_applied = torch.zeros_like(robot.data.joint_pos)
    for step in range(round(args.duration_s / dt)):
        t = step * dt
        squat_requested = not args.recover_from_start and args.mode == "cycle" and (
            (t < 2.0 or t >= 6.0) if args.start_squat else (2.0 <= t < 6.0)
        )
        command.vel_command_b[:, 0] = 1.0 if squat_requested else 0.0
        with torch.inference_mode():
            # Capture the exact values preceding the observation and chosen action.
            # Previous actions are the actual prior call to this evaluator's env.step.
            command_x_pre = command.command[0, 0].detach().clone()
            rise_phase_pre = command.alpha[0].detach().clone()
            actor_observation = env.get_observations()
            actor_actions = actor_policy(actor_observation)
            script_actions = script_policy(actor_observation)
            critic_values = critic(actor_observation)
            gravity_x_pre = float(actor_observation["actor"][0, 3])
            scripted_step = args.intervention_route == "scripted"
            actions = (torch.zeros_like(actor_actions)
                       if args.intervention_route == "environment_timing" else
                       script_actions if scripted_step else actor_actions)
            if args.probe_joint and args.probe_start_s <= t < args.probe_end_s:
                actions = actions.clone()
                indices = (2, 7) if args.probe_joint == "hip_pitch" else (4, 9)
                actions[:, indices] += args.probe_bias_action
            if step == 0:
                first_action_hash = hashlib.sha256(actions.cpu().numpy().tobytes()).hexdigest()
            bounded_actions = actions.clone()
            if args.squat_active_mask:
                if not args.cove50_body_xml:
                    parser.error("Squat action mask requires the Cove body")
                inactive = set(range(bounded_actions.shape[1])) - set(SQUAT_ACTION_SCALE)
                bounded_actions[:, list(inactive)] = 0.0
            if args.action_abs_bound is not None:
                if args.action_abs_bound <= 0:
                    parser.error("Action bound must be positive")
                bounded_actions.clamp_(-args.action_abs_bound, args.action_abs_bound)
            if args.action_smoothing_alpha is not None:
                if not 0 < args.action_smoothing_alpha <= 1:
                    parser.error("Action smoothing alpha must be in (0, 1]")
                smoothed_action = (bounded_actions.clone() if smoothed_action is None
                                   else args.action_smoothing_alpha * bounded_actions +
                                   (1 - args.action_smoothing_alpha) * smoothed_action)
                applied_actions = smoothed_action
            else:
                applied_actions = bounded_actions
            _, reward, done, _ = env.step(applied_actions)
            # Read back the action that crossed the wrapper/action-manager
            # boundary. This includes any wrapper clip before motor targets.
            effective_applied_actions = env.unwrapped.action_manager.action.detach().clone()
        previous_raw_next = actions.detach().clone()
        previous_applied_next = effective_applied_actions
        observed_motor_delay_lags.update(
            int(act._delay_buffer.current_lags[0])
            for act in robot.actuators if act._delay_buffer is not None
        )
        if native_frames is not None:
            reward_rate = np.asarray([values[0] for _, values in reward_manager.get_active_iterable_terms(0)])
            assert len(reward_rate) == len(reward_term_names)
            for name, values in (
                ("actor_observation", actor_observation["actor"]),
                ("critic_observation", actor_observation["critic"]),
                ("critic_value", critic_values),
                ("actor_candidate_action", actor_actions),
                ("script_candidate_action", script_actions),
                ("raw_action", actions),
                ("applied_action", effective_applied_actions),
                ("processed_target", action_term._processed_actions),
                ("qpos", env.unwrapped.sim.data.qpos),
                ("qvel", env.unwrapped.sim.data.qvel),
            ):
                native_frames[name].append(values[0].detach().cpu().numpy().copy())
            native_frames["selected_action_is_script"].append(scripted_step)
            native_frames["reward_rate"].append(reward_rate)
            native_frames["control_index"].append(step)
            native_frames["episode_step_index"].append(episode_step)
            native_frames["pre_action_time_s"].append(t)
            native_frames["previous_action_valid"].append(episode_step > 0)
            native_frames["previous_action_time_s"].append((step - 1) * dt if episode_step else -1.0)
            native_frames["previous_action_age_s"].append(dt if episode_step else 0.0)
            native_frames["previous_raw_action"].append(previous_raw[0].cpu().numpy().copy())
            native_frames["previous_applied_action"].append(previous_applied[0].cpu().numpy().copy())
            native_frames["command_squat_pre"].append(squat_requested)
            native_frames["command_x_pre"].append(float(command_x_pre))
            native_frames["rise_phase_pre"].append(float(rise_phase_pre))
        if args.state_out:
            qpos_frames.append(np.concatenate((
                robot.data.root_link_pose_w[0].cpu().numpy(),
                robot.data.joint_pos[0].cpu().numpy(),
            )))
        if args.video and step % 2 == 0:
            frames.append(env.unwrapped.render())
        quat = robot.data.root_link_pose_w[0, 3:7].cpu().tolist()
        root_xy = robot.data.root_link_pos_w[0, :2].cpu().tolist()
        yaw = math.atan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                         1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0,
            1 - 2 * (quat[1] ** 2 + quat[2] ** 2)))))
        hand_force = float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).max())
        self_body_contact = bool(sensors["self_body_contact"].data.found[0].any()) if candidate_model or args.body_xml or args.cove50_body_xml else False
        floor_body_contact = bool(sensors["nonsole_body_ground_contact"].data.found[0].any()) if args.cove50_body_xml else False
        body_contact = self_body_contact or floor_body_contact
        feet = sensors["feet_ground_contact"].data.found[0].cpu().tolist()
        foot_forces = sensors["feet_ground_contact"].data.force[0].norm(dim=-1).cpu().tolist()
        effort = robot.data.actuator_force[0, :10].abs()
        hand_z = robot.data.site_pos_w[0, hand_site_ids, 2].cpu().tolist()
        termination_flags = [bool(env.unwrapped.termination_manager.get_term(name)[0])
                             for name in termination_terms]
        if native_frames is not None:
            native_frames["post_time_s"].append((step + 1) * dt)
            native_frames["post_sole_contact"].append(np.asarray(feet, dtype=bool))
            native_frames["post_sole_force_n"].append(np.asarray(foot_forces, dtype=float))
            native_frames["post_sole_site_xyz_m"].append(
                robot.data.site_pos_w[0, sole_site_ids, :].cpu().numpy().copy())
            native_frames["post_leg_torque_nm"].append(
                robot.data.actuator_force[0, :10].cpu().numpy().copy())
            native_frames["post_all_actuator_force_nm"].append(
                robot.data.actuator_force[0].cpu().numpy().copy())
            native_frames["post_leg_rated_duty"].append(bool((effort >= leg_rated_effort * 0.995).any()))
            native_frames["post_pelvis_pose_w"].append(
                robot.data.root_link_pose_w[0].cpu().numpy().copy())
            native_frames["post_pelvis_z_m"].append(float(robot.data.root_link_pos_w[0, 2]))
            native_frames["post_hand_site_xyz_m"].append(
                robot.data.site_pos_w[0, hand_site_ids, :].cpu().numpy().copy())
            native_frames["post_tilt_deg"].append(tilt)
            native_frames["post_hand_force_n"].append(hand_force)
            native_frames["post_body_contact"].append(body_contact)
            native_frames["post_done"].append(bool(done[0]))
            native_frames["post_floor_nonsole_body_contact"].append(floor_body_contact)
            native_frames["post_termination_flags"].append(termination_flags)
            native_frames["post_command_x"].append(float(command.command[0, 0]))
            native_frames["post_rise_phase"].append(float(command.alpha[0]))
            native_frames["post_step_reward"].append(float(reward[0]))
        records.append({
            "time_s": round(t + dt, 3),
            "command_squat": squat_requested,
            "target_blend": round(float(command.alpha[0]), 3),
            "command_heading_error_rad": round(float(command.vel_command_b[0, 2]), 3),
            "pelvis_z_m": round(float(robot.data.root_link_pos_w[0, 2]), 4),
            "root_xy_m": [round(x, 4) for x in root_xy],
            "sole_site_xy_m": robot.data.site_pos_w[0, sole_site_ids, :2].cpu().numpy().round(6).tolist(),
            "actor_observed_command_x": round(float(command.command[0, 0]), 4),
            "critic_value": round(float(critic_values[0].reshape(-1)[0]), 6),
            "selected_action_is_script": scripted_step,
            "gravity_x_pre": round(gravity_x_pre, 6),
            "yaw_rad": round(yaw, 3),
            "tilt_deg": round(tilt, 2),
            "feet_grounded": [bool(x) for x in feet],
            "foot_force_n": [round(float(x), 2) for x in foot_forces],
            "hand_force_n": round(hand_force, 3),
            "hand_height_m": [round(float(z), 4) for z in hand_z],
            "body_contact": body_contact,
            "floor_nonsole_body_contact": floor_body_contact,
            "leg_joint_pos_rad": [round(x, 3) for x in robot.data.joint_pos[0, :10].cpu().tolist()],
            "leg_raw_action": [round(x, 3) for x in actions[0, :10].cpu().tolist()],
            "leg_applied_action": [round(x, 3) for x in effective_applied_actions[0, :10].cpu().tolist()],
            "leg_joint_effort_nm": [round(x, 3) for x in robot.data.actuator_force[0, :10].cpu().tolist()],
            "max_leg_torque_nm": round(float(effort.max()), 3),
            "mean_leg_torque_nm": round(float(effort.mean()), 3),
            "leg_torque_saturated": bool((effort >= leg_rated_effort * 0.995).any()),
            "step_reward": round(float(reward[0]), 4),
            "done": bool(done[0]),
            "episode_step_index": episode_step,
        })
        previous_raw = previous_raw_next
        previous_applied = previous_applied_next
        if bool(done[0]):
            manager = env.unwrapped.termination_manager
            terms = [name for name in manager.active_terms
                     if bool(manager.get_term(name)[0])]
            done_events.append({"after_control_step": step + 1, "termination_terms": terms})
            if args.timing_probe:
                if step + 1 < round(args.duration_s / dt):
                    # MJLab has auto_reset=False. Its step() sets manual-reset pending
                    # on done and refuses another step until reset(env_ids) is called.
                    # Native delay buffers were created under inference mode by step().
                    # Their reset mutates those tensors, so reset must use the same mode.
                    with torch.inference_mode():
                        env.unwrapped.reset(env_ids=torch.tensor([0], dtype=torch.int64,
                                                                  device=args.device))
                    reset_events.append({"after_control_step": step + 1,
                                         "termination_terms": terms})
                    previous_raw = torch.zeros_like(previous_raw)
                    previous_applied = torch.zeros_like(previous_applied)
                    episode_step = 0
                else:
                    end_termination_terms = terms
                continue
            end_termination_terms = terms
            break
        episode_step += 1

    def window(start: float, end: float) -> dict:
        rows = [r for r in records if start <= r["time_s"] < end]
        if not rows:
            return None
        return {
            "mean_pelvis_z_m": round(sum(r["pelvis_z_m"] for r in rows) / len(rows), 4),
            "max_tilt_deg": round(max(r["tilt_deg"] for r in rows), 2),
            "hand_contact_frames": sum(r["hand_force_n"] >= 1.0 for r in rows),
            "mean_lowest_hand_height_m": round(sum(min(r["hand_height_m"])
                                                      for r in rows) / len(rows), 4),
            "body_contact_fraction": round(sum(r["body_contact"] for r in rows) / len(rows), 3),
            "both_feet_grounded_fraction": round(
                sum(all(r["feet_grounded"]) for r in rows) / len(rows), 3),
            "saturated_leg_fraction": round(
                sum(r["leg_torque_saturated"] for r in rows) / len(rows), 3),
            "mean_step_reward": round(sum(r["step_reward"] for r in rows) / len(rows), 4),
            "mean_leg_torque_nm": round(sum(r["mean_leg_torque_nm"] for r in rows) / len(rows), 3),
        }

    report = {
        "task": task_id,
        "body_xml": str(candidate_model or args.body_xml or args.cove50_body_xml or ape_walk.task.ELBOW_MODEL),
        "body_sha256": hashlib.sha256((candidate_model or args.body_xml or args.cove50_body_xml or ape_walk.task.ELBOW_MODEL).read_bytes()).hexdigest(),
        "physics_model": "cove50" if args.cove50_body_xml else args.physics_model,
        "leg_rated_effort_limits_nm": [round(float(v), 6) for v in leg_rated_effort],
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "evaluator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "intervention_route": args.intervention_route,
        "evaluation_common_step_counter_after_load": evaluation_counter,
        "action_process_scale": action_term._scale[0].detach().cpu().tolist(),
        "action_process_offset": action_term._offset[0].detach().cpu().tolist(),
        "action_process_clip": (action_term._clip[0].detach().cpu().tolist()
                                if action_term.cfg.clip is not None else None),
        "critic_checkpoint_loaded": True,
        "zero_actions": args.zero_actions,
        "scripted_target_probe": args.intervention_route == "scripted",
        "scripted_leg_override_rad": ([args.scripted_hip_pitch, args.scripted_knee,
                                        args.scripted_ankle_pitch]
                                       if args.scripted_hip_pitch is not None else None),
        "motor_rail_v": args.motor_rail_v if args.cove50_body_xml else None,
        "command_delay_ms": args.command_delay_ms if args.cove50_body_xml else None,
        "leg_target_margin_rad": args.leg_target_margin_rad,
        "preflight_sentinel": args.preflight_sentinel,
        "render_preflight": args.render_preflight,
        "timing_probe": args.timing_probe,
        "timing_route_action": "zero" if args.timing_probe else "bc_actor",
        "control_step_count": len(records),
        "render_frame_count": len(frames),
        "native_done_count": len(done_events),
        "manual_reset_count": len(reset_events),
        "native_done_events": done_events,
        "manual_reset_events": reset_events,
        "action_smoothing_alpha": args.action_smoothing_alpha,
        "squat_active_mask": args.squat_active_mask,
        "action_abs_bound": args.action_abs_bound,
        "wrapper_action_clip": action_clip,
        "expose_squat_blend": args.expose_squat_blend,
        "mask_action_history": args.mask_action_history,
        "start_squat": args.start_squat,
        "settled_squat_reset": args.settled_squat_reset,
        "mode": args.mode,
        "device": args.device,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "nvidia_driver_capabilities": os.environ.get("NVIDIA_DRIVER_CAPABILITIES"),
        "gpu_id": "NVIDIA A40",
        "nvidia_smi_name": gpu_names[0],
        "cuda_device_name": cuda_device_name,
        "hardware_shift": "fresh_A40_development_no_RTX4090_pooling",
        "recover_from_start": args.recover_from_start,
        "posture_ramp_s": args.posture_ramp_s,
        "initial_blend_override": args.initial_blend,
        "seated_rise_stage": args.seated_rise_stage,
        "randomized_physics": args.randomized_physics,
        "env_cfg_play": env_cfg_play,
        "actor_observation_corruption": cfg.observations["actor"].enable_corruption,
        "actor_observation_terms": list(cfg.observations["actor"].terms),
        "critic_observation_terms": list(cfg.observations["critic"].terms),
        "motor_delay_config": motor_delay_config,
        "observed_motor_delay_lags": sorted(observed_motor_delay_lags),
        "initial_physics": initial_physics,
        "initial_motors": initial_motors,
        "scripted_min_blend": args.scripted_min_blend,
        "end_termination_terms": end_termination_terms,
        "probe_joint": args.probe_joint,
        "probe_bias_action": args.probe_bias_action,
        "probe_window_s": [args.probe_start_s, args.probe_end_s],
        "seed": args.seed,
        "initial_state_sha256": initial_state_hash.hexdigest(),
        "initial_predicted_sole_gap_m": initial_predicted_sole_gap_m,
        "initial_sole_site_xy_m": initial_sole_xy.tolist(),
        "initial_qpos": initial_qpos.tolist(),
        "initial_qvel": initial_qvel.tolist(),
        "first_action_sha256": first_action_hash,
        "reward_term_names": reward_term_names if native_frames is not None else None,
        "termination_term_names": termination_terms,
        "native_record_alignment": "row i: pre-action time i*dt with corrupted actor and critic observations, command/rise phase, actual previous action; selected raw action at i; processed target and physical/reward/termination state after env.step at (i+1)*dt",
        "reward_term_weights": ([float(reward_manager.get_term_cfg(name).weight)
                                 for name in reward_term_names] if native_frames is not None else None),
        "reward_rate_units": "weighted reward per second; MJLab reward buffer multiplies by step dt",
        "control_dt_s": dt,
        "duration_s": records[-1]["time_s"],
        "ended_early": bool(records[-1]["time_s"] < args.duration_s - 0.01),
        "net_xy_m": [round(records[-1]["root_xy_m"][i] - records[0]["root_xy_m"][i], 4)
                     for i in range(2)],
        "net_yaw_rad": round(sum(math.atan2(
            math.sin(b["yaw_rad"] - a["yaw_rad"]),
            math.cos(b["yaw_rad"] - a["yaw_rad"]),
        ) for a, b in zip(records, records[1:])), 3),
        "initial_hold": window(1.0, 2.0),
        "middle_hold": window(5.0, 6.0) if args.mode == "cycle" else None,
        "final_hold": window(9.0, 10.0) if records[-1]["time_s"] >= 10.0 else None,
        "stand_before": window(1.0, 2.0) if not args.start_squat else None,
        "squat_hold": window(5.0, 6.0) if args.mode == "cycle" and not args.start_squat else None,
        "stand_after": window(9.0, 10.0) if not args.start_squat and records[-1]["time_s"] >= 10.0 else None,
        "timeseries": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.state_out:
        args.state_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.state_out, qpos=np.asarray(qpos_frames), dt=dt)
    if args.native_trajectory_out:
        args.native_trajectory_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.native_trajectory_out,
                            initial_qpos=initial_qpos, initial_qvel=initial_qvel,
                            **{name: np.asarray(frames) for name, frames in native_frames.items()})
    print(json.dumps({k: v for k, v in report.items() if k != "timeseries"}, indent=2))
    if args.video:
        import imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=25, macro_block_size=16)
        print("video:", args.video)
    env.close()


if __name__ == "__main__":
    main()
