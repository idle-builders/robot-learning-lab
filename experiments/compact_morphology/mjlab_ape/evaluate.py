"""Fresh-process closed-loop ape walk replay for a saved MJLab checkpoint."""

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

import ape_walk.task  # Register the task.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--task", choices=(
        ape_walk.task.TASK_ID, ape_walk.task.BACKWARD_TASK_ID,
        ape_walk.task.SIGNED_TASK_ID, ape_walk.task.SIDEWAYS_TASK_ID,
        ape_walk.task.PLANAR_TASK_ID,
        ape_walk.task.ELBOW_TASK_ID,
        ape_walk.task.CAD_V8_WALK_TASK_ID,
        ape_walk.task.ELBOW_BACKWARD_TASK_ID, ape_walk.task.ELBOW_SIDEWAYS_TASK_ID,
        ape_walk.task.ELBOW_PLANAR_TASK_ID,
        ape_walk.task.ELBOW_PLANAR_PROXY60_TASK_ID,
        ape_walk.task.ELBOW_PLANAR_NARROW55_TASK_ID,
    ),
                        default=ape_walk.task.TASK_ID)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--speed", type=float, default=None,
                        help="Legacy alias for --velocity-x")
    parser.add_argument("--velocity-x", type=float, default=None)
    parser.add_argument("--velocity-y", type=float, default=None)
    parser.add_argument("--angular-z", type=float, default=None,
                        help="Yaw-rate command in radians per second")
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--state-out", type=Path,
                        help="Save world-zero full poses for self-contact audit")
    parser.add_argument("--diagnostic-trace", type=Path,
                        help="Save synchronized native per-step effort, velocity, targets and contacts")
    parser.add_argument("--imitation-trace", type=Path,
                        help="Save pre-action actor observations and full policy actions for whole-episode imitation audit")
    parser.add_argument("--zero-action", action="store_true",
                        help="Replace actor output with zero offsets for a passive control baseline")
    parser.add_argument("--body-xml", type=Path,
                        help="Use an exact-geometry collision variant for closed-loop replay")
    parser.add_argument("--initial-root-z", type=float,
                        help="Pelvis reset height in metres for a different body")
    parser.add_argument("--yaw-only-head", action="store_true",
                        help="Remove the absent head-pitch action for a 21-axis body")
    parser.add_argument("--arms-down", action="store_true",
                        help="Use a neutral arms-down gait reset on the longer ape body")
    parser.add_argument("--physics-model", choices=("baseline", "candidate", "frame50"), default="baseline",
                        help="Use ideal XML servos, historical 22-axis candidate, or selected 21-axis motor envelope")
    parser.add_argument("--candidate-narrow55", action="store_true",
                        help="Use the 55 mm sole with candidate physics (separate live-dynamics screen)")
    parser.add_argument("--motor-rail-v", type=float, default=12.0,
                        help="Feetech rail voltage for candidate sensitivity (default 12 V)")
    parser.add_argument("--ankle-motor-model", choices=("selected", "st3250"),
                        default="selected",
                        help="Catalog motor-law-only ankle sensitivity on frame50; no mass or CAD change")
    parser.add_argument("--swing-hip-roll-inward-rad", type=float, default=0.0,
                        help="Replay-only contact-timed inward swing-foot target on the selected frame")
    parser.add_argument("--command-delay-ms", type=float, default=0.0,
                        help="Candidate command-lag sensitivity; unmeasured, default 0 ms")
    parser.add_argument("--action-limit-margin-rad", type=float,
                        help="Clip leg position targets inside the selected Cove joint limits")
    parser.add_argument("--ankle-target-margin-rad", type=float,
                        help="Clip only ankle position targets inside the selected Cove joint limits")
    parser.add_argument("--contact-slide-friction", type=float,
                        help="Override MJLab's imported foot/hand sliding friction for sensitivity replay")
    parser.add_argument("--physics-dt", type=float, choices=(0.002, 0.004, 0.005))
    parser.add_argument("--legacy-policy-on-elbow", action="store_true",
                        help="Replay the saved 20-DOF actor on the 22-DOF body by joint name")
    parser.add_argument("--stop-at-s", type=float,
                        help="Switch the velocity command to zero during the same replay")
    parser.add_argument("--switch-at-s", type=float,
                        help="Switch to another planar command without resetting the scene")
    parser.add_argument("--switch-velocity-x", type=float)
    parser.add_argument("--switch-velocity-y", type=float)
    parser.add_argument("--switch-angular-z", type=float)
    parser.add_argument("--duration-s", type=float, default=8.0,
                        help="Replay duration in seconds (default 8)")
    parser.add_argument("--stop-policy-checkpoint", type=Path,
                        help="Use a second saved 20-DOF actor after --stop-at-s")
    args = parser.parse_args()
    legacy_task_for_elbow = {
        ape_walk.task.ELBOW_TASK_ID: ape_walk.task.TASK_ID,
        ape_walk.task.CAD_V8_WALK_TASK_ID: ape_walk.task.TASK_ID,
        ape_walk.task.ELBOW_BACKWARD_TASK_ID: ape_walk.task.BACKWARD_TASK_ID,
        ape_walk.task.ELBOW_SIDEWAYS_TASK_ID: ape_walk.task.SIDEWAYS_TASK_ID,
        ape_walk.task.ELBOW_PLANAR_TASK_ID: ape_walk.task.PLANAR_TASK_ID,
        ape_walk.task.ELBOW_PLANAR_PROXY60_TASK_ID: ape_walk.task.PLANAR_TASK_ID,
        ape_walk.task.ELBOW_PLANAR_NARROW55_TASK_ID: ape_walk.task.PLANAR_TASK_ID,
    }
    if args.legacy_policy_on_elbow and args.task not in legacy_task_for_elbow:
        parser.error("--legacy-policy-on-elbow requires the forward or backward elbow task")
    if args.duration_s <= 0:
        parser.error("--duration-s must be positive")
    if args.stop_at_s is not None and not 0 < args.stop_at_s < args.duration_s:
        parser.error("--stop-at-s must be between 0 and --duration-s")
    switched = (args.switch_at_s, args.switch_velocity_x,
                args.switch_velocity_y, args.switch_angular_z)
    if any(value is not None for value in switched):
        if any(value is None for value in switched):
            parser.error("--switch-at-s requires all three switch command values")
        if not 0 < args.switch_at_s < args.duration_s:
            parser.error("--switch-at-s must be between 0 and --duration-s")
        if args.stop_at_s is not None:
            parser.error("Use either --stop-at-s or --switch-at-s")
        if math.hypot(args.switch_velocity_x, args.switch_velocity_y) == 0 and args.switch_angular_z == 0:
            parser.error("A zero successor command belongs to --stop-at-s")
    if args.stop_policy_checkpoint and (args.stop_at_s is None or not args.legacy_policy_on_elbow):
        parser.error("--stop-policy-checkpoint requires --stop-at-s and --legacy-policy-on-elbow")

    if args.physics_model == "candidate" and args.body_xml:
        parser.error("--body-xml and --physics-model candidate are mutually exclusive")
    if args.physics_model == "frame50" and not args.body_xml:
        parser.error("--physics-model frame50 requires the selected --body-xml")
    if args.action_limit_margin_rad is not None and args.physics_model != "frame50":
        parser.error("--action-limit-margin-rad requires --physics-model frame50")
    if args.ankle_target_margin_rad is not None and args.physics_model != "frame50":
        parser.error("--ankle-target-margin-rad requires --physics-model frame50")
    if args.action_limit_margin_rad is not None and args.ankle_target_margin_rad is not None:
        parser.error("Choose either all-leg or ankle-only target clipping")
    if args.candidate_narrow55 and args.physics_model != "candidate":
        parser.error("--candidate-narrow55 requires --physics-model candidate")
    if args.ankle_motor_model != "selected" and args.physics_model != "frame50":
        parser.error("--ankle-motor-model st3250 requires --physics-model frame50")
    if args.swing_hip_roll_inward_rad and (args.physics_model != "frame50" or
                                           not 0.0 <= args.swing_hip_roll_inward_rad <= 0.10):
        parser.error("--swing-hip-roll-inward-rad requires frame50 and 0–0.10 rad")
    cfg = load_env_cfg(args.task, play=True)
    if args.physics_dt:
        cfg.sim.mujoco.timestep = args.physics_dt
        cfg.decimation = round(0.02 / args.physics_dt)
    candidate_model = None
    if args.physics_model == "candidate":
        from ape_walk.physics.candidate import MODEL, NARROW55_MODEL, configure_candidate

        candidate_model = NARROW55_MODEL if args.candidate_narrow55 else MODEL
        cfg = configure_candidate(
            cfg, motor_rail_v=args.motor_rail_v,
            delay_s=args.command_delay_ms / 1000.0,
            model_path=candidate_model,
        )
    if args.body_xml:
        cfg.scene.entities["robot"].spec_fn = partial(ape_walk.task._ape_spec, args.body_xml.resolve())
        # Keep the supplied XML's body contact masks. The default selective
        # sole/hand CollisionCfg otherwise erases non-listed collision geoms.
        cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
    if args.contact_slide_friction is not None:
        if args.contact_slide_friction <= 0:
            parser.error("--contact-slide-friction must be positive")
        cfg.scene.entities["robot"].collisions[0].friction = (args.contact_slide_friction,)
    if args.physics_model == "frame50":
        from ape_walk.physics.frame50 import (clip_frame50_leg_targets,
                                              configure_frame50_motors,
                                              give_soles_contact_priority,
                                              reveal_frame50_proxy_geometry_for_video)

        cfg = configure_frame50_motors(
            cfg, args.body_xml, rail_v=args.motor_rail_v,
            delay_s=args.command_delay_ms / 1000.0,
            ankle_motor_model=args.ankle_motor_model,
        )
        # Match the Cove trainer: sampled/overridden sole friction must win
        # against the generated terrain contact, including low-grip replays.
        give_soles_contact_priority(cfg)
        if args.action_limit_margin_rad is not None:
            clip_frame50_leg_targets(cfg, args.body_xml,
                                           margin_rad=args.action_limit_margin_rad)
        if args.ankle_target_margin_rad is not None:
            clip_frame50_leg_targets(cfg, args.body_xml,
                                     margin_rad=args.ankle_target_margin_rad,
                                     axes=("ankle_pitch",))
    if args.physics_model == "frame50":
        # Catalog-rated limits vary by joint and by rail voltage.  A fixed
        # 1.56 Nm gate would count weak-rail episodes as unsaturated even when
        # every hip/knee motor is clipping at its lower 10 V rating.
        leg_rated_effort = np.asarray([
            act.effort_limit for act in
            cfg.scene.entities["robot"].articulation.actuators[:10]
        ], dtype=float)
    else:
        leg_rated_effort = np.full(10, 1.56, dtype=float)
    if args.initial_root_z is not None:
        cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, args.initial_root_z)
    if args.yaw_only_head:
        cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
    if args.arms_down:
        pose = cfg.scene.entities["robot"].init_state.joint_pos
        pose[r".*shoulder_pitch.*"] = 0.0
        pose[r".*elbow_pitch.*"] = 0.0
    registered_body = (ape_walk.task.CAD_V8_MODEL if args.task == ape_walk.task.CAD_V8_WALK_TASK_ID
                       else ape_walk.task.ELBOW_MODEL if "Elbows" in args.task
                       else ape_walk.task.MODEL)
    active_body = candidate_model or args.body_xml or registered_body
    contact_variant = bool(args.body_xml or candidate_model or
                           args.task == ape_walk.task.CAD_V8_WALK_TASK_ID)
    if contact_variant:
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
    cfg.scene.num_envs = args.worlds
    cfg.seed = args.seed
    cfg.episode_length_s = args.duration_s
    cfg.auto_reset = False
    cfg.commands["twist"].rel_standing_envs = 0.0
    command = cfg.commands["twist"]
    vx = args.velocity_x if args.velocity_x is not None else args.speed
    if vx is None:
        vx = sum(command.ranges.lin_vel_x) / 2
    vy = args.velocity_y
    if vy is None:
        vy = sum(command.ranges.lin_vel_y) / 2
    omega = args.angular_z
    if omega is None:
        omega = sum(command.ranges.ang_vel_z) / 2
    if math.hypot(vx, vy) == 0 and omega == 0 and not args.zero_action:
        raise ValueError("Replay needs a nonzero commanded planar or angular velocity")
    command.ranges.lin_vel_x = (vx, vx)
    command.ranges.lin_vel_y = (vy, vy)
    command.ranges.ang_vel_z = (omega, omega)
    if hasattr(command, "rel_turn_in_place_envs"):
        command.rel_turn_in_place_envs = 0.0  # Keep replay commands exact.
    if args.switch_at_s is not None:
        command.rel_forward_envs = 0.0
        command.rel_heading_envs = 0.0
        command.rel_world_envs = 0.0
    if args.stop_at_s is not None or args.switch_at_s is not None:
        command.resampling_time_range = (100.0, 100.0)
    command_norm = math.hypot(vx, vy)
    along_x, along_y = ((vx / command_norm, vy / command_norm)
                        if command_norm else (0.0, 0.0))
    if args.video:
        if command_norm == 0 or args.switch_at_s is not None:
            # A body-following camera rotates with the ape and conceals turns.
            cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
            cfg.viewer.lookat = (0.0, 0.0, 0.18)
            cfg.viewer.distance = 0.8
            cfg.viewer.elevation = -18.0
            cfg.viewer.azimuth = 45.0
        else:
            cfg.viewer.origin_type = cfg.viewer.OriginType.ASSET_BODY
            cfg.viewer.entity_name = "robot"
            cfg.viewer.body_name = "pelvis"
            cfg.viewer.distance = 1.3
            cfg.viewer.elevation = -12.0
            cfg.viewer.azimuth = 90.0
        cfg.viewer.width = 640
        cfg.viewer.height = 480
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(
            cfg=cfg, device="cuda:0",
            render_mode="rgb_array" if args.video else None,
        ),
        clip_actions=load_rl_cfg(args.task).clip_actions,
    )
    if args.video and args.physics_model == "frame50":
        reveal_frame50_proxy_geometry_for_video(env.unwrapped.sim.mj_model)
    runner_task = legacy_task_for_elbow[args.task] if args.legacy_policy_on_elbow else args.task
    legacy_env = None
    if args.legacy_policy_on_elbow:
        legacy_cfg = load_env_cfg(runner_task, play=True)
        legacy_cfg.scene.num_envs = args.worlds
        legacy_cfg.seed = args.seed
        legacy_env = RslRlVecEnvWrapper(
            ManagerBasedRlEnv(cfg=legacy_cfg, device="cuda:0"),
            clip_actions=load_rl_cfg(runner_task).clip_actions,
        )
    agent_cfg = load_rl_cfg(runner_task)
    runner_cls = load_runner_cls(runner_task)
    runner = runner_cls(legacy_env or env, asdict(agent_cfg), device="cuda:0")
    runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    native_policy = runner.get_inference_policy(device="cuda:0")
    stop_native_policy = None
    if args.stop_policy_checkpoint:
        stop_runner = runner_cls(
            legacy_env, asdict(load_rl_cfg(ape_walk.task.TASK_ID)), device="cuda:0"
        )
        stop_runner.load(str(args.stop_policy_checkpoint), load_cfg={"actor": True},
                         strict=True, map_location="cuda:0")
        stop_native_policy = stop_runner.get_inference_policy(device="cuda:0")
    stop_roots = None
    switch_roots = None
    switch_yaw_unwrapped = None

    if args.legacy_policy_on_elbow:
        def names(model_path):
            model = mujoco.MjModel.from_xml_path(str(model_path))
            joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                      for i in range(1, model.njnt)]
            actions = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                       int(model.actuator_trnid[i, 0])) for i in range(model.nu)]
            return joints, actions

        old_joints, old_actions = names(ape_walk.task.MODEL)
        new_joints, new_actions = names(ape_walk.task.ELBOW_MODEL)
        assert len(old_joints) == len(old_actions) == 20
        assert len(new_joints) == len(new_actions) == 22
        assert set(new_joints) - set(old_joints) == {"left_elbow_pitch", "right_elbow_pitch"}
        joint_take = [new_joints.index(name) for name in old_joints]
        action_take = [new_actions.index(name) for name in old_actions]

        def policy(observations):
            actor = observations["actor"]
            assert actor.shape == (args.worlds, 75), actor.shape
            old_actor = torch.cat((
                actor[:, :6],
                actor[:, 6:28][:, joint_take],
                actor[:, 28:50][:, joint_take],
                actor[:, 50:72][:, action_take],
                actor[:, 72:75],
            ), dim=1)
            assert old_actor.shape == (args.worlds, 69)
            old_observations = legacy_env.get_observations()
            old_observations["actor"] = old_actor
            old_output = (stop_native_policy if stop_roots is not None and stop_native_policy
                          else native_policy)(old_observations)
            output = torch.zeros((args.worlds, 22), device=old_output.device,
                                 dtype=old_output.dtype)
            output[:, action_take] = old_output
            return output
    else:
        policy = native_policy
    n = args.worlds
    robot = env.unwrapped.scene["robot"]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    if args.zero_action:
        policy = lambda observations: torch.zeros(
            (args.worlds, len(action_term.target_names)), device="cuda:0")
    leg_target_bounds = None
    if args.physics_model == "frame50":
        body_model = mujoco.MjModel.from_xml_path(str(active_body))
        leg_target_bounds = np.asarray([
            body_model.jnt_range[mujoco.mj_name2id(
                body_model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in action_term.target_names[:10]
        ])
    sensors = env.unwrapped.scene.sensors
    swing_roll_offset = None
    if args.swing_hip_roll_inward_rad:
        if action_term.target_names[1] != "left_hip_roll" or action_term.target_names[6] != "right_hip_roll":
            raise ValueError("Unexpected hip-roll action order for swing-foot correction")
        swing_roll_offset = torch.zeros((n, 2), device=robot.data.joint_pos.device)
    initial_x = robot.data.root_link_pose_w[:, 0].cpu().tolist()
    initial_y = robot.data.root_link_pose_w[:, 1].cpu().tolist()
    def quat_yaw(quat):
        qw, qx, qy, qz = quat
        return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    previous_yaw = [quat_yaw(pose[3:7]) for pose in
                    robot.data.root_link_pose_w.cpu().tolist()]
    yaw_unwrapped = [0.0] * args.worlds
    stop_yaw_unwrapped = None
    initial_state_hash = hashlib.sha256()
    for tensor in (
        env.unwrapped.sim.data.qpos,
        env.unwrapped.sim.data.qvel,
        env.unwrapped.command_manager.get_command("twist"),
    ):
        initial_state_hash.update(tensor.cpu().numpy().tobytes())
    first_action_hash = None
    first_action_world0 = None
    active = [True] * n
    air_run = [[0, 0] for _ in range(n)]
    touchdowns = [[] for _ in range(n)]
    post_stop_touchdowns = [[] for _ in range(n)]
    late_post_stop_touchdowns = [[] for _ in range(n)]
    late_speed_sum = [0.0] * n
    late_yaw_speed_sum = [0.0] * n
    late_double_support_frames = [0] * n
    late_frames = [0] * n
    slip = [[0.0, 0.0] for _ in range(n)]
    peak_lift = [[0.0, 0.0] for _ in range(n)]
    peak_effort = [0.0] * n
    peak_hand_force = [0.0] * n
    peak_forward = [0.0] * n
    peak_progress = [0.0] * n
    peak_abs_yaw = [0.0] * n
    first_meter = [None] * n
    saturated_frames = [0] * n
    moving_frames = [0] * n
    moving_saturated_frames = [0] * n
    body_contact_frames = [0] * n
    saturated_by_joint = [[0] * 10 for _ in range(n)]
    moving_saturated_by_joint = [[0] * 10 for _ in range(n)]
    out_of_range_target_by_joint = [[0] * 10 for _ in range(n)]
    action_abs_sum = [[0.0] * 10 for _ in range(n)]
    records = [None] * n
    frames = []
    qpos_frames = []
    diagnostic_frames = []
    imitation_frames = []
    dt = env.unwrapped.step_dt

    for step in range(env.unwrapped.max_episode_length):
        if args.switch_at_s is not None and switch_roots is None and step * dt >= args.switch_at_s:
            switch_roots = robot.data.root_link_pose_w[:, :3].cpu().tolist()
            switch_yaw_unwrapped = yaw_unwrapped.copy()
            term = env.unwrapped.command_manager.get_term("twist")
            term.command[:, 0] = args.switch_velocity_x
            term.command[:, 1] = args.switch_velocity_y
            term.command[:, 2] = args.switch_angular_z
            term.is_standing_env[:] = False
        if args.stop_at_s is not None and stop_roots is None and step * dt >= args.stop_at_s:
            stop_roots = robot.data.root_link_pose_w[:, :3].cpu().tolist()
            stop_yaw_unwrapped = yaw_unwrapped.copy()
            term = env.unwrapped.command_manager.get_term("twist")
            term.command.zero_()
            term.is_standing_env[:] = True
        # MJLab's delayed actuator buffer is reset outside this block when a
        # vector world ends. Inference tensors reject that in-place reset.
        with torch.no_grad():
            command_now = env.unwrapped.command_manager.get_term("twist").command.cpu().numpy().copy()
            observations = env.get_observations()
            if args.imitation_trace:
                actor_observation = observations["actor"].cpu().numpy().copy()
                pre_action_active = np.asarray(active, dtype=bool)
                pre_action_qpos = env.unwrapped.sim.data.qpos.cpu().numpy().copy()
                pre_action_qvel = env.unwrapped.sim.data.qvel.cpu().numpy().copy()
            actions = policy(observations)
            if swing_roll_offset is not None:
                feet_now = sensors["feet_ground_contact"].data.found > 0
                desired = torch.stack((
                    -(~feet_now[:, 0] & feet_now[:, 1]).to(actions.dtype),
                    (feet_now[:, 0] & ~feet_now[:, 1]).to(actions.dtype),
                ), dim=1) * args.swing_hip_roll_inward_rad
                # A fixed 40 ms filter avoids a target jump at toe-off and touchdown.
                alpha = dt / (0.04 + dt)
                swing_roll_offset += alpha * (desired - swing_roll_offset)
                actions = actions.clone()
                actions[:, 1] += swing_roll_offset[:, 0] / action_term.scale[..., 1]
                actions[:, 6] += swing_roll_offset[:, 1] / action_term.scale[..., 6]
            requested_leg_targets = (
                (actions * action_term.scale + action_term.offset)[:, :10].cpu().numpy()
                if leg_target_bounds is not None else None
            )
            if step == 0:
                first_action_hash = hashlib.sha256(actions.cpu().numpy().tobytes()).hexdigest()
                first_action_world0 = actions[0].cpu().tolist()
            if args.imitation_trace:
                imitation_frames.append({
                    "actor_observation": actor_observation,
                    "policy_action": actions.cpu().numpy().copy(),
                    "command": command_now,
                    "active": pre_action_active,
                    "qpos": pre_action_qpos,
                    "qvel": pre_action_qvel,
                })
            _, _, dones, _ = env.step(actions)
        if args.state_out and active[0]:
            qpos_frames.append(np.concatenate((
                robot.data.root_link_pose_w[0].cpu().numpy(),
                robot.data.joint_pos[0].cpu().numpy(),
            )))
        if args.video and step % 2 == 0:
            frames.append(env.unwrapped.render())
        roots = robot.data.root_link_pose_w.cpu().tolist()
        feet = sensors["feet_ground_contact"].data.found.cpu().tolist()
        body_contacts = (sensors["self_body_contact"].data.found.flatten(start_dim=1).any(dim=1).cpu().tolist()
                         if contact_variant else [False] * n)
        hands = sensors["hand_ground_contact"].data.force.norm(dim=-1).amax(dim=-1).cpu().tolist()
        heights = sensors["foot_height_scan"].data.frame_pos_w[:, :, 2].cpu().tolist()
        efforts = robot.data.actuator_force[:, :10].abs().cpu().numpy()
        action_magnitudes = actions[:, :10].abs().cpu().tolist()
        sole_velocities = [
            sensors[f"robot/{side}_sole_center_global_linvel"].data[:, :2].norm(dim=-1).cpu().tolist()
            for side in ("left", "right")
        ]
        if args.diagnostic_trace:
            diagnostic_frames.append({
                "time_s": (step + 1) * dt,
                "command": command_now,
                "active": np.asarray(active, dtype=bool),
                "root_pose": robot.data.root_link_pose_w.cpu().numpy().copy(),
                "root_velocity": robot.data.root_link_lin_vel_w.cpu().numpy().copy(),
                "joint_pos": robot.data.joint_pos[:, :10].cpu().numpy().copy(),
                "joint_vel": robot.data.joint_vel[:, :10].cpu().numpy().copy(),
                "actuator_force": robot.data.actuator_force[:, :10].cpu().numpy().copy(),
                "raw_action": actions[:, :10].cpu().numpy().copy(),
                "unclipped_position_target": (requested_leg_targets.copy() if requested_leg_targets is not None
                                              else np.full((n, 10), np.nan)),
                "sole_found": np.asarray(feet),
                "sole_contact_force": sensors["feet_ground_contact"].data.force.cpu().numpy().copy(),
                "sole_speed": np.asarray(sole_velocities).T,
                "hand_force": np.asarray(hands),
                "body_contact": np.asarray(body_contacts, dtype=bool),
            })
        finished = []
        for i in range(n):
            if not active[i]:
                continue
            peak_effort[i] = max(peak_effort[i], float(efforts[i].max()))
            peak_hand_force[i] = max(peak_hand_force[i], hands[i])
            peak_forward[i] = max(peak_forward[i], roots[i][0] - initial_x[i])
            dx, dy = roots[i][0] - initial_x[i], roots[i][1] - initial_y[i]
            progress = dx * along_x + dy * along_y
            cross_track = -dx * along_y + dy * along_x
            peak_progress[i] = max(peak_progress[i], progress)
            yaw = quat_yaw(roots[i][3:7])
            yaw_delta = math.atan2(math.sin(yaw - previous_yaw[i]),
                                   math.cos(yaw - previous_yaw[i]))
            yaw_unwrapped[i] += yaw_delta
            previous_yaw[i] = yaw
            peak_abs_yaw[i] = max(peak_abs_yaw[i], abs(yaw))
            clipped_legs = efforts[i] >= leg_rated_effort * 0.995
            saturated_frames[i] += int(bool(clipped_legs.any()))
            if stop_roots is None:
                moving_frames[i] += 1
                moving_saturated_frames[i] += int(bool(clipped_legs.any()))
            body_contact_frames[i] += int(body_contacts[i])
            for j, clipped in enumerate(clipped_legs.tolist()):
                saturated_by_joint[i][j] += int(clipped)
                if stop_roots is None:
                    moving_saturated_by_joint[i][j] += int(clipped)
                action_abs_sum[i][j] += action_magnitudes[i][j]
                if requested_leg_targets is not None:
                    target = requested_leg_targets[i, j]
                    out_of_range_target_by_joint[i][j] += int(
                        target < leg_target_bounds[j, 0] or
                        target > leg_target_bounds[j, 1]
                    )
            for side in (0, 1):
                peak_lift[i][side] = max(peak_lift[i][side], heights[i][side])
                if feet[i][side] > 0:
                    slip[i][side] += sole_velocities[side][i] * dt
                    if air_run[i][side] * dt >= 0.08:
                        touchdowns[i].append(side)
                        if stop_roots is not None:
                            post_stop_touchdowns[i].append(side)
                            if (step + 1) * dt >= args.stop_at_s + 2.0:
                                late_post_stop_touchdowns[i].append(side)
                    air_run[i][side] = 0
                else:
                    air_run[i][side] += 1
            if stop_roots is not None and (step + 1) * dt >= args.stop_at_s + 2.0:
                late_frames[i] += 1
                late_speed_sum[i] += float(robot.data.root_link_lin_vel_w[i, :2].norm())
                late_yaw_speed_sum[i] += abs(yaw_delta) / dt
                late_double_support_frames[i] += int(feet[i][0] > 0 and feet[i][1] > 0)
            if first_meter[i] is None and progress >= 1.0:
                seq = touchdowns[i]
                first_meter[i] = {
                    "time_s": round((step + 1) * dt, 3),
                    "lateral_m": round(cross_track, 4),
                    "yaw_rad": round(yaw, 3),
                    "touchdowns_left_right": [seq.count(0), seq.count(1)],
                    "alternating_touchdowns": sum(a != b for a, b in zip(seq, seq[1:])),
                    "contact_slip_m_left_right": [round(x, 3) for x in slip[i]],
                    "peak_sole_height_m_left_right": [round(x, 3) for x in peak_lift[i]],
                    "hand_force_n": round(peak_hand_force[i], 3),
                    "fraction_frames_with_saturated_leg": round(saturated_frames[i] / (step + 1), 3),
                    "fraction_moving_frames_with_saturated_leg": round(moving_saturated_frames[i] / max(1, moving_frames[i]), 3),
                    "fraction_frames_with_body_contact": (round(body_contact_frames[i] / (step + 1), 3)
                                                          if contact_variant else None),
                }
            if bool(dones[i]) or step + 1 == env.unwrapped.max_episode_length:
                active[i] = False
                finished.append(i)
                seq = touchdowns[i]
                records[i] = {
                    "world": i,
                    "duration_s": round((step + 1) * dt, 3),
                    "forward_m": round(roots[i][0] - initial_x[i], 4),
                    "peak_forward_m": round(peak_forward[i], 4),
                    "command_progress_m": round(progress, 4),
                    "peak_command_progress_m": round(peak_progress[i], 4),
                    "first_meter": first_meter[i],
                    "lateral_m": round(roots[i][1] - initial_y[i], 4),
                    "cross_track_m": round(cross_track, 4),
                    "root_height_m": round(roots[i][2], 4),
                    "final_yaw_rad": round(yaw, 3),
                    "net_yaw_rad": round(yaw_unwrapped[i], 3),
                    "command_yaw_progress_rad": round(
                        ((yaw_unwrapped[i] - switch_yaw_unwrapped[i])
                         * (1 if args.switch_angular_z >= 0 else -1)
                         if switch_roots is not None else
                         yaw_unwrapped[i] * (1 if omega >= 0 else -1)), 3),
                    "target_yaw_change_rad": round(
                        (omega * min((step + 1) * dt, args.switch_at_s)
                         + args.switch_angular_z * max(0, (step + 1) * dt - args.switch_at_s)
                         if args.switch_at_s is not None else
                         omega * min((step + 1) * dt,
                                     args.stop_at_s if args.stop_at_s is not None
                                     else args.duration_s)), 3),
                    "peak_abs_yaw_rad": round(peak_abs_yaw[i], 3),
                    "hand_force_n": round(peak_hand_force[i], 3),
                    "touchdowns_left_right": [seq.count(0), seq.count(1)],
                    "touchdown_sequence": "".join("LR"[side] for side in seq),
                    "alternating_touchdowns": sum(a != b for a, b in zip(seq, seq[1:])),
                    "contact_slip_m_left_right": [round(x, 3) for x in slip[i]],
                    "peak_sole_height_m_left_right": [round(x, 3) for x in peak_lift[i]],
                    "peak_leg_actuator_force_n_m": round(peak_effort[i], 3),
                    "fraction_frames_with_saturated_leg": round(saturated_frames[i] / (step + 1), 3),
                    "fraction_moving_frames_with_saturated_leg": round(moving_saturated_frames[i] / max(1, moving_frames[i]), 3),
                    "fraction_frames_with_body_contact": (round(body_contact_frames[i] / (step + 1), 3)
                                                          if contact_variant else None),
                    "saturation_fraction_by_leg_joint": [
                        round(count / (step + 1), 3) for count in saturated_by_joint[i]
                    ],
                    "moving_saturation_fraction_by_leg_joint": [
                        round(count / max(1, moving_frames[i]), 3)
                        for count in moving_saturated_by_joint[i]
                    ],
                    "raw_out_of_range_target_fraction_by_leg_joint": (
                        [round(count / (step + 1), 3)
                         for count in out_of_range_target_by_joint[i]]
                        if leg_target_bounds is not None else None
                    ),
                    "mean_abs_raw_action_by_leg_joint": [
                        round(value / (step + 1), 3) for value in action_abs_sum[i]
                    ],
                    "ended_by_timeout": bool(env.unwrapped.reset_time_outs[i]),
                }
                if stop_roots is not None:
                    records[i]["stop"] = {
                        "command_time_s": args.stop_at_s,
                        "position_at_command_m": [round(v, 4) for v in stop_roots[i]],
                        "travel_after_command_m": round(
                            math.hypot(roots[i][0] - stop_roots[i][0],
                                       roots[i][1] - stop_roots[i][1]), 4),
                        "yaw_after_command_rad": round(
                            yaw_unwrapped[i] - stop_yaw_unwrapped[i], 3),
                        "final_planar_speed_m_s": round(float(
                            robot.data.root_link_lin_vel_w[i, :2].norm()), 4),
                        "late_mean_planar_speed_m_s": round(
                            late_speed_sum[i] / max(1, late_frames[i]), 4),
                        "late_mean_abs_yaw_speed_rad_s": round(
                            late_yaw_speed_sum[i] / max(1, late_frames[i]), 4),
                        "late_double_support_fraction": round(
                            late_double_support_frames[i] / max(1, late_frames[i]), 3),
                        "touchdowns_after_command": "".join(
                            "LR"[side] for side in post_stop_touchdowns[i]),
                        "touchdowns_in_last_second": "".join(
                            "LR"[side] for side in late_post_stop_touchdowns[i]),
                    }
                if switch_roots is not None:
                    records[i]["switch"] = {
                        "command_time_s": args.switch_at_s,
                        "command_velocity_m_s": [args.switch_velocity_x, args.switch_velocity_y],
                        "command_angular_rad_s": args.switch_angular_z,
                        "position_at_command_m": [round(v, 4) for v in switch_roots[i]],
                        "yaw_at_command_rad": round(switch_yaw_unwrapped[i], 3),
                        "travel_after_command_m": round(math.hypot(
                            roots[i][0] - switch_roots[i][0],
                            roots[i][1] - switch_roots[i][1]), 4),
                        "yaw_after_command_rad": round(
                            yaw_unwrapped[i] - switch_yaw_unwrapped[i], 3),
                    }
        if len(finished) == n or not any(active):
            break
        # The vector environment still steps worlds whose first episode was
        # already recorded. Reset every ended world, including inactive ones,
        # or a long surviving world can trip MJLab's auto_reset=False guard.
        reset_ids = torch.nonzero(dones).flatten()
        if reset_ids.numel():
            env.unwrapped.reset(env_ids=reset_ids)
            if swing_roll_offset is not None:
                swing_roll_offset[reset_ids] = 0

    report = {
        "task": args.task,
        "body_xml": str(active_body),
        "body_sha256": hashlib.sha256(active_body.read_bytes()).hexdigest(),
        "initial_root_z_m": args.initial_root_z,
        "yaw_only_head": args.yaw_only_head,
        "arms_down": args.arms_down,
        "preserve_body_xml_collision_masks": contact_variant,
        "physics_model": args.physics_model,
        "leg_rated_effort_limits_nm": [round(v, 6) for v in leg_rated_effort],
        "candidate_motor_rail_v": args.motor_rail_v if args.physics_model != "baseline" else None,
        "ankle_motor_model": args.ankle_motor_model if args.physics_model == "frame50" else None,
        "swing_hip_roll_inward_rad": args.swing_hip_roll_inward_rad,
        "swing_hip_roll_filter_s": 0.04 if args.swing_hip_roll_inward_rad else None,
        "candidate_command_delay_ms": args.command_delay_ms if args.physics_model != "baseline" else None,
        "action_limit_margin_rad": args.action_limit_margin_rad,
        "ankle_target_margin_rad": args.ankle_target_margin_rad,
        "imported_contact_slide_friction": args.contact_slide_friction,
        "legacy_policy_on_elbow": args.legacy_policy_on_elbow,
        "checkpoint": str(args.checkpoint),
        "speed_command_m_s": vx,
        "velocity_command_m_s": [vx, vy],
        "angular_command_rad_s": omega,
        "stop_at_s": args.stop_at_s,
        "switch_at_s": args.switch_at_s,
        "switch_velocity_command_m_s": ([args.switch_velocity_x, args.switch_velocity_y]
                                        if args.switch_at_s is not None else None),
        "switch_angular_command_rad_s": args.switch_angular_z,
        "episode_length_s": args.duration_s,
        "stop_policy_checkpoint": str(args.stop_policy_checkpoint) if args.stop_policy_checkpoint else None,
        "physics_dt_s": cfg.sim.mujoco.timestep,
        "seed": args.seed,
        "zero_action": args.zero_action,
        "diagnostic_trace": str(args.diagnostic_trace) if args.diagnostic_trace else None,
        "imitation_trace": str(args.imitation_trace) if args.imitation_trace else None,
        "initial_state_sha256": initial_state_hash.hexdigest(),
        "first_action_sha256": first_action_hash,
        "first_action_world0": first_action_world0,
        "worlds": records,
        "one_meter_with_alternating_steps": sum(
            r["command_progress_m"] >= 1.0 and r["alternating_touchdowns"] >= 2
            and r["hand_force_n"] < 1.0 for r in records
        ),
    }
    if args.diagnostic_trace:
        args.diagnostic_trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.diagnostic_trace,
            **{key: np.stack([frame[key] for frame in diagnostic_frames])
               for key in diagnostic_frames[0]},
            leg_names=np.asarray(action_term.target_names[:10]),
            rated_effort_limits_nm=leg_rated_effort,
            step_dt_s=dt,
        )
    if args.imitation_trace:
        args.imitation_trace.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.imitation_trace,
            **{key: np.stack([frame[key] for frame in imitation_frames])
               for key in imitation_frames[0]},
            action_names=np.asarray(action_term.target_names),
            step_dt_s=dt,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.state_out:
        args.state_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.state_out, qpos=np.asarray(qpos_frames), dt=dt)
    print(json.dumps({k: v for k, v in report.items() if k != "worlds"}, indent=2))
    print("forward_m:", [r["forward_m"] for r in records])
    print("alternating_touchdowns:", [r["alternating_touchdowns"] for r in records])
    if args.video:
        import imageio

        args.video.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(args.video, frames, fps=25, macro_block_size=16)
        print("video:", args.video)
    env.close()
    if legacy_env is not None:
        legacy_env.close()


if __name__ == "__main__":
    main()
