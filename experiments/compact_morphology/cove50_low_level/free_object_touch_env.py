"""Experimental exact-body, observed-RGB free-object touch lesson.

Touch means sustained physical fingertip/cube contact while the cube remains
free and floor supported. Opposing-pad possession and lift are separate tasks.
Run --preflight-only locally before any training; it claims setup only.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg

import ape_walk.task  # register native tasks before importing reach
from ape_walk import reach
from ape_walk.physics.frame50 import (configure_frame50_motors,
                                      give_soles_contact_priority,
                                      require_direct_frame)

ROOT = Path(__file__).resolve().parents[3]
ENTRY_TRACE = ROOT / ("artifacts/ape-cove50-reach-goal-channels-20261010/v158/"
                      "native-development-contact/seed-24607-contact-24607-prescribed.npz")
ENTRY_FRAME = 600
TABLE_FRAME = 0
TABLE_CUBE_XYZ = (0.10, -0.13, 0.16)
STAND_TABLE_CUBE_XYZ = (0.05, -0.14, 0.14)
sys.path.insert(0, str(ROOT / "experiments/compact_morphology/cove50_policy_transfer"))
from reach_observation import TimedRgbTarget  # noqa: E402


def supported_object_touch(env, object_name: str = "cube",
                           support_z_m: float = .01) -> torch.Tensor:
    """Actor reward requires real finger contact and supported free cube."""
    finger = env.scene.sensors["right_finger_tip_cube_contact"].data.found.any(dim=-1)
    feet = env.scene.sensors["feet_ground_contact"].data.found.all(dim=-1)
    hand_floor = env.scene.sensors["hand_ground_contact"].data.found.any(dim=-1)
    cube_z = env.scene[object_name].data.root_link_pos_w[:, 2]
    upright = env.scene["robot"].data.root_link_pos_w[:, 2] >= 0.10
    cube_supported = (cube_z >= support_z_m - .004) & (cube_z <= support_z_m + .011)
    if support_z_m > .02:
        cube_supported &= env.scene.sensors["cube_table_contact"].data.found.any(dim=-1)
    return (finger & feet & ~hand_floor & cube_supported & upright).float()


def physical_touch_gate_frame(env, support_z_m: float = .01) -> bool:
    """Independent episode evaluator; do not derive pass from reward values."""
    finger = bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any())
    feet = bool(env.scene.sensors["feet_ground_contact"].data.found[0].all())
    hand_floor = bool(env.scene.sensors["hand_ground_contact"].data.found[0].any())
    cube_z = float(env.scene["cube"].data.root_link_pos_w[0, 2])
    pelvis_z = float(env.scene["robot"].data.root_link_pos_w[0, 2])
    cube_supported = support_z_m - .004 <= cube_z <= support_z_m + .011
    if support_z_m > .02:
        cube_supported &= bool(env.scene.sensors["cube_table_contact"].data.found[0].any())
    return finger and feet and not hand_floor and cube_supported and pelvis_z >= .10


def hand_distance(env, object_name: str = "cube") -> torch.Tensor:
    hand = env.scene["robot"].data.site_pos_w[:, env.scene["robot"].find_sites("right_hand_center")[0][0]]
    cube = env.scene[object_name].data.root_link_pos_w
    return torch.exp(-torch.linalg.norm(hand - cube, dim=-1) / 0.05)


def table_spec(x: float = .10, y: float = -.13,
               center_z: float = .10, half_x: float = .04) -> mujoco.MjSpec:
    """A static 8 cm square support, separate from the free cube and robot."""
    return mujoco.MjSpec.from_string(
        f'<mujoco><worldbody><body name="touch_table" pos="{x} {y} {center_z}">'
        f'<geom name="touch_table_top" type="box" size="{half_x} 0.04 0.05" '
        'friction="0.8 0.01 0.001" rgba="0.45 0.35 0.25 1"/>'
        '</body></worldbody></mujoco>')


def touch_env_cfg(body: Path, *, entry_mode: str = "floor-reach",
                  stand_cube_xy: tuple[float, float] | None = None,
                  stand_cube_xy_range: tuple[tuple[float, float],
                                             tuple[float, float]] | None = None):
    if entry_mode not in ("floor-reach", "tabletop", "stand-tabletop"):
        raise ValueError(entry_mode)
    if stand_cube_xy_range is not None and entry_mode != "stand-tabletop":
        raise ValueError("placement range requires stand-tabletop entry")
    body = body.resolve()
    model = require_direct_frame(body)
    frame = ENTRY_FRAME if entry_mode == "floor-reach" else TABLE_FRAME
    if entry_mode == "stand-tabletop":
        entry_qpos = np.zeros(35)
        entry_qpos[:28] = model.qpos0
        entry_qpos[2] = .2305
        entry_qpos[model.joint("head_yaw").qposadr[0]] = -1.25
        entry_qpos[28:31] = (*stand_cube_xy, STAND_TABLE_CUBE_XYZ[2]) if stand_cube_xy else STAND_TABLE_CUBE_XYZ
        entry_qpos[31] = 1
    else:
        with np.load(ENTRY_TRACE) as saved:
            entry_qpos = saved["qpos"][frame].astype(float)
    assert entry_qpos.shape == (35,)
    cfg = reach.reach_env_cfg(play=False)
    if entry_mode == "tabletop":
        cfg.scene.entities["table"] = EntityCfg(spec_fn=table_spec)
    elif entry_mode == "stand-tabletop":
        cfg.scene.entities["table"] = EntityCfg(spec_fn=partial(
            table_spec, x=.05, y=-.14, center_z=.08, half_x=.025))
    robot = cfg.scene.entities["robot"]
    robot.collisions[0].disable_other_geoms = False
    give_soles_contact_priority(cfg)
    configure_frame50_motors(cfg, body, rail_v=12.0, delay_s=0.0)
    robot.articulation.actuators = tuple(
        replace(act, delay_min_lag=0, delay_max_lag=4)
        for act in robot.articulation.actuators)
    # MJLab's default 90% soft joint reset clips the attained hip/knee state
    # and tips the robot. Use the actual hard source travel at this entry.
    robot.articulation.soft_joint_pos_limit_factor = 1.0
    # A source-observed, motor-attained reach state at 12.0 s puts the toy in
    # the real lower RGB frame while leaving >30 mm from both pads. It is an
    # isolated reset lesson, not a demonstrated no-reset skill handoff.
    robot.init_state.pos = tuple(map(float, entry_qpos[:3]))
    robot.init_state.rot = tuple(map(float, entry_qpos[3:7]))
    pose = robot.init_state.joint_pos
    pose.clear()
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        pose[f"^{name}$"] = float(entry_qpos[model.jnt_qposadr[joint]])
    # The toy is free on the floor and outside both pads at this attained
    # entry; no preheld or object-in-aperture reset is used.
    command = cfg.commands["lift_height"]
    cube_x, cube_y, cube_z = (entry_qpos[28:31] if entry_mode == "floor-reach"
                             else TABLE_CUBE_XYZ if entry_mode == "tabletop"
                             else entry_qpos[28:31])
    command.object_pose_range.x = (float(cube_x),) * 2
    command.object_pose_range.y = (float(cube_y),) * 2
    command.object_pose_range.z = (float(cube_z),) * 2
    command.object_pose_range.yaw = (0.0, 0.0)
    command.target_position_range.x = (float(cube_x),) * 2
    command.target_position_range.y = (float(cube_y),) * 2
    command.target_position_range.z = (float(cube_z),) * 2
    if stand_cube_xy_range is not None:
        # The independently recorded teacher screen used this local table
        # envelope. Sample one free-object placement per world and reset.
        (x_low, x_high), (y_low, y_high) = stand_cube_xy_range
        if not (.046 <= x_low < x_high <= .054 and
                -.144 <= y_low < y_high <= -.136):
            raise ValueError("placement range exceeds the recorded teacher envelope")
        command.object_pose_range.x = (x_low, x_high)
        command.object_pose_range.y = (y_low, y_high)
        command.target_position_range.x = (x_low, x_high)
        command.target_position_range.y = (y_low, y_high)
    command.resampling_time_range = (100.0, 100.0)
    cfg.scene.sensors += (
        CameraSensorCfg(name="touch_rgb", camera_name="robot/lower_rgb",
                        width=160, height=120, data_types=("rgb",),
                        enabled_geom_groups=(0, 1, 2), use_shadows=False,
                        use_textures=True),
        ContactSensorCfg(name="right_finger_tip_cube_contact",
                         primary=ContactMatch(mode="geom", pattern="right_finger_tip", entity="robot"),
                         secondary=ContactMatch(mode="geom", pattern="cube_geom", entity="cube"),
                         fields=("found", "force"), reduce="netforce", num_slots=1),
        ContactSensorCfg(name="right_hand_cube_contact",
                         primary=ContactMatch(mode="geom", pattern="right_hand", entity="robot"),
                         secondary=ContactMatch(mode="geom", pattern="cube_geom", entity="cube"),
                         fields=("found", "force"), reduce="netforce", num_slots=1),
    )
    if entry_mode in ("tabletop", "stand-tabletop"):
        cfg.scene.sensors += (ContactSensorCfg(
            name="cube_table_contact",
            primary=ContactMatch(mode="geom", pattern="cube_geom", entity="cube"),
            secondary=ContactMatch(mode="geom", pattern="touch_table_top", entity="table"),
            fields=("found", "force"), reduce="netforce", num_slots=1),)
    actor = cfg.observations["actor"].terms
    actor.pop("ee_to_cube")
    actor.pop("cube_to_goal")
    actor["rgb_target"] = ObservationTermCfg(
        func=TimedRgbTarget,
        params={"sensor_name": "touch_rgb", "max_age_s": 0.2, "min_pixels": 10},
    )
    cfg.observations["critic"].terms.pop("cube_to_goal")
    cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
    if entry_mode == "tabletop":
        # The source posture controller actively moves the legs; freezing
        # them makes this physical squat fall in under one second.
        for side in ("left", "right"):
            for axis in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch"):
                cfg.actions["joint_pos"].scale[f"^{side}_{axis}$"] = 1.6
    if entry_mode == "stand-tabletop":
        cfg.actions["joint_pos"].scale["^right_elbow_pitch$"] = 1.0
    # This is a touch task, not the upstream lift reward: hand proximity is
    # weak shaping, and floor-supported physical contact dominates.
    cfg.rewards.pop("lift")
    cfg.rewards.pop("lift_precise")
    cfg.rewards["approach"] = RewardTermCfg(func=hand_distance, weight=0.4)
    cfg.rewards["supported_touch"] = RewardTermCfg(
        func=supported_object_touch, weight=4.0,
        params={"support_z_m": float(cube_z)})
    cfg.episode_length_s = 6.0
    return cfg


def source_target_probe(env, body: Path, *, source_velocity: bool = False,
                        entry_frame: int = ENTRY_FRAME,
                        support_z_m: float = .01) -> dict:
    """Candidate source motor-target replay; acceptance needs physical touch."""
    source_model = require_direct_frame(body)
    with np.load(ENTRY_TRACE) as saved:
        qpos = saved["qpos"].astype(float)
    env.reset(seed=24607)
    if source_velocity:
        # The trace saves 50 Hz qpos only. Finite differencing provides a
        # diagnostic approximation to the reached state's linear/joint rates;
        # it is not an accepted reset dataset or a teacher label.
        if entry_frame == 0:
            velocity = (qpos[1] - qpos[0]) / env.step_dt
        else:
            velocity = (qpos[entry_frame + 1] - qpos[entry_frame - 1]) / (2 * env.step_dt)
        env.sim.data.qvel[0, :3] = torch.as_tensor(velocity[:3], device=env.device)
        env.sim.data.qvel[0, 6:27] = torch.as_tensor(velocity[7:28], device=env.device)
        env.sim.forward()
    action_term = env.action_manager.get_term("joint_pos")
    names = action_term._target_names
    offset = action_term._offset[0]
    scale = action_term._scale[0]
    first_contact = None
    longest = running = 0
    finger_frames = fixed_frames = hand_floor_frames = 0
    max_cube_z = 0.0
    samples = []
    for index in range(entry_frame, len(qpos)):
        target = torch.tensor([
            qpos[index, source_model.joint(name).qposadr[0]] for name in names
        ], dtype=torch.float32, device=env.device)
        raw = torch.where(scale.abs() > 1e-8,
                          (target - offset) / scale.clamp_min(1e-8),
                          torch.zeros_like(target))
        command = raw.clamp(-1, 1)[None, :]
        _, _, terminated, timed_out, _ = env.step(command)
        finger = bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any())
        fixed = bool(env.scene.sensors["right_hand_cube_contact"].data.found[0].any())
        hand_floor = bool(env.scene.sensors["hand_ground_contact"].data.found[0].any())
        physical = physical_touch_gate_frame(env, support_z_m=support_z_m)
        finger_frames += finger
        fixed_frames += fixed
        hand_floor_frames += hand_floor
        running = running + 1 if physical else 0
        longest = max(longest, running)
        if physical and first_contact is None:
            first_contact = round((index - entry_frame) * env.step_dt, 3)
        cube_z = float(env.scene["cube"].data.root_link_pos_w[0, 2])
        max_cube_z = max(max_cube_z, cube_z)
        if (index - entry_frame) % 25 == 0 or index == len(qpos) - 1:
            samples.append({"elapsed_s": round((index - entry_frame) * env.step_dt, 3),
                            "cube_z_m": round(cube_z, 5), "finger": finger,
                            "fixed": fixed, "hand_floor": hand_floor,
                            "physical_touch": physical})
        if bool(terminated[0]) or bool(timed_out[0]):
            break
    return {"candidate": "retained source physical qpos as normalized native joint target",
            "finite_difference_source_velocity_injected": source_velocity,
            "entry_frame": entry_frame, "steps": index - entry_frame + 1,
            "first_physical_touch_s": first_contact,
            "finger_contact_frames": finger_frames,
            "fixed_contact_frames": fixed_frames,
            "hand_floor_frames": hand_floor_frames,
            "longest_supported_touch_s": round(longest * env.step_dt, 3),
            "max_cube_center_z_m": round(max_cube_z, 5),
            "touch_gate_pass": bool(longest * env.step_dt >= 0.2),
            "samples": samples,
            "claim_limit": "Source-target candidate, not an object-aware corrective teacher or learned policy."}


def stand_table_ik_probe(env) -> dict:
    """Bounded, contact-aware authored arm trajectory; no learned weights."""
    env.reset(seed=24607)
    term = env.action_manager.get_term("joint_pos")
    names = term._target_names
    offset = term._offset[0]
    scale = term._scale[0]
    ik = {"right_shoulder_yaw": -.24237,
          "right_shoulder_roll": .11798,
          "right_shoulder_pitch": .22665,
          "right_elbow_pitch": -.74521,
          "right_gripper": .01164}
    target = offset.clone()
    for name, value in ik.items():
        target[names.index(name)] = value
    first_touch = None
    hold_alpha = None
    longest = running = 0
    finger_frames = fixed_frames = table_frames = hand_floor_frames = 0
    min_pelvis_z = float("inf")
    samples = []
    for step in range(200):
        alpha = min(1.0, (step + 1) / 75) if hold_alpha is None else hold_alpha
        desired = offset + alpha * (target - offset)
        raw = torch.where(scale.abs() > 1e-8,
                          (desired - offset) / scale.clamp_min(1e-8),
                          torch.zeros_like(desired))
        command = raw.clamp(-1, 1)[None, :]
        _, _, terminated, timed_out, _ = env.step(command)
        finger = bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any())
        fixed = bool(env.scene.sensors["right_hand_cube_contact"].data.found[0].any())
        table = bool(env.scene.sensors["cube_table_contact"].data.found[0].any())
        hand_floor = bool(env.scene.sensors["hand_ground_contact"].data.found[0].any())
        valid = physical_touch_gate_frame(env, support_z_m=STAND_TABLE_CUBE_XYZ[2])
        if valid and first_touch is None:
            first_touch = round((step + 1) * env.step_dt, 3)
            hold_alpha = alpha
        finger_frames += finger
        fixed_frames += fixed
        table_frames += table
        hand_floor_frames += hand_floor
        running = running + 1 if valid else 0
        longest = max(longest, running)
        min_pelvis_z = min(min_pelvis_z, float(env.scene["robot"].data.root_link_pos_w[0, 2]))
        if step % 25 == 0 or step == 199 or bool(terminated[0]) or bool(timed_out[0]):
            samples.append({"elapsed_s": round((step + 1) * env.step_dt, 3),
                            "alpha": round(alpha, 4), "finger": finger,
                            "fixed": fixed, "cube_table": table,
                            "cube_xyz_m": env.scene["cube"].data.root_link_pos_w[0].detach().cpu().tolist(),
                            "pelvis_z_m": float(env.scene["robot"].data.root_link_pos_w[0, 2]),
                            "valid_touch": valid,
                            "terminated": bool(terminated[0]), "timed_out": bool(timed_out[0])})
        if bool(terminated[0]) or bool(timed_out[0]):
            break
    pass_gate = bool(longest * env.step_dt >= .2)
    return {"candidate": "offline exact-body geometric arm target, ramped 1.5 s; freeze target when physical supported touch occurs",
            "ik_target_rad": ik, "steps": step + 1,
            "first_physical_touch_s": first_touch,
            "finger_frames": finger_frames, "fixed_frames": fixed_frames,
            "cube_table_contact_frames": table_frames,
            "hand_floor_frames": hand_floor_frames,
            "min_pelvis_z_m": round(min_pelvis_z, 5),
            "longest_supported_touch_s": round(longest * env.step_dt, 3),
            "touch_gate_pass": pass_gate,
            "samples": samples,
            "claim_limit": "Authored feasibility teacher; IK uses evaluator cube truth, so labels cannot be deployed as actor observations."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path,
                        default=ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true", required=True)
    parser.add_argument("--teacher-probe", action="store_true")
    parser.add_argument("--ik-teacher-probe", action="store_true")
    parser.add_argument("--source-velocity-probe", action="store_true")
    parser.add_argument("--entry-mode", choices=("floor-reach", "tabletop", "stand-tabletop"), default="floor-reach")
    args = parser.parse_args()
    cfg = touch_env_cfg(args.body_xml, entry_mode=args.entry_mode)
    cfg.scene.num_envs = 1
    cfg.seed = 24607
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    obs, _ = env.reset(seed=24607)
    first_rgb_target = obs["actor"][0, -6:].detach().cpu().tolist()
    assert first_rgb_target[3] == 1 and first_rgb_target[5] == 0
    actor_terms = env.observation_manager.active_terms["actor"]
    assert "ee_to_cube" not in actor_terms and "cube_to_goal" not in actor_terms
    # Sensor truth is read for reporting only. It is absent from actor terms.
    cube = env.scene["cube"].data.root_link_pos_w[0].detach().cpu()
    source_frame = ENTRY_FRAME if args.entry_mode == "floor-reach" else TABLE_FRAME
    if args.entry_mode == "stand-tabletop":
        model = require_direct_frame(args.body_xml)
        expected_robot_qpos = model.qpos0.copy()
        expected_robot_qpos[2] = .2305
        expected_robot_qpos[model.joint("head_yaw").qposadr[0]] = -1.25
    else:
        with np.load(ENTRY_TRACE) as saved:
            expected_robot_qpos = saved["qpos"][source_frame, :28]
    reset_robot_qpos = env.sim.data.qpos[0, :28].detach().cpu().numpy()
    reset_qpos_gap = float(np.max(np.abs(reset_robot_qpos - expected_robot_qpos)))
    assert reset_qpos_gap < 1e-4
    robot = env.scene["robot"]
    pads = [robot.data.geom_pos_w[0, robot.find_geoms(name)[0][0]].detach().cpu()
            for name in ("right_hand", "right_finger_tip")]
    entry_gap = min(float(torch.linalg.norm(pad - cube)) for pad in pads)
    finger = bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any())
    fixed = bool(env.scene.sensors["right_hand_cube_contact"].data.found[0].any())
    action = torch.zeros((1, env.action_space.shape[1]), device="cpu")
    sample = []
    for step in range(10):
        obs, _, terminated, timed_out, _ = env.step(action)
        sample.append({"step": step + 1,
                       "finger": bool(env.scene.sensors["right_finger_tip_cube_contact"].data.found[0].any()),
                       "fixed": bool(env.scene.sensors["right_hand_cube_contact"].data.found[0].any()),
                       "cube_z_m": float(env.scene["cube"].data.root_link_pos_w[0, 2]),
                       "physical_touch_reward": float(supported_object_touch(
                           env, support_z_m=.01 if args.entry_mode == "floor-reach"
                           else TABLE_CUBE_XYZ[2] if args.entry_mode == "tabletop"
                           else STAND_TABLE_CUBE_XYZ[2])[0])})
        assert not bool(terminated[0]) and not bool(timed_out[0])
    if args.teacher_probe and args.entry_mode == "stand-tabletop":
        raise ValueError("Retained source targets are not a stand-tabletop teacher")
    if args.ik_teacher_probe and args.entry_mode != "stand-tabletop":
        raise ValueError("IK teacher only applies to stand-tabletop")
    teacher = source_target_probe(env, args.body_xml,
                                  source_velocity=args.source_velocity_probe,
                                  entry_frame=source_frame,
                                  support_z_m=.01 if args.entry_mode == "floor-reach"
                                  else TABLE_CUBE_XYZ[2]) if args.teacher_probe else None
    ik_teacher = stand_table_ik_probe(env) if args.ik_teacher_probe else None
    report = {"schema": "robot-learning-lab.cove50-free-object-native-touch-preflight.v1",
              "body_sha256": __import__("hashlib").sha256(args.body_xml.read_bytes()).hexdigest(),
              "entry_mode": args.entry_mode,
              "entry_source": ({"fixed_source_keyframe": "exact body neutral stand with pelvis z=.2305 m, head yaw=-1.25 rad"}
                               if args.entry_mode == "stand-tabletop" else
                               {"trace": str(ENTRY_TRACE.relative_to(ROOT)),
                                "frame": source_frame,
                                "trace_sha256": __import__("hashlib").sha256(ENTRY_TRACE.read_bytes()).hexdigest()}),
              "device": "cpu", "worlds": 1, "step_dt_s": env.step_dt,
              "actor_terms": actor_terms,
              "actor_shape": list(obs["actor"].shape),
              "critic_shape": list(obs["critic"].shape),
              "action_shape": list(action.shape),
              "entry": {"cube_xyz_m": cube.tolist(), "pad_center_gap_to_cube_m": entry_gap,
                        "max_robot_qpos_gap_to_attained_source": reset_qpos_gap,
                        "finger_contact": finger, "fixed_contact": fixed,
                        "outside_open_aperture": not (finger or fixed) and entry_gap > .03},
              "first_rgb_target": first_rgb_target,
              "zero_action_steps": sample,
              "source_target_probe": teacher,
              "stand_table_ik_probe": ik_teacher,
              "physical_gate": "At least 0.2 s consecutive right finger/cube contact, free cube supported by its declared floor/table surface, both soles loaded, pelvis >=0.10 m, no hand-floor collision; evaluate a closed-loop policy on independent starts.",
              "lift_gate": "Blocked pending opposing-pad contact, >=4 cm cube center height for >=1 s, stable support and rated motor duty.",
              "claim_limit": ("Native neutral-stand reset and ten CPU zero-action steps; scripted teacher only, no learned skill or CUDA parity."
                              if args.entry_mode == "stand-tabletop" else
                              "Attained-pose native reset and ten CPU zero-action steps; no no-reset handoff, skill success or CUDA parity.")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
