"""Continue the right support actor through a timed upright-to-hold lesson."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import mujoco
import torch
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from ape_walk import task as ape
from ape_walk.physics.frame50 import clip_frame50_leg_targets, configure_frame50_motors
from experiments.compact_morphology.cove50_support_recovery.phase_actions import PhasePositionActionCfg
from experiments.compact_morphology.cove50_support_recovery.phase_rewards import configure_phase_rewards, endpoint_com_offset
from experiments.compact_morphology.cove50_support_recovery.residual import configure_residual_actions


def survival_reward(env):
    return torch.ones(env.num_envs, device=env.device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pose-json", type=Path, required=True)
    p.add_argument("--stance", choices=("left", "right"), default="right")
    p.add_argument("--preload-json", type=Path, required=True)
    p.add_argument("--body-xml", type=Path, required=True)
    p.add_argument("--resume-run", required=True)
    p.add_argument("--resume-checkpoint", required=True)
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--run-name", required=True)
    p.add_argument("--freeze-swing", action="store_true")
    p.add_argument("--phase-rewards", action="store_true")
    p.add_argument("--lift-s", type=float, default=2.0)
    p.add_argument("--hold-s", type=float, default=3.0)
    p.add_argument("--lower-s", type=float, default=2.0)
    p.add_argument("--arm-clearance-bump-rad", type=float, default=0.0)
    p.add_argument("--staged", action="store_true")
    p.add_argument("--urgent-support", action="store_true")
    p.add_argument("--stability-focus", action="store_true")
    args = p.parse_args()
    body = args.body_xml.resolve()
    model = mujoco.MjModel.from_xml_path(str(body))
    names = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    swing = "left" if args.stance == "right" else "right"
    spec = json.loads(args.pose_json.read_text())
    preload = json.loads(args.preload_json.read_text())["joints"]
    pose = [spec["joints"][name] for name in names]
    _, _, _, roll, pitch = spec["root_xyz_roll_pitch"]
    source = (ape.ELBOW_FLAMINGO_COMMAND_LIMIT_TASK_ID if args.stance == "right"
              else ape.ELBOW_FLAMINGO_LEFT_TASK_ID)
    cfg = load_env_cfg(source, play=False)
    configure_frame50_motors(cfg, body, rail_v=12.0, delay_s=0.0)
    cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
    clip_frame50_leg_targets(cfg, body, margin_rad=0.02)
    configure_residual_actions(cfg, preload)
    old = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = PhasePositionActionCfg(
        entity_name=old.entity_name, actuator_names=old.actuator_names,
        scale=old.scale, offset=old.offset, clip=old.clip,
        preserve_order=old.preserve_order, use_default_offset=False,
        target_joints=tuple(preload[name] for name in names),
        lift_s=args.lift_s, hold_s=args.hold_s, lower_s=args.lower_s,
        freeze_swing_side=swing if args.freeze_swing else None,
        arm_clearance_bump_rad=args.arm_clearance_bump_rad,
        staged=args.staged,
        stance_side=args.stance,
    )
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, float(model.qpos0[2]))
    cfg.episode_length_s = args.lift_s + args.hold_s + args.lower_s + 1.0
    cfg.events.pop("set_flamingo_state")
    cfg.rewards["pose_flamingo"].params["target_overrides"] = dict(enumerate(pose))
    cfg.rewards["gravity_flamingo"].params["target"] = (
        math.sin(pitch), -math.sin(roll) * math.cos(pitch),
        -math.cos(roll) * math.cos(pitch),
    )
    cfg.rewards["swing_foot_clear"].params["target"] = 0.07
    cfg.rewards["joint_torques_l2"].weight = -0.05
    cfg.rewards["survival"] = RewardTermCfg(func=survival_reward, weight=3.0)
    if args.phase_rewards:
        configure_phase_rewards(
            cfg, pose,
            (math.sin(pitch), -math.sin(roll) * math.cos(pitch),
             -math.cos(roll) * math.cos(pitch)),
            lift_s=args.lift_s, hold_s=args.hold_s, lower_s=args.lower_s,
            arm_index=names.index(f"{swing}_shoulder_roll"),
            arm_bump=args.arm_clearance_bump_rad,
            swing_indices=[i for i, name in enumerate(names) if
                           name.startswith(f"{swing}_") and any(name.endswith(axis)
                           for axis in ("hip_yaw", "hip_roll", "hip_pitch",
                                        "knee", "ankle_pitch"))],
            staged=args.staged,
            com_offset_xy=endpoint_com_offset(model, spec, args.stance),
            stance=args.stance, urgent_support=args.urgent_support)
        if args.urgent_support:
            cfg.rewards["survival"].weight = 1.0
        if args.stability_focus:
            cfg.rewards["phase_com"].weight = 6.0
            cfg.rewards["phase_com"].params["std"] = 0.025
            cfg.rewards["phase_gravity"].weight = 4.0
            cfg.rewards["foot_slip"].weight = -2.0
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ape.ContactSensorCfg(
            name="thigh_torso_contact",
            primary=ape.ContactMatch(
                mode="geom", pattern=r"^(left_thigh_frame_budget|right_thigh_frame_budget)$",
                entity="robot"),
            secondary=ape.ContactMatch(
                mode="geom", pattern="torso_frame_compute_budget", entity="robot"),
            fields=("found", "force"), reduce="maxforce", num_slots=1,
        ),
    )
    cfg.terminations["thigh_torso_contact"] = TerminationTermCfg(
        func=ape.mdp.illegal_contact,
        params={"sensor_name": "thigh_torso_contact", "force_threshold": 0.1},
    )
    agent = deepcopy(load_rl_cfg(source))
    agent.experiment_name = f"ape_cove50_support_residual_{args.stance}"
    task_id = f"Mjlab-Cove50-Stand-Transition-{args.stance.title()}"
    ape.register_mjlab_task(task_id=task_id, env_cfg=cfg, play_env_cfg=cfg,
                            rl_cfg=agent, runner_cls=ape.VelocityOnPolicyRunner)
    train = TrainConfig.from_task(task_id)
    train.env.scene.num_envs = args.worlds
    train.env.sim.mujoco.timestep = 0.002
    train.env.decimation = 10
    train.agent.max_iterations = args.updates
    train.agent.save_interval = 25
    train.agent.run_name = args.run_name
    train.agent.logger = "tensorboard"
    train.agent.resume = True
    train.agent.load_run = args.resume_run
    train.agent.load_checkpoint = args.resume_checkpoint
    launch_training(task_id, train)


if __name__ == "__main__":
    main()
