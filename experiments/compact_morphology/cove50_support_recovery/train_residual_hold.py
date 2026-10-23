"""Train native-motor corrections around a feasible XML-preloaded support pose.

This stage starts from a raised-foot reset. It does not teach the transition.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path

import mujoco
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from ape_walk import task as ape
from ape_walk.physics.frame50 import clip_frame50_leg_targets
from experiments.compact_morphology.cove50_policy_transfer.train_foot_hold import retarget
from experiments.compact_morphology.cove50_support_recovery.residual import configure_residual_actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-json", type=Path, required=True)
    parser.add_argument("--preload-json", type=Path, required=True)
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--stance", choices=("left", "right"), required=True)
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--updates", type=int, default=150)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--domain-variation", action="store_true",
                        help="Sample mass, grip, motor strength, PD, and 0–10 ms lag")
    parser.add_argument("--resume-run")
    parser.add_argument("--resume-checkpoint")
    args = parser.parse_args()
    if bool(args.resume_run) != bool(args.resume_checkpoint):
        parser.error("--resume-run and --resume-checkpoint must be given together")
    spec = json.loads(args.pose_json.read_text())
    preload = json.loads(args.preload_json.read_text())
    body = args.body_xml.resolve()
    model = mujoco.MjModel.from_xml_path(str(body))
    if model.nu != 21:
        raise ValueError("Expected selected 21-axis Cove body")
    names = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    pose = [spec["joints"][name] for name in names]
    _, _, z, roll, pitch = spec["root_xyz_roll_pitch"]
    source = (ape.ELBOW_FLAMINGO_LEFT_TASK_ID if args.stance == "left"
              else ape.ELBOW_FLAMINGO_COMMAND_LIMIT_TASK_ID)
    cfg = load_env_cfg(source, play=False)
    retarget(cfg, body, model, args.stance, variance=args.domain_variation)
    if args.domain_variation:
        cfg.scene.entities["robot"].articulation.actuators = tuple(
            replace(act, delay_min_lag=0, delay_max_lag=5)
            for act in cfg.scene.entities["robot"].articulation.actuators
        )
        cfg.events["cove_link_density"].params["alpha_range"] = (
            math.log(0.85) / 2, math.log(1.15) / 2,
        )
        cfg.events["cove_sole_friction"].params["ranges"] = (0.55, 1.15)
        cfg.events["cove_effort"].params["effort_limit_range"] = (0.83, 1.05)
    clip_frame50_leg_targets(cfg, body, margin_rad=0.02)
    configure_residual_actions(cfg, preload["joints"])
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, z)
    cfg.rewards["pose_flamingo"].params["target_overrides"] = dict(enumerate(pose))
    cfg.rewards["gravity_flamingo"].params["target"] = (
        math.sin(pitch), -math.sin(roll) * math.cos(pitch),
        -math.cos(roll) * math.cos(pitch),
    )
    cfg.rewards["swing_foot_clear"].params["target"] = 0.07
    cfg.rewards["joint_torques_l2"].weight = -0.05
    cfg.events["set_flamingo_state"].params.update({
        "joint_pose": pose, "base_roll": roll, "base_pitch": pitch,
        "z_min": z, "z_max": z + 0.002,
        "tilt_noise": math.radians(1), "joint_noise_std": 0.005,
        "standing_prob": 0.0,
    })
    agent = deepcopy(load_rl_cfg(source))
    agent.experiment_name = f"ape_cove50_support_residual_{args.stance}"
    agent.actor.distribution_cfg["init_std"] = 0.25
    task_id = f"Mjlab-Cove50-Support-Residual-{args.stance.title()}"
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
    train.agent.resume = bool(args.resume_run)
    if args.resume_run:
        train.agent.load_run = args.resume_run
        train.agent.load_checkpoint = args.resume_checkpoint
    launch_training(task_id, train)


if __name__ == "__main__":
    main()
