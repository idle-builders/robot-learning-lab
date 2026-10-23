"""Continue a one-foot hold actor from the corrected Cove flat-sole reset.

This stage learns only a raised-foot hold. Stand-to-lift is a later gate.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import mujoco
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from ape_walk import task as ape
from ape_walk.physics.frame50 import clip_frame50_leg_targets
from experiments.compact_morphology.cove50_policy_transfer.train_foot_hold import retarget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-json", type=Path, required=True)
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--stance", choices=("left", "right"), required=True)
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--torque-weight", type=float, default=-0.4,
                        help="Stage hold survival before rated-effort tuning")
    parser.add_argument("--resume-run", required=True)
    parser.add_argument("--resume-checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    spec = json.loads(args.pose_json.read_text())
    body = args.body_xml.resolve()
    model = mujoco.MjModel.from_xml_path(str(body))
    if model.nu != 21:
        raise ValueError("Expected selected 21-axis Cove body")
    names = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    pose = [spec["joints"][name] for name in names]
    _, _, z, roll, pitch = spec["root_xyz_roll_pitch"]
    source_task = (ape.ELBOW_FLAMINGO_LEFT_TASK_ID if args.stance == "left"
                   else ape.ELBOW_FLAMINGO_COMMAND_LIMIT_TASK_ID)
    cfg = load_env_cfg(source_task, play=False)
    retarget(cfg, body, model, args.stance, variance=False)
    clip_frame50_leg_targets(cfg, body, margin_rad=0.02)
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, z)
    cfg.rewards["pose_flamingo"].params["target_overrides"] = dict(enumerate(pose))
    cfg.rewards["gravity_flamingo"].params["target"] = (
        math.sin(pitch), -math.sin(roll) * math.cos(pitch),
        -math.cos(roll) * math.cos(pitch),
    )
    cfg.rewards["swing_foot_clear"].params["target"] = 0.10
    cfg.rewards["joint_torques_l2"].weight = args.torque_weight
    cfg.events["set_flamingo_state"].params.update({
        "joint_pose": pose, "base_roll": roll, "base_pitch": pitch,
        "z_min": z, "z_max": z + 0.002,
        "tilt_noise": math.radians(1), "joint_noise_std": 0.005,
        "standing_prob": 0.0,
    })
    agent = deepcopy(load_rl_cfg(source_task))
    agent.experiment_name = f"ape_cove50_support_recovery_{args.stance}"
    task_id = f"Mjlab-Cove50-Support-Recovery-{args.stance.title()}"
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
