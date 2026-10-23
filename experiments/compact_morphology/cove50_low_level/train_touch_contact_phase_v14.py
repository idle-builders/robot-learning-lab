"""Bounded PPO pilot for a staged, five-arm-channel contact-phase residual."""

from __future__ import annotations

import argparse
from copy import deepcopy

from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_rl_cfg, register_mjlab_task

import ape_walk.task  # register native tasks before reach imports grasp
from ape_walk import reach
from touch_reference_residual_v14 import residual_touch_env_cfg
from train_touch_reference_residual_v10 import FreshClockManipulationRunner


TASK_ID = "Mjlab-Cove50-Observed-One-Pad-Contact-Phase-V14"


def build_train_config(*, worlds: int = 256, updates: int = 12,
                       seed: int = 37121,
                       run_name: str = "cove50-touch-contact-phase-v14"):
    if worlds not in (128, 256, 512) or updates not in (8, 12, 16):
        raise ValueError("V14 needs a bounded 8/12/16-update pilot")
    env = residual_touch_env_cfg(cube_xy_range=((.046, .054), (-.144, -.136)))
    env.actions["joint_pos"].residual_gain = 0.005
    # One complete 200-step route must be observable within 12 updates / 288
    # steps per world; V10's 300-step horizon yielded no full-episode metrics.
    env.episode_length_s = 4.0
    agent = deepcopy(load_rl_cfg(reach.TASK_ID))
    agent.experiment_name = "ape_cove50_touch_contact_phase_v14"
    agent.run_name = run_name
    agent.seed = seed
    agent.max_iterations = updates
    agent.save_interval = 1
    agent.resume = False
    agent.logger = "tensorboard"
    register_mjlab_task(task_id=TASK_ID, env_cfg=env, play_env_cfg=env,
                        rl_cfg=agent, runner_cls=FreshClockManipulationRunner)
    cfg = TrainConfig.from_task(TASK_ID)
    cfg.env.scene.num_envs, cfg.env.seed = worlds, seed
    assert cfg.env.scene.env_spacing == 0.0
    assert cfg.agent.num_steps_per_env == 24
    assert cfg.env.decimation == 10 and cfg.env.sim.mujoco.timestep == .002
    return TASK_ID, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--seed", type=int, default=37121)
    parser.add_argument("--run-name", default="cove50-touch-contact-phase-v14")
    args = parser.parse_args()
    task_id, cfg = build_train_config(worlds=args.worlds, updates=args.updates,
                                      seed=args.seed, run_name=args.run_name)
    launch_training(task_id, cfg)


if __name__ == "__main__":
    main()
