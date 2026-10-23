"""One bounded, teacher-ramped arm residual PPO pilot for free-object touch."""

from __future__ import annotations

import argparse
from copy import deepcopy

from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_rl_cfg, register_mjlab_task

import ape_walk.task  # noqa: F401 - register native tasks
from ape_walk import reach
from touch_reference_residual_v11 import residual_touch_env_cfg
from train_touch_reference_residual_v10 import FreshClockManipulationRunner


def build_train_config(*, worlds: int = 512, updates: int = 96,
                       seed: int = 36221, run_name: str = "cove50-touch-arm-ramp-v11"):
    if worlds != 512 or updates != 96:
        raise ValueError("V11 freezes one 512-world, 96-update fit")
    env = residual_touch_env_cfg(cube_xy_range=((.046, .054), (-.144, -.136)))
    agent = deepcopy(load_rl_cfg(reach.TASK_ID))
    agent.experiment_name = "ape_cove50_touch_arm_ramp_v11"
    agent.run_name = run_name
    agent.seed = seed
    agent.max_iterations = updates
    agent.save_interval = 12
    agent.resume = False
    agent.logger = "tensorboard"
    task_id = "Mjlab-Cove50-Observed-One-Pad-Arm-Ramp-V11"
    register_mjlab_task(task_id=task_id, env_cfg=env, play_env_cfg=env,
                        rl_cfg=agent, runner_cls=FreshClockManipulationRunner)
    cfg = TrainConfig.from_task(task_id)
    cfg.env.scene.num_envs = worlds
    cfg.env.seed = seed
    assert cfg.env.scene.env_spacing == 0.0
    assert cfg.agent.num_steps_per_env == 24
    assert cfg.env.decimation == 10 and cfg.env.sim.mujoco.timestep == .002
    return task_id, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument("--updates", type=int, default=96)
    parser.add_argument("--seed", type=int, default=36221)
    parser.add_argument("--run-name", default="cove50-touch-arm-ramp-v11")
    args = parser.parse_args()
    task_id, cfg = build_train_config(worlds=args.worlds, updates=args.updates,
                                      seed=args.seed, run_name=args.run_name)
    launch_training(task_id, cfg)


if __name__ == "__main__":
    main()
