"""Bounded PPO correction around the observed-RGB one-pad teacher reference."""

from __future__ import annotations

import argparse
from copy import deepcopy

from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_rl_cfg

from ape_walk import reach
from ape_walk import task as ape
from touch_reference_residual_v9 import residual_touch_env_cfg


def build_train_config(*, worlds: int = 256, updates: int = 8,
                       seed: int = 36021, run_name: str = "cove50-touch-reference-residual-v9"):
    if worlds not in (128, 256, 512, 1024) or updates not in (4, 8, 12):
        raise ValueError("V9 uses one fresh 4/8/12-update fit and benchmarked world counts")
    env = residual_touch_env_cfg(cube_xy_range=((.046, .054), (-.144, -.136)))
    agent = deepcopy(load_rl_cfg(reach.TASK_ID))
    agent.experiment_name = "ape_cove50_touch_reference_residual_v9"
    agent.run_name = run_name
    agent.seed = seed
    agent.max_iterations = updates
    agent.save_interval = 1
    agent.resume = False
    agent.logger = "tensorboard"
    task_id = "Mjlab-Cove50-Observed-One-Pad-Residual-V9"
    ape.register_mjlab_task(task_id=task_id, env_cfg=env, play_env_cfg=env,
                            rl_cfg=agent, runner_cls=reach.ManipulationOnPolicyRunner)
    cfg = TrainConfig.from_task(task_id)
    cfg.env.scene.num_envs = worlds
    cfg.env.seed = seed
    assert cfg.agent.num_steps_per_env == 24
    assert cfg.env.decimation == 10 and cfg.env.sim.mujoco.timestep == .002
    return task_id, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=36021)
    parser.add_argument("--run-name", default="cove50-touch-reference-residual-v9")
    args = parser.parse_args()
    task_id, cfg = build_train_config(worlds=args.worlds, updates=args.updates,
                                      seed=args.seed, run_name=args.run_name)
    launch_training(task_id, cfg)


if __name__ == "__main__":
    main()
