"""Short, varied, observed-target PPO lesson for one-pad supported touch.

This starts a new actor; the prior BC pilot is an evaluation baseline, not a
compatible PPO checkpoint. Free-object possession and lift are separate skills.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_rl_cfg

from ape_walk import reach
from ape_walk import task as ape
from free_object_touch_env import ROOT, touch_env_cfg


def build_train_config(worlds: int = 256, updates: int = 12,
                       seed: int = 35021, run_name: str = "cove50-one-pad-touch-v1") -> tuple[str, TrainConfig]:
    if worlds not in (128, 256) or not 1 <= updates <= 12:
        raise ValueError("This bounded pilot supports 128/256 worlds and at most 12 updates")
    body = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
    env = touch_env_cfg(body, entry_mode="stand-tabletop",
                        stand_cube_xy_range=((.046, .054), (-.144, -.136)))
    agent = deepcopy(load_rl_cfg(reach.TASK_ID))
    agent.experiment_name = "ape_cove50_one_pad_touch"
    agent.run_name = run_name
    agent.seed = seed
    agent.max_iterations = updates
    agent.save_interval = updates
    agent.resume = False
    agent.logger = "tensorboard"
    task_id = "Mjlab-Cove50-Observed-One-Pad-Touch"
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
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--seed", type=int, default=35021)
    parser.add_argument("--run-name", default="cove50-one-pad-touch-v1")
    args = parser.parse_args()
    task_id, cfg = build_train_config(args.worlds, args.updates, args.seed, args.run_name)
    launch_training(task_id, cfg)


if __name__ == "__main__":
    main()
