"""Short PPO pilot with aligned batched table and a true fresh episode clock."""

from __future__ import annotations

import argparse
from copy import deepcopy

import torch
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_rl_cfg, register_mjlab_task
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner

import ape_walk.task  # register native tasks before reach imports grasp
from ape_walk import reach
from touch_reference_residual_v10 import residual_touch_env_cfg


class FreshClockManipulationRunner(ManipulationOnPolicyRunner):
    """Keep the teacher ramp synchronized with its physical reset state.

    MJLab's generic train launcher requests randomized initial episode lengths.
    That would jump this task's 75-step reference ramp forward while the robot,
    cube, RGB cache and actor observation still represent a fresh entry.
    """

    def learn(self, num_learning_iterations: int,
              init_at_random_ep_len: bool = False) -> None:
        lengths = self.env.episode_length_buf
        if bool(torch.any(lengths != 0)):
            raise RuntimeError("V10 requires zero episode age at PPO entry")
        print("V10 fresh-clock PPO: initial episode age zero; randomization disabled", flush=True)
        return super().learn(num_learning_iterations=num_learning_iterations,
                             init_at_random_ep_len=False)


def build_train_config(*, worlds: int = 256, updates: int = 8,
                       seed: int = 36021,
                       run_name: str = "cove50-touch-reference-residual-v10"):
    if worlds not in (128, 256, 512, 1024) or updates not in (4, 8, 12):
        raise ValueError("V10 uses one bounded 4/8/12-update fit")
    env = residual_touch_env_cfg(cube_xy_range=((.046, .054), (-.144, -.136)))
    agent = deepcopy(load_rl_cfg(reach.TASK_ID))
    agent.experiment_name = "ape_cove50_touch_reference_residual_v10"
    agent.run_name = run_name
    agent.seed = seed
    agent.max_iterations = updates
    agent.save_interval = 1
    agent.resume = False
    agent.logger = "tensorboard"
    task_id = "Mjlab-Cove50-Observed-One-Pad-Residual-V10"
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
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=36021)
    parser.add_argument("--run-name", default="cove50-touch-reference-residual-v10")
    args = parser.parse_args()
    task_id, cfg = build_train_config(worlds=args.worlds, updates=args.updates,
                                      seed=args.seed, run_name=args.run_name)
    launch_training(task_id, cfg)


if __name__ == "__main__":
    main()
