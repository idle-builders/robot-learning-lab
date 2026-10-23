"""Train the independent shoulder flags with upstream MJLab/Pollen PPO."""

import ape_walk.task
from mjlab.scripts.train import TrainConfig, launch_training


if __name__ == "__main__":
    task = ape_walk.task.ELBOW_ARM_INDEPENDENT_TASK_ID
    cfg = TrainConfig.from_task(task)
    cfg.env.scene.num_envs = 4096
    cfg.agent.max_iterations = 250
    cfg.agent.save_interval = 50
    cfg.agent.experiment_name = "ape_elbows_arms_independent"
    cfg.agent.run_name = "independent_flags_250"
    cfg.agent.logger = "tensorboard"
    launch_training(task, cfg)
