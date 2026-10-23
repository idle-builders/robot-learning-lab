"""Train an absolute-action ape step policy from motion imitation plus MJLab PPO."""

import argparse

from ape_walk import step_distill  # registers left and right tasks
from mjlab.scripts.train import TrainConfig, launch_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--bc-epochs", type=int, default=1)
    parser.add_argument("--resume-run", help="Existing run directory name to continue")
    parser.add_argument("--resume-checkpoint", default="model_149.pt")
    args = parser.parse_args()
    task_id = f"Mjlab-Distilled-Ape-Step-{args.side.title()}"
    cfg = TrainConfig.from_task(task_id)
    cfg.env.scene.num_envs = args.worlds
    cfg.agent.max_iterations = args.updates
    cfg.agent.algorithm.bc_epochs = args.bc_epochs
    cfg.agent.logger = "tensorboard"
    if args.resume_run:
        cfg.agent.resume = True
        cfg.agent.load_run = args.resume_run
        cfg.agent.load_checkpoint = args.resume_checkpoint
        cfg.agent.run_name = f"bc{args.bc_epochs}_from_{args.resume_checkpoint.removesuffix('.pt')}"
    launch_training(task_id, cfg)
