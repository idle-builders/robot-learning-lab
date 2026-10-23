"""Run Pollen/MJLab PPO for commanded forward touchdown, no motion reference."""

import argparse

from ape_walk.foot_place_command import TASK_ID, NEAR_TASK_ID, RAISED_TASK_ID
from mjlab.scripts.train import TrainConfig, launch_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=2048)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--warmstart", action="store_true")
    parser.add_argument("--warmstart-run", default="warmstart")
    parser.add_argument("--near", action="store_true", help="Start near landing from a physics-replayed reset")
    parser.add_argument("--raised", action="store_true", help="Start from the lifted foot before forward swing")
    parser.add_argument("--slip-weight", type=float,
                        help="Override the native MJLab contact-foot slip penalty")
    args = parser.parse_args()
    if args.near and args.raised:
        parser.error("choose one reset")
    task_id = RAISED_TASK_ID if args.raised else NEAR_TASK_ID if args.near else TASK_ID
    cfg = TrainConfig.from_task(task_id)
    cfg.env.scene.num_envs = args.worlds
    if args.slip_weight is not None:
        cfg.env.rewards["foot_slip"].weight = args.slip_weight
    cfg.agent.max_iterations = args.updates
    cfg.agent.logger = "tensorboard"
    if args.warmstart:
        cfg.agent.resume = True
        cfg.agent.load_run = args.warmstart_run
        cfg.agent.load_checkpoint = "model_0.pt"
        if args.warmstart_run != "warmstart":
            cfg.agent.run_name = args.warmstart_run
    launch_training(task_id, cfg)
