"""Train the ape's stand -> foot lift -> set-down motion in MJLab tracking."""

import argparse

from ape_walk.foot_transition_tracking import TASK_ID
from mjlab.scripts.train import TrainConfig, launch_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=2048)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--residual", action="store_true",
                        help="Train corrections around the reference motion")
    parser.add_argument("--push", action="store_true",
                        help="Include MJLab's random velocity-push event in residual training")
    parser.add_argument("--hold-push", action="store_true",
                        help="Train MJLab velocity pushes near the one-foot hold")
    parser.add_argument("--right", action="store_true",
                        help="Train the mirrored right-foot reference")
    parser.add_argument("--step-side", choices=("left", "right"),
                        help="Train a one-step reference tracker for the chosen lead foot")
    args = parser.parse_args()
    task_id = TASK_ID
    if args.step_side:
        from ape_walk import step  # noqa: F401 - registers both step tasks
        task_id = f"Mjlab-Tracking-Ape-Single-Step-{args.step_side.title()}"
    elif args.right:
        from ape_walk.foot_transition_residual_right import TASK_ID as right_task_id
        task_id = right_task_id
    elif args.hold_push:
        from ape_walk.foot_transition_residual_hold_push import TASK_ID as hold_push_task_id
        task_id = hold_push_task_id
    elif args.push:
        from ape_walk.foot_transition_residual_push import TASK_ID as push_task_id
        task_id = push_task_id
    elif args.residual:
        from ape_walk.foot_transition_residual import TASK_ID as residual_task_id
        task_id = residual_task_id
    cfg = TrainConfig.from_task(task_id)
    cfg.env.scene.num_envs = args.worlds
    cfg.agent.max_iterations = args.updates
    cfg.agent.logger = "tensorboard"
    launch_training(task_id, cfg)
