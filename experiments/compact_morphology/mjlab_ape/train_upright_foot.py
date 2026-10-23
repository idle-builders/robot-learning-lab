"""Train the compact right-foot endpoint with upstream Pollen/MJLab PPO."""

import argparse

import ape_walk.task
from ape_walk.foot_upright import (
    TASK_ID, CLEAR_TASK_ID, LEFT_CLEAR_TASK_ID,
    RIGHT_CONTACT_SAFE_TASK_ID, LEFT_CONTACT_SAFE_TASK_ID,
)
from mjlab.scripts.train import TrainConfig, launch_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true", help="Require airborne swing contact for height reward")
    parser.add_argument("--left", action="store_true", help="Train the mirrored left-stance endpoint")
    parser.add_argument("--target-seeded", action="store_true", help="Resume from physics-held motor-target actor")
    parser.add_argument("--contact-safe", action="store_true",
                        help="Continue a saved hold with native arm-pelvis contact penalty")
    args = parser.parse_args()
    if args.left and not args.clear:
        parser.error("--left requires --clear")
    if args.contact_safe:
        task_id = LEFT_CONTACT_SAFE_TASK_ID if args.left else RIGHT_CONTACT_SAFE_TASK_ID
    else:
        task_id = LEFT_CLEAR_TASK_ID if args.left else CLEAR_TASK_ID if args.clear else TASK_ID
    cfg = TrainConfig.from_task(task_id)
    cfg.env.scene.num_envs = 2048
    cfg.agent.max_iterations = 250
    cfg.agent.save_interval = 50
    cfg.agent.logger = "tensorboard"
    if args.target_seeded:
        if not args.clear:
            parser.error("--target-seeded requires --clear")
        cfg.agent.resume = True
        cfg.agent.load_run = "seeded_target_v1"
        cfg.agent.load_checkpoint = "model_0.pt"
        cfg.agent.run_name = ("upright_left_hold_target_seeded" if args.left
                              else "upright_right_hold_target_seeded")
    if args.contact_safe:
        if args.target_seeded:
            parser.error("--contact-safe continues a trained hold, not --target-seeded")
        cfg.agent.resume = True
        cfg.agent.load_run = "seeded_hold_v1"
        cfg.agent.load_checkpoint = "model_0.pt"
        cfg.agent.max_iterations = 150
    launch_training(task_id, cfg)
