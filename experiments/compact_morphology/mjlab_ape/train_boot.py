"""Run the upstream MJLab/Pollen velocity PPO lesson on an exported CAD boot.

Only the robot MJCF changes. The task observations, actions, command sampler,
rewards, PPO implementation and 2 ms physics step stay with the reference.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path

from ape_walk import task as ape
from mjlab.scripts.train import TrainConfig, launch_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=4096)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--run-name", default="boot_cad_mass_warmstart")
    parser.add_argument("--resume-run")
    parser.add_argument("--resume-checkpoint", default="model_0.pt")
    parser.add_argument("--initial-root-z", type=float,
                        help="Pelvis reset height in metres for a different body")
    parser.add_argument("--yaw-only-head", action="store_true",
                        help="Remove the absent head-pitch action for the 21-axis ape")
    parser.add_argument("--arms-down", action="store_true",
                        help="Reset the longer ape arms by the torso for the gait lesson")
    parser.add_argument("--physics-model", choices=("baseline", "frame50"), default="baseline",
                        help="Use the selected 21-axis catalog motor-envelope sensitivity")
    parser.add_argument("--motor-rail-v", type=float, default=12.0)
    parser.add_argument("--command-delay-ms", type=float, default=0.0)
    args = parser.parse_args()

    task_id = "Mjlab-Velocity-Flat-Ape-Elbows-Boot"
    env = ape.env_cfg(elbowed=True)
    env.scene.entities["robot"].spec_fn = partial(ape._ape_spec, args.body_xml.resolve())
    env.scene.entities["robot"].collisions[0].disable_other_geoms = False
    if args.physics_model == "frame50":
        from ape_walk.physics.frame50 import configure_frame50_motors

        configure_frame50_motors(
            env, args.body_xml, rail_v=args.motor_rail_v,
            delay_s=args.command_delay_ms / 1000.0,
        )
    if args.initial_root_z is not None:
        env.scene.entities["robot"].init_state.pos = (0.0, 0.0, args.initial_root_z)
    if args.yaw_only_head:
        env.actions["joint_pos"].scale.pop("^head_pitch$", None)
    if args.arms_down:
        pose = env.scene.entities["robot"].init_state.joint_pos
        pose[r".*shoulder_pitch.*"] = 0.0
        pose[r".*elbow_pitch.*"] = 0.0
    agent = deepcopy(ape.elbow_ape_rl_cfg)
    agent.experiment_name = "ape_boot_velocity"
    ape.register_mjlab_task(task_id=task_id, env_cfg=env, play_env_cfg=env,
                            rl_cfg=agent, runner_cls=ape.VelocityOnPolicyRunner)

    cfg = TrainConfig.from_task(task_id)
    cfg.env.scene.num_envs = args.worlds
    cfg.env.sim.mujoco.timestep = 0.002
    cfg.env.decimation = 10
    cfg.agent.max_iterations = args.updates
    cfg.agent.save_interval = 50
    cfg.agent.run_name = args.run_name
    cfg.agent.logger = "tensorboard"
    if args.resume_run:
        cfg.agent.resume = True
        cfg.agent.load_run = args.resume_run
        cfg.agent.load_checkpoint = args.resume_checkpoint
    launch_training(task_id, cfg)


if __name__ == "__main__":
    main()
