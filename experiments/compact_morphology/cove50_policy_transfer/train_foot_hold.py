"""Continue retained right/left one-foot holds from raised-foot Cove resets.

The 35 cm reset pose was mapped by joint name to Cove's 21 axes.  Fresh Cove
kinematics place either stance sole on the floor at a 0.2545 m pelvis height,
with its opposite sole about 0.09 m above the floor.  This does not teach a
stand-to-lift transition or qualify physical motor duty.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from functools import partial
import math
from pathlib import Path

import mujoco
from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from ape_walk import task as ape
from ape_walk.physics.frame50 import (configure_frame50_motors,
                                      give_soles_contact_priority,
                                      require_direct_frame)


TASKS = {
    "right": ape.ELBOW_FLAMINGO_COMMAND_LIMIT_TASK_ID,
    "left": ape.ELBOW_FLAMINGO_LEFT_TASK_ID,
}
RAISED_Z = 0.2545


def cove_pose(side: str) -> list[float]:
    source = (ape.APE_FLAMINGO_RIGHT_POSE if side == "right"
              else ape.APE_FLAMINGO_LEFT_POSE)
    if len(source) != 22 or source[11] != 0.0:
        raise ValueError("Historical foot-hold head-pitch neutral changed")
    return list(source[:11] + source[12:])


def retarget(cfg, body: Path, model: mujoco.MjModel, side: str,
             *, variance: bool = True) -> None:
    cfg.scene.entities["robot"].spec_fn = partial(ape._ape_spec, body)
    cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
    give_soles_contact_priority(cfg)
    configure_frame50_motors(cfg, body, rail_v=12.0, delay_s=0.0)
    robot = cfg.scene.entities["robot"]
    if variance:
        robot.articulation.actuators = tuple(
            replace(act, delay_min_lag=0, delay_max_lag=4)
            for act in robot.articulation.actuators
        )
    robot.init_state.pos = (0.0, 0.0, RAISED_Z)
    cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
    pose = cove_pose(side)
    cfg.rewards["pose_flamingo"].params["target_overrides"] = dict(enumerate(pose))
    cfg.rewards["swing_foot_clear"].params["target"] = 0.09
    reset = cfg.events["set_flamingo_state"].params
    reset["joint_pose"] = pose
    reset["z_min"] = RAISED_Z
    reset["z_max"] = RAISED_Z + 0.004
    if not variance:
        return
    reset["joint_noise_std"] = 0.04
    reset["tilt_noise"] = math.radians(3.0)
    bodies = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                   for i in range(1, model.nbody))
    cfg.events["cove_link_density"] = EventTermCfg(
        mode="reset", func=dr.pseudo_inertia,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=bodies),
                "alpha_range": (math.log(0.9) / 2, math.log(1.1) / 2)},
    )
    cfg.events["cove_sole_friction"] = EventTermCfg(
        mode="reset", func=dr.geom_friction,
        params={"asset_cfg": SceneEntityCfg("robot", geom_names=("left_sole", "right_sole")),
                "operation": "abs", "ranges": (0.65, 1.15),
                "axes": [0], "shared_random": True},
    )
    cfg.events["cove_effort"] = EventTermCfg(
        mode="reset", func=dr.effort_limits,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "effort_limit_range": (0.9, 1.05), "operation": "scale"},
    )
    cfg.events["cove_pd"] = EventTermCfg(
        mode="reset", func=dr.pd_gains,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "kp_range": (0.9, 1.1), "kd_range": (0.9, 1.1),
                "operation": "scale"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=TASKS, required=True)
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--updates", type=int, default=150)
    parser.add_argument("--resume-run", required=True)
    parser.add_argument("--resume-checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    body = args.body_xml.resolve()
    model = require_direct_frame(body)
    cfg = load_env_cfg(TASKS[args.side], play=False)
    retarget(cfg, body, model, args.side)
    agent = deepcopy(load_rl_cfg(TASKS[args.side]))
    agent.experiment_name = f"ape_cove50_hold_{args.side}"
    task_id = f"Mjlab-Cove50-Hold-{args.side.title()}"
    ape.register_mjlab_task(task_id=task_id, env_cfg=cfg, play_env_cfg=cfg,
                            rl_cfg=agent, runner_cls=ape.VelocityOnPolicyRunner)
    train = TrainConfig.from_task(task_id)
    train.env.scene.num_envs = args.worlds
    train.env.sim.mujoco.timestep = 0.002
    train.env.decimation = 10
    train.agent.max_iterations = args.updates
    train.agent.save_interval = 25
    train.agent.run_name = args.run_name
    train.agent.logger = "tensorboard"
    train.agent.resume = True
    train.agent.load_run = args.resume_run
    train.agent.load_checkpoint = args.resume_checkpoint
    launch_training(task_id, train)


if __name__ == "__main__":
    main()
