"""Commanded forward touchdown from a raised-foot reset, without a motion file.

This is the next half-step lesson after the collision-safe one-foot hold.  It
keeps Pollen's posture PPO environment and native joint action; the only new
task signal is a requested forward displacement of the swing sole relative to
the planted sole.  No reference joint positions are read during training or
inference.
"""

from copy import deepcopy
import json
from pathlib import Path

import torch
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from . import foot_upright, task


TASK_ID = "Mjlab-ForwardTouchdown-Ape-RightSupport"
NEAR_TASK_ID = "Mjlab-ForwardTouchdownNear-Ape-RightSupport"
RAISED_TASK_ID = "Mjlab-ForwardTouchdownRaised-Ape-RightSupport"
LANE_Y_M = 0.075
NEAR_POSE = (Path(__file__).resolve().parents[5] /
             "experiments/compact_morphology/motions/ape_touchdown_reset_left_t10.json")
RAISED_POSE = (Path(__file__).resolve().parents[5] /
               "experiments/compact_morphology/motions/ape_step_reset_left_t8.json")


def _swing_from_stance_b(env, asset_cfg: SceneEntityCfg):
    """Sole displacement in the robot heading frame; Pollen randomizes reset yaw."""
    robot = env.scene[asset_cfg.name]
    soles = robot.data.site_pos_w[:, asset_cfg.site_ids, :]
    delta = soles[:, 0, :2] - soles[:, 1, :2]
    w, x, y, z = robot.data.root_link_quat_w.unbind(dim=1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = torch.cos(yaw), torch.sin(yaw)
    return c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]


def foot_forward_target(env, asset_cfg: SceneEntityCfg, command_name: str, std: float):
    """Dense commanded foot displacement, in the planted sole's frame."""
    forward, _ = _swing_from_stance_b(env, asset_cfg)
    target_dx = env.command_manager.get_command(command_name)[:, 0]
    error = forward - target_dx
    return torch.exp(-0.5 * (error / std).square())


def swing_lateral_lane(env, asset_cfg: SceneEntityCfg, std: float):
    """Shape the outward swing needed before the feet can land side by side."""
    _, lateral = _swing_from_stance_b(env, asset_cfg)
    error = lateral - LANE_Y_M
    return torch.exp(-0.5 * (error / std).square())


def land_at_forward_target(
    env, asset_cfg: SceneEntityCfg, command_name: str, sensor_name: str,
    x_tolerance: float, y_tolerance: float,
):
    """Contact counts only at the requested x and the original lateral lane."""
    forward, lateral = _swing_from_stance_b(env, asset_cfg)
    target_dx = env.command_manager.get_command(command_name)[:, 0]
    x_ok = (forward - target_dx).abs() < x_tolerance
    # A physics-replayed forward step lands with about 75 mm separation.
    y_ok = (lateral - LANE_Y_M).abs() < y_tolerance
    found = env.scene.sensors[sensor_name].data.found
    return (x_ok & y_ok & (found[:, 0] > 0.5) & (found[:, 1] > 0.5)).float()


def env_cfg(play: bool = False):
    cfg = foot_upright.contact_safe_env_cfg(play=play, side="right")
    cfg.episode_length_s = 6.0
    # The inherited command is a three-slot Pollen velocity command.  Here its
    # first observed slot is a foot-placement goal, held for the whole episode.
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (0.03, 0.05)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.resampling_time_range = (10.0, 10.0)
    cfg.commands["twist"] = command
    for name in (
        "com_over_stance_foot", "swing_foot_clear", "swing_foot_touch",
        "pose_flamingo", "gravity_flamingo", "stillness",
    ):
        cfg.rewards.pop(name, None)
    soles = SceneEntityCfg("robot", site_names=("left_sole_center", "right_sole_center"))
    cfg.rewards["forward_foot_position"] = RewardTermCfg(
        func=foot_forward_target, weight=3.0,
        params={"asset_cfg": soles, "command_name": "twist", "std": 0.02},
    )
    cfg.rewards["swing_lateral_lane"] = RewardTermCfg(
        func=swing_lateral_lane, weight=2.0,
        params={"asset_cfg": soles, "std": 0.04},
    )
    cfg.rewards["land_in_target"] = RewardTermCfg(
        func=land_at_forward_target, weight=3.0,
        params={"asset_cfg": soles, "command_name": "twist",
                "sensor_name": "feet_ground_contact", "x_tolerance": 0.012,
                "y_tolerance": 0.015},
    )
    cfg.rewards["foot_slip"].weight = -1.0
    cfg.rewards["stance_foot_grounded"].weight = 1.0
    cfg.rewards["self_body_collision"].weight = -1.0
    # No pose script: native upright and action/effort/limit terms remain.
    cfg.events["set_flamingo_state"].params["standing_prob"] = 0.0
    cfg.curriculum.pop("flamingo_com_std", None)
    return cfg


def pose_reset_env_cfg(pose_path: Path, play: bool = False):
    """Start from one measured airborne pose; never read later motion frames."""
    cfg = env_cfg(play=play)
    pose = json.loads(pose_path.read_text())
    values = [pose["joints"][name] for name in foot_upright.JOINT_NAMES]
    _, _, z, roll, pitch = pose["root_xyz_roll_pitch"]
    cfg.events["set_flamingo_state"].params.update({
        "joint_pose": values, "base_roll": roll, "base_pitch": pitch,
        "z_min": z, "z_max": z + 0.002,
        "tilt_noise": 0.005, "joint_noise_std": 0.005,
    })
    cfg.commands["twist"].ranges.lin_vel_x = (0.035, 0.045)
    return cfg


def near_env_cfg(play: bool = False):
    return pose_reset_env_cfg(NEAR_POSE, play=play)


def raised_env_cfg(play: bool = False):
    return pose_reset_env_cfg(RAISED_POSE, play=play)


rl_cfg = deepcopy(foot_upright.right_safe_runner_cfg)
rl_cfg.experiment_name = "ape_forward_touchdown"
rl_cfg.run_name = "command_from_raised_foot_v1"
rl_cfg.max_iterations = 300
rl_cfg.save_interval = 50
register_mjlab_task(
    task_id=TASK_ID, env_cfg=env_cfg(), play_env_cfg=env_cfg(play=True),
    rl_cfg=rl_cfg, runner_cls=VelocityOnPolicyRunner,
)
near_rl_cfg = deepcopy(rl_cfg)
near_rl_cfg.experiment_name = "ape_forward_touchdown_near"
near_rl_cfg.run_name = "body_frame_v2"
register_mjlab_task(
    task_id=NEAR_TASK_ID,
    env_cfg=near_env_cfg(), play_env_cfg=near_env_cfg(play=True),
    rl_cfg=near_rl_cfg, runner_cls=VelocityOnPolicyRunner,
)
raised_rl_cfg = deepcopy(near_rl_cfg)
raised_rl_cfg.experiment_name = "ape_forward_touchdown_raised"
raised_rl_cfg.run_name = "from_t8_airborne_v1"
register_mjlab_task(
    task_id=RAISED_TASK_ID,
    env_cfg=raised_env_cfg(), play_env_cfg=raised_env_cfg(play=True),
    rl_cfg=raised_rl_cfg, runner_cls=VelocityOnPolicyRunner,
)
