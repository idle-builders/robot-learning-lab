"""Bounded native PPO continuation on the selected direct-frame control body.

The variation is an explicit sensitivity distribution, not a measured hardware
uncertainty interval. Resolved MJLab config and run checkpoint remain in logs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
import json
import math
from pathlib import Path

import mujoco
import torch
from mjlab.envs.mdp import dr
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scripts.train import TrainConfig, launch_training

from ape_walk import task as ape
from ape_walk.physics.frame50 import (
    clip_frame50_leg_targets, configure_frame50_motors, motor_configs,
    give_soles_contact_priority, require_direct_frame,
)


class MoveThenStopCommand(ape.microduck_mdp.VelocityCommandCommandOnly):
    """Begin each episode moving, then command a true same-episode stop."""

    def __init__(self, cfg, env):
        self._fresh = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        self._stopped = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._just_stopped = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        super().__init__(cfg, env)

    @property
    def just_stopped(self):
        return self._just_stopped

    def reset(self, env_ids=None):
        ids = env_ids if env_ids is not None else slice(None)
        self._fresh[ids] = True
        self._stopped[ids] = False
        self._just_stopped[ids] = False
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        first = self._fresh[env_ids].clone()
        super()._resample_command(env_ids)
        stop = ~first & ~self._stopped[env_ids]
        self._stopped[env_ids] |= stop
        if bool(self._stopped[env_ids].any()):
            stop_ids = env_ids[self._stopped[env_ids]]
            self.vel_command_b[stop_ids] = 0.0
            self.vel_command_w[stop_ids] = 0.0
            self.is_standing_env[stop_ids] = True
        self._just_stopped[env_ids] = stop
        self._fresh[env_ids] = False


@dataclass(kw_only=True)
class MoveThenStopCommandCfg(ape.microduck_mdp.VelocityCommandCommandOnlyCfg):
    class_type: type = MoveThenStopCommand

    def build(self, env):
        return MoveThenStopCommand(self, env)


class CardinalMoveThenStopCommand(MoveThenStopCommand):
    """Expose each basic planar direction before a same-episode stop."""

    def _resample_command(self, env_ids):
        first = self._fresh[env_ids].clone()
        super()._resample_command(env_ids)
        moving_ids = env_ids[first]
        if len(moving_ids) == 0:
            return
        draw = torch.rand(len(moving_ids), device=self.device)
        speed = torch.rand(len(moving_ids), device=self.device)
        yaw = (torch.rand(len(moving_ids), device=self.device) * 2.0 - 1.0) * 0.15
        backward = draw < 0.30
        left = (draw >= 0.30) & (draw < 0.55)
        right = (draw >= 0.55) & (draw < 0.80)
        forward = draw >= 0.80
        self.vel_command_b[moving_ids] = 0.0
        self.vel_command_b[moving_ids[backward], 0] = -(0.20 + 0.20 * speed[backward])
        self.vel_command_b[moving_ids[left], 1] = 0.12 + 0.10 * speed[left]
        self.vel_command_b[moving_ids[right], 1] = -(0.12 + 0.10 * speed[right])
        self.vel_command_b[moving_ids[forward], 0] = 0.20 + 0.20 * speed[forward]
        self.vel_command_b[moving_ids, 2] = yaw
        self.vel_command_w[moving_ids] = self.vel_command_b[moving_ids]
        self.is_standing_env[moving_ids] = False


@dataclass(kw_only=True)
class CardinalMoveThenStopCommandCfg(MoveThenStopCommandCfg):
    class_type: type = CardinalMoveThenStopCommand

    def build(self, env):
        return CardinalMoveThenStopCommand(self, env)


class SharedForwardTurnCommand(ape.microduck_mdp.VelocityCommandCommandOnly):
    """Balanced forward, left, right and sparse zero commands with live resampling."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        n = len(env_ids)
        if not n:
            return
        draw = torch.rand(n, device=self.device)
        magnitude = torch.rand(n, device=self.device)
        moving = draw < 0.90
        forward = moving & (draw < 0.30)
        left = moving & (draw >= 0.30) & (draw < 0.60)
        right = moving & (draw >= 0.60)
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids[forward], 0] = 0.20 + 0.20 * magnitude[forward]
        self.vel_command_b[env_ids[left], 2] = 0.25 + 0.30 * magnitude[left]
        self.vel_command_b[env_ids[right], 2] = -(0.25 + 0.30 * magnitude[right])
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
        self.is_standing_env[env_ids] = ~moving


@dataclass(kw_only=True)
class SharedForwardTurnCommandCfg(ape.microduck_mdp.VelocityCommandCommandOnlyCfg):
    class_type: type = SharedForwardTurnCommand

    def build(self, env):
        return SharedForwardTurnCommand(self, env)


def rated_leg_effort_overage(env, asset_cfg: SceneEntityCfg,
                            rated_nm: tuple[float, ...], onset_ratio: float = 0.7,
                            joint_indices: tuple[int, ...] = tuple(range(10))):
    """Penalize selected leg drives before they reach rated torque."""
    forces = env.scene[asset_cfg.name].data.actuator_force[:, list(joint_indices)]
    limits = torch.as_tensor(rated_nm, device=forces.device, dtype=forces.dtype)
    return torch.square(torch.relu(forces.abs() / limits - onset_ratio)).sum(dim=1)


def moving_load_overlap(env, command_name: str = "twist",
                        min_command_m_s: float = 0.18,
                        min_speed_m_s: float = 0.12,
                        per_foot_load_n: float = 8.0):
    """Reward a short, genuinely loaded two-sole phase during forward gait.

    The source actor spends nearly all moving time in exclusive support. This
    term asks for overlapping load while already translating, so standing in
    place receives zero. It is a provisional simulation reward, not a force
    sensor requirement for the physical robot.
    """
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w[:, :2], dim=1)
    feet = env.scene["feet_ground_contact"].data
    both_found = (feet.found[:, 0] > 0) & (feet.found[:, 1] > 0)
    vertical_load = feet.force[:, :, 2].abs()
    shared_load = torch.clamp(vertical_load.min(dim=1).values / per_foot_load_n, 0.0, 1.0)
    moving = (command[:, 0] >= min_command_m_s).to(speed.dtype)
    progress = torch.clamp((speed - min_speed_m_s) / 0.1, 0.0, 1.0)
    return moving * both_found.to(speed.dtype) * progress * shared_load


def signed_directional_progress(env, command_name: str = "twist"):
    """Reward sustained body-frame progress along a nonzero planar command.

    The source Gaussian tracking reward remains high when the robot parks at a
    slow side command. This term is zero when parked, negative when moving the
    wrong way, and capped to avoid buying reward through uncontrolled speed.
    Contact, fall and motor costs remain separate; this is not a task score.
    """
    command = env.command_manager.get_command(command_name)[:, :2]
    velocity = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    command_norm2 = torch.sum(command.square(), dim=1)
    projected_ratio = torch.sum(command * velocity, dim=1) / command_norm2.clamp_min(0.01)
    return torch.where(command_norm2 >= 0.01,
                       projected_ratio.clamp(-1.25, 1.25),
                       torch.zeros_like(projected_ratio))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=25,
                        help="PPO update interval for intermediate checkpoint saves")
    parser.add_argument("--run-name", default="cove50-varied-forward")
    parser.add_argument("--profile", choices=("forward", "backward", "planar", "turn", "stop", "cardinal", "cardinal_move", "shared_forward_turn"), default="forward")
    parser.add_argument("--forward-speed-range", type=float, nargs=2,
                        help="Override the forward command curriculum in m/s")
    parser.add_argument("--forward-yaw-range", type=float, nargs=2,
                        help="Override the forward yaw command curriculum in rad/s")
    parser.add_argument("--forward-standing-fraction", type=float, default=0.10,
                        help="Fraction of zero-command worlds in forward training")
    parser.add_argument("--rail-v", type=float, default=12.0)
    parser.add_argument("--mass-range", type=float, nargs=2, default=(0.9, 1.1))
    parser.add_argument("--grip-range", type=float, nargs=2, default=(0.65, 1.15))
    parser.add_argument("--effort-range", type=float, nargs=2, default=(0.9, 1.05))
    parser.add_argument("--pd-range", type=float, nargs=2, default=(0.9, 1.1))
    parser.add_argument("--delay-max-steps", type=int, default=4)
    parser.add_argument("--tight-heading", action="store_true")
    parser.add_argument("--effort-weight", type=float, default=0.0)
    parser.add_argument("--rated-effort-penalty-weight", type=float, default=0.0,
                        help="Soft penalty on leg effort above 70%% of catalog-rated torque")
    parser.add_argument("--hip-roll-rated-penalty-weight", type=float, default=0.0,
                        help="Soft penalty on just the two hip-roll drives above 70%% rated torque")
    parser.add_argument("--moving-load-overlap-weight", type=float, default=0.0,
                        help="Forward-only loaded double-support shaping while already moving")
    parser.add_argument("--directional-progress-weight", type=float, default=0.0,
                        help="Cardinal movement reward for measured progress along the signed command")
    parser.add_argument("--walking-hip-roll-pose-std", type=float,
                        help="Relax the walking/running hip-roll pose prior; standing stays at its source value")
    parser.add_argument("--action-limit-margin-rad", type=float,
                        help="Clip leg position targets inside the Cove joint limits")
    parser.add_argument("--ankle-target-margin-rad", type=float,
                        help="Clip only ankle position targets inside the Cove joint limits")
    parser.add_argument("--resume-run", help="Existing run to continue; omit for a scratch benchmark or new actor")
    parser.add_argument("--resume-checkpoint", default="model_497.pt")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Compile and step four local CPU worlds; do not train")
    args = parser.parse_args()
    if args.forward_speed_range and not (0 < args.forward_speed_range[0] < args.forward_speed_range[1]):
        parser.error("--forward-speed-range must be positive and increasing")
    if args.save_interval < 1:
        parser.error("--save-interval must be positive")
    if args.forward_yaw_range and args.forward_yaw_range[0] > args.forward_yaw_range[1]:
        parser.error("--forward-yaw-range must be increasing")
    if not 0 <= args.forward_standing_fraction <= 1:
        parser.error("--forward-standing-fraction must be between 0 and 1")
    if args.action_limit_margin_rad is not None and args.ankle_target_margin_rad is not None:
        parser.error("Choose either all-leg or ankle-only target clipping")
    if args.moving_load_overlap_weight < 0 or (args.moving_load_overlap_weight and args.profile != "forward"):
        parser.error("--moving-load-overlap-weight must be nonnegative and forward-only")
    if args.directional_progress_weight < 0 or (args.directional_progress_weight and args.profile != "cardinal_move"):
        parser.error("--directional-progress-weight must be nonnegative and cardinal_move-only")
    if args.walking_hip_roll_pose_std is not None:
        if args.profile not in ("forward", "stop", "cardinal_move") or args.walking_hip_roll_pose_std <= 0:
            parser.error("--walking-hip-roll-pose-std must be positive and forward/stop/cardinal_move-only")
        if args.moving_load_overlap_weight:
            parser.error("Use one changed gait-shaping factor per exploratory branch")
    body = args.body_xml.resolve()
    model = require_direct_frame(body)
    task_id = "Mjlab-Velocity-Flat-Ape-Cove50-Varied"
    env = ape.env_cfg(elbowed=True, motion=(
        "planar" if args.profile in ("turn", "cardinal", "cardinal_move", "shared_forward_turn") else
        "signed" if args.profile == "stop" else args.profile
    ))
    if args.walking_hip_roll_pose_std is not None:
        for phase in ("std_walking", "std_running"):
            env.rewards["pose"].params[phase][r".*hip_roll.*"] = args.walking_hip_roll_pose_std
    env.scene.entities["robot"].spec_fn = partial(ape._ape_spec, body)
    env.scene.entities["robot"].collisions[0].disable_other_geoms = False
    give_soles_contact_priority(env)
    configure_frame50_motors(env, body, rail_v=args.rail_v, delay_s=0.0)
    # The upstream actuator buffer samples a fresh 0..4 physics-step command
    # delay per world/step. The distribution is unmeasured sensitivity.
    env.scene.entities["robot"].articulation.actuators = tuple(
        replace(act, delay_min_lag=0, delay_max_lag=args.delay_max_steps)
        for act in env.scene.entities["robot"].articulation.actuators
    )
    env.scene.entities["robot"].init_state.pos = (0.0, 0.0, 0.231)
    env.actions["joint_pos"].scale.pop("^head_pitch$", None)
    if args.action_limit_margin_rad is not None:
        clip_frame50_leg_targets(env, body, margin_rad=args.action_limit_margin_rad)
    if args.ankle_target_margin_rad is not None:
        clip_frame50_leg_targets(env, body, margin_rad=args.ankle_target_margin_rad,
                                 axes=("ankle_pitch",))
    pose = env.scene.entities["robot"].init_state.joint_pos
    pose[r".*shoulder_pitch.*"] = 0.0
    pose[r".*elbow_pitch.*"] = 0.0

    # Command and reset variance. Zero-command worlds are a real stopping
    # lesson, while small yaw commands expose path drift in the actor's input.
    command = env.commands["twist"]
    if args.profile == "forward":
        command.ranges.lin_vel_x = tuple(args.forward_speed_range or (0.2, 0.45))
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = tuple(args.forward_yaw_range or (-0.35, 0.35))
        command.rel_standing_envs = args.forward_standing_fraction
    elif args.profile == "backward":
        command.ranges.lin_vel_x = (-0.45, -0.2)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.2, 0.2)
        command.rel_standing_envs = 0.10
        env.rewards["stillness_at_zero"] = RewardTermCfg(
            func=ape.microduck_mdp.stillness_at_zero_command,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("robot"),
                    "command_name": "twist", "vel_std": 0.1},
        )
    elif args.profile == "turn":
        # A dedicated turn/settle actor can spend the whole lesson on the
        # signed yaw command and a real zero-command dwell. Its observation
        # and action contract remains the retained planar actor's contract.
        command.ranges.lin_vel_x = (0.0, 0.0)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.55, 0.55)
        command.rel_standing_envs = 0.30
        command.rel_turn_in_place_envs = 0.70
        command.resampling_time_range = (2.5, 4.5)
        env.curriculum.pop("standing_envs", None)
        env.rewards["stillness_at_zero"] = RewardTermCfg(
            func=ape.microduck_mdp.stillness_at_zero_command,
            weight=3.0,
            params={"asset_cfg": SceneEntityCfg("robot"),
                    "command_name": "twist", "vel_std": 0.1},
        )
        env.rewards["joint_torques_l2"] = RewardTermCfg(
            func=ape.microduck_mdp.joint_torques_l2, weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
    elif args.profile == "stop":
        # Each episode starts with a real move command. Its first 3–4 s
        # resample forces zero, leaving a 4–5 s settling tail in the same
        # physical rollout. The evaluator still checks actual prior motion.
        command.ranges.lin_vel_x = (0.2, 0.45)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.2, 0.2)
        command.rel_standing_envs = 0.0
        command.resampling_time_range = (3.0, 4.0)
        env.commands["twist"] = MoveThenStopCommandCfg(
            **{**vars(command), "class_type": MoveThenStopCommand}
        )
        env.curriculum.pop("standing_envs", None)
        env.rewards["stillness_at_zero"] = RewardTermCfg(
            func=ape.microduck_mdp.stillness_at_zero_command,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("robot"),
                    "command_name": "twist", "vel_std": 0.1},
        )
    elif args.profile == "shared_forward_turn":
        command.ranges.lin_vel_x = (0.0, 0.40)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (-0.55, 0.55)
        command.rel_standing_envs = 0.0
        command.rel_turn_in_place_envs = 0.0
        command.rel_forward_envs = 0.0
        command.rel_world_envs = 0.0
        command.rel_heading_envs = 0.0
        command.init_velocity_prob = 0.0
        command.resampling_time_range = (3.0, 4.0)
        env.commands["twist"] = SharedForwardTurnCommandCfg(
            **{**vars(command), "class_type": SharedForwardTurnCommand}
        )
        env.curriculum.pop("standing_envs", None)
        env.rewards["stillness_at_zero"] = RewardTermCfg(
            func=ape.microduck_mdp.stillness_at_zero_command,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("robot"),
                    "command_name": "twist", "vel_std": 0.1},
        )
        env.rewards["joint_torques_l2"] = RewardTermCfg(
            func=ape.microduck_mdp.joint_torques_l2, weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
    elif args.profile in ("cardinal", "cardinal_move"):
        # V1 used 3–4 s movement followed by zero. V2 isolates command
        # exposure: same directional sampler, reward and variation, but a
        # 9 s resample interval keeps each 8 s episode moving throughout.
        command.ranges.lin_vel_x = (-0.40, 0.40)
        command.ranges.lin_vel_y = (-0.22, 0.22)
        command.ranges.ang_vel_z = (-0.15, 0.15)
        command.rel_standing_envs = 0.0
        command.rel_turn_in_place_envs = 0.0
        command.rel_forward_envs = 0.0
        command.rel_world_envs = 0.0
        command.init_velocity_prob = 0.0
        command.resampling_time_range = (9.0, 9.0) if args.profile == "cardinal_move" else (3.0, 4.0)
        env.commands["twist"] = CardinalMoveThenStopCommandCfg(
            **{**vars(command), "class_type": CardinalMoveThenStopCommand}
        )
        env.curriculum.pop("standing_envs", None)
        env.rewards["stillness_at_zero"] = RewardTermCfg(
            func=ape.microduck_mdp.stillness_at_zero_command,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("robot"),
                    "command_name": "twist", "vel_std": 0.1},
        )
        env.rewards["joint_torques_l2"] = RewardTermCfg(
            func=ape.microduck_mdp.joint_torques_l2, weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
    else:
        command.ranges.lin_vel_x = (-0.4, 0.45)
        command.ranges.lin_vel_y = (-0.2, 0.2)
        command.ranges.ang_vel_z = (-0.7, 0.7)
        command.rel_standing_envs = 0.12
        command.rel_turn_in_place_envs = 0.18
        env.curriculum.pop("standing_envs", None)
        env.rewards["stillness_at_zero"] = RewardTermCfg(
            func=ape.microduck_mdp.stillness_at_zero_command,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("robot"),
                    "command_name": "twist", "vel_std": 0.1},
        )
        env.rewards["joint_torques_l2"] = RewardTermCfg(
            func=ape.microduck_mdp.joint_torques_l2, weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
    if args.tight_heading:
        env.rewards["track_angular_velocity"].weight = 4.0
        env.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.1)
    if args.moving_load_overlap_weight:
        env.rewards["moving_load_overlap"] = RewardTermCfg(
            func=moving_load_overlap,
            weight=args.moving_load_overlap_weight,
            params={"command_name": "twist"},
        )
    if args.directional_progress_weight:
        env.rewards["signed_directional_progress"] = RewardTermCfg(
            func=signed_directional_progress,
            weight=args.directional_progress_weight,
            params={"command_name": "twist"},
        )
    if args.effort_weight:
        if args.effort_weight >= 0:
            raise ValueError("Effort penalty must be negative")
        env.rewards["joint_torques_l2"] = RewardTermCfg(
            func=ape.microduck_mdp.joint_torques_l2,
            weight=args.effort_weight,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
    if args.rated_effort_penalty_weight:
        if args.rated_effort_penalty_weight >= 0:
            raise ValueError("Rated-effort penalty must be negative")
        expected = [f"{side}_{axis}" for side in ("left", "right") for axis in (
            "hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")]
        actual = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                  int(model.actuator_trnid[i, 0])) for i in range(10)]
        if actual != expected:
            raise ValueError(f"Selected leg actuator order changed: {actual}")
        rated = tuple(float(act.effort_limit) for act in motor_configs(body, rail_v=args.rail_v)[:10])
        env.rewards["rated_leg_effort_overage"] = RewardTermCfg(
            func=rated_leg_effort_overage,
            weight=args.rated_effort_penalty_weight,
            params={"asset_cfg": SceneEntityCfg("robot"), "rated_nm": rated},
        )
    if args.hip_roll_rated_penalty_weight:
        if args.hip_roll_rated_penalty_weight >= 0:
            raise ValueError("Hip-roll rated-effort penalty must be negative")
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                 int(model.actuator_trnid[i, 0])) for i in (1, 6)]
        if names != ["left_hip_roll", "right_hip_roll"]:
            raise ValueError(f"Selected hip-roll actuator order changed: {names}")
        rated = tuple(float(motor_configs(body, rail_v=args.rail_v)[i].effort_limit)
                      for i in (1, 6))
        env.rewards["rated_hip_roll_overage"] = RewardTermCfg(
            func=rated_leg_effort_overage,
            weight=args.hip_roll_rated_penalty_weight,
            params={"asset_cfg": SceneEntityCfg("robot"), "rated_nm": rated,
                    "joint_indices": (1, 6)},
        )
    env.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.006),
        "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05),
        "yaw": (-0.15, 0.15),
    }
    env.events["reset_robot_joints"].params["position_range"] = (-0.04, 0.04)
    env.events["reset_robot_joints"].params["velocity_range"] = (-0.15, 0.15)

    body_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                       for i in range(1, model.nbody))
    env.events["cove_link_density"] = EventTermCfg(
        mode="reset", func=dr.pseudo_inertia,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=body_names),
                "alpha_range": (math.log(args.mass_range[0]) / 2,
                                math.log(args.mass_range[1]) / 2)},
    )
    env.events["cove_sole_friction"] = EventTermCfg(
        mode="reset", func=dr.geom_friction,
        params={"asset_cfg": SceneEntityCfg("robot", geom_names=("left_sole", "right_sole")),
                "operation": "abs", "ranges": tuple(args.grip_range),
                "axes": [0], "shared_random": True},
    )
    env.events["cove_effort"] = EventTermCfg(
        mode="reset", func=dr.effort_limits,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "effort_limit_range": tuple(args.effort_range),
                "operation": "scale"},
    )
    env.events["cove_pd"] = EventTermCfg(
        mode="reset", func=dr.pd_gains,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "kp_range": tuple(args.pd_range),
                "kd_range": tuple(args.pd_range),
                "operation": "scale"},
    )

    agent = deepcopy(ape.elbow_ape_rl_cfg)
    agent.experiment_name = "ape_cove50_velocity"
    ape.register_mjlab_task(task_id=task_id, env_cfg=env, play_env_cfg=env,
                            rl_cfg=agent, runner_cls=ape.VelocityOnPolicyRunner)
    cfg = TrainConfig.from_task(task_id)
    cfg.env.scene.num_envs = args.worlds
    cfg.env.sim.mujoco.timestep = 0.002
    cfg.env.decimation = 10
    cfg.agent.max_iterations = args.updates
    cfg.agent.save_interval = args.save_interval
    cfg.agent.run_name = args.run_name
    cfg.agent.logger = "tensorboard"
    if args.resume_run:
        cfg.agent.resume = True
        cfg.agent.load_run = args.resume_run
        cfg.agent.load_checkpoint = args.resume_checkpoint
    else:
        cfg.agent.resume = False
    if args.preflight_only:
        cfg.env.scene.num_envs = min(args.worlds, 128)
        cfg.env.auto_reset = False
        cfg.env.seed = 99
        local = ManagerBasedRlEnv(cfg=cfg.env, device="cpu")
        observation, _ = local.reset()
        report = {
            "schema": "robot-learning-lab.cove50-velocity-entry.v1",
            "profile": args.profile,
            "body": str(body),
            "actor_shape": list(observation["actor"].shape),
            "critic_shape": list(observation["critic"].shape),
            "action_shape": list(local.action_space.shape),
            "runtime_integrator": int(local.sim.mj_model.opt.integrator),
            "configured_integrator": cfg.env.sim.mujoco.integrator,
            "actor_first_12": observation["actor"][0, :12].tolist(),
            "initial_command": local.command_manager.get_command("twist")[:, :3].tolist(),
            "claim_limit": "Local CPU setup worlds only; no CUDA score, optimizer update or learned skill.",
        }
        local.step(torch.zeros(local.action_space.shape))
        if args.profile in ("stop", "cardinal"):
            command_term = local.command_manager.get_term("twist")
            ids = torch.arange(local.num_envs, device=local.device)
            command_term._resample_command(ids)
            report["first_resample_command"] = command_term.command[:, :3].tolist()
            report["first_resample_just_stopped"] = command_term.just_stopped.tolist()
            command_term._resample_command(ids)
            report["second_resample_command"] = command_term.command[:, :3].tolist()
            report["second_resample_just_stopped"] = command_term.just_stopped.tolist()
            if (any(any(abs(v) > 1e-9 for v in row)
                    for row in report["first_resample_command"])
                    or not all(report["first_resample_just_stopped"])):
                raise ValueError("Move command did not stop at first resample")
            if any(any(abs(v) > 1e-9 for v in row)
                   for row in report["second_resample_command"]):
                raise ValueError("Stop command restarted movement inside an episode")
        if args.profile == "cardinal_move":
            report["command_resampling_time_range_s"] = list(command.resampling_time_range)
            report["episode_length_s"] = cfg.env.episode_length_s
            if min(command.resampling_time_range) <= cfg.env.episode_length_s:
                raise ValueError("Movement-only command resamples before episode end")
        if args.profile == "shared_forward_turn":
            initial = local.command_manager.get_command("twist")[:, :3]
            report["command_bucket_counts"] = {
                "forward": int((initial[:, 0] > 0).sum()),
                "left": int((initial[:, 2] > 0).sum()),
                "right": int((initial[:, 2] < 0).sum()),
                "zero": int(((initial[:, 0] == 0) & (initial[:, 2] == 0)).sum()),
            }
        local.close()
        print(json.dumps(report, indent=2))
        return
    launch_training(task_id, cfg)


if __name__ == "__main__":
    main()
