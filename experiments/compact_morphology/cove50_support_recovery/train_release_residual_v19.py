"""V19 bounded upright-to-foot-release PPO residual on the Direct Frame.

This always starts upright. The zero residual follows the validated two-foot
load reference; the seven-channel policy adds coupled swing-leg release.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import imageio.v2 as imageio
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs.mdp import dr
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sim.sim_data import TorchArray
from mjlab.managers import EventTermCfg, MetricsTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.algorithms import PPO

from ape_walk import task as ape
from ape_walk.physics.frame50 import (
    clip_frame50_leg_targets, configure_frame50_motors,
    give_soles_contact_priority, require_direct_frame,
    reveal_frame50_proxy_geometry_for_video,
)
from release_reference_action import (
    RELEASE_RESIDUAL_JOINTS, ReleaseReferenceActionCfg,
)


# The last three channels are mapped to the requested swing leg by the action
# adapter. Their pulse authority is a prefit measurement, not assumed success.
REFERENCE_RESIDUAL_JOINTS = RELEASE_RESIDUAL_JOINTS


class CoveSupportCycleCommand(ape.microduck_mdp.FlamingoCommand):
    """Expose the evaluator's stand → requested side → stand command in each episode."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # The reset event calls request_side before command reset. Super uses
        # that pending side, while its random flag must not reach actor frame 0.
        super()._resample_command(env_ids)
        self.vel_command_b[env_ids, 0] = 0.0
        self.vel_command_b[env_ids, 1] = 0.0

    def _update_command(self) -> None:
        t = self._env_ref.episode_length_buf * self._env_ref.step_dt
        hold = (t >= 1.0) & (t < 5.0)
        self.vel_command_b[:, 0] = hold.float()
        self.vel_command_b[:, 1] = torch.where(
            hold, self._stance_side, torch.zeros_like(self._stance_side))


@dataclass(kw_only=True)
class CoveSupportCycleCommandCfg(ape.microduck_mdp.FlamingoCommandCfg):
    class_type: type = CoveSupportCycleCommand

    def build(self, env: ManagerBasedRlEnv) -> CoveSupportCycleCommand:
        return CoveSupportCycleCommand(self, env)


def support_skill_time(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Current 50 Hz reference index in seconds, including the reset frame."""
    return (env.episode_length_buf * env.step_dt).float().unsqueeze(-1)


def support_latched_side(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Observable request retained through idle and return (right=+1)."""
    return env.command_manager.get_term("twist").stance_side.unsqueeze(-1)


def support_reference_error(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Current deployable joint positions minus the decoded source reference."""
    action = env.action_manager.get_term("joint_pos")
    return env.scene["robot"].data.joint_pos[:, action.target_ids] - action.reference_target()


def support_action_over_limit(env: ManagerBasedRlEnv, overshoot: float = 0.1) -> torch.Tensor:
    """Evaluate the full 21D target before the inherited physical joint clip."""
    action = env.action_manager.get_term("joint_pos")
    hard = env.scene["robot"].data.joint_pos_limits[:, action.target_ids]
    target = action.last_unclipped_target
    excess = (target - (hard[..., 1] + overshoot)).clamp_min(0.0)
    excess += ((hard[..., 0] - overshoot) - target).clamp_min(0.0)
    return excess.sum(dim=-1)


def commanded_clear_support(env, left_cfg: SceneEntityCfg, right_cfg: SceneEntityCfg,
                            clearance_m: float = 0.025):
    """Release-window exclusive support on the native 50 Hz skill clock."""
    command = env.command_manager.get_term("twist")
    right_stance = command.stance_side > 0
    found = env.scene.sensors["feet_ground_contact"].data.found.bool()
    stance = torch.where(right_stance, found[:, 1], found[:, 0])
    swing = torch.where(right_stance, found[:, 0], found[:, 1])
    robot = env.scene["robot"]
    left_z = robot.data.site_pos_w[:, left_cfg.site_ids[0], 2]
    right_z = robot.data.site_pos_w[:, right_cfg.site_ids[0], 2]
    height = torch.where(right_stance, left_z, right_z) - env.scene.terrain.env_origins[:, 2]
    t = env.episode_length_buf * env.step_dt
    return ((t >= 2.0) & (t < 5.0) & stance & ~swing
            & (height >= clearance_m)).float()


def commanded_two_foot_load(env, minimum_fraction: float = 0.75):
    """Stage-A reward: requested foot carries load while both soles keep contact."""
    command = env.command_manager.get_term("twist")
    feet = env.scene.sensors["feet_ground_contact"].data
    force = feet.force.norm(dim=-1)
    stance_force = torch.where(command.stance_side > 0, force[:, 1], force[:, 0])
    load = stance_force / force.sum(dim=-1).clamp_min(1.0)
    robot = env.scene["robot"]
    quat = robot.data.root_link_pose_w[:, 3:7]
    upright = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
    clear_body = ~env.scene.sensors["nonsole_ground_contact"].data.found.any(dim=-1)
    clear_self = ~env.scene.sensors["self_body_contact"].data.found.any(dim=-1)
    clear_hand = env.scene.sensors["hand_ground_contact"].data.force.norm(dim=-1).amax(dim=-1) < 1.0
    return ((command.flag > .5) & feet.found[:, 0].bool() & feet.found[:, 1].bool() &
            (load >= minimum_fraction) & (robot.data.root_link_pos_w[:, 2] >= .20) &
            (upright >= math.cos(math.radians(15))) & clear_body & clear_self &
            clear_hand).float()


def release_entry_load_reward(env):
    t = env.episode_length_buf * env.step_dt
    return (((t >= 1.0) & (t <= 2.2)).float()
            * commanded_two_foot_load(env))


def standing_two_foot_return(env):
    """Terminal state component; the evaluator checks the full support cycle."""
    command = env.command_manager.get_term("twist")
    found = env.scene.sensors["feet_ground_contact"].data.found.bool()
    return ((command.flag < 0.5) & (command.alpha < 0.1)
            & found[:, 0] & found[:, 1]).float()


def rated_leg_frame(env, rated_nm: tuple[float, ...]):
    force = env.scene["robot"].data.actuator_force[:, :10].abs()
    limits = torch.as_tensor(rated_nm, device=force.device, dtype=force.dtype)
    return (force >= .995 * limits).any(dim=1).float()


def release_contact_state(env, clearance_m: float = .025):
    """Online learning proxy; native scoring checks the full stance patch."""
    command = env.command_manager.get_term("twist")
    right = command.stance_side > 0
    feet = env.scene.sensors["feet_ground_contact"].data.found.bool()
    stance = torch.where(right, feet[:, 1], feet[:, 0])
    swing = torch.where(right, feet[:, 0], feet[:, 1])
    robot = env.scene["robot"]
    sites = robot.data.site_pos_w
    left_id = robot.site_names.index("left_sole_center")
    right_id = robot.site_names.index("right_sole_center")
    swing_z = torch.where(right, sites[:, left_id, 2], sites[:, right_id, 2])
    height = swing_z - env.scene.terrain.env_origins[:, 2]
    quat = robot.data.root_link_pose_w[:, 3:7]
    upright = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
    sensors = env.scene.sensors
    clear = (~sensors["nonsole_ground_contact"].data.found.any(dim=-1)
             & ~sensors["self_body_contact"].data.found.any(dim=-1)
             & (sensors["hand_ground_contact"].data.force.norm(dim=-1).amax(dim=-1) < 1.0))
    stable = (stance & (robot.data.root_link_pos_w[:, 2] >= .20)
              & (upright >= math.cos(math.radians(15))) & clear)
    exclusive = stable & ~swing & (height >= clearance_m)
    return stable, swing, height, exclusive


class AttainedLoadReleaseProgress:
    """Dense release progress only after a physically attained two-foot load."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self.ready = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.load_frames = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    def __call__(self, env: ManagerBasedRlEnv):
        t = env.episode_length_buf * env.step_dt
        loaded = ((t >= 1.8) & (t <= 2.2)
                  & (commanded_two_foot_load(env) > .5))
        self.load_frames = torch.where(loaded, self.load_frames + 1, 0)
        self.ready |= self.load_frames >= 10
        stable, swing_contact, height, _ = release_contact_state(env)
        # A grounded sole can rock upward at its center. It receives no
        # clearance progress until the native contact sensor loses contact.
        progress = (height / .025).clamp(0, 1)
        progress = torch.where(swing_contact, 0.0, progress)
        return torch.where(self.ready & stable & (t >= 2) & (t < 5),
                           progress, 0.0)

    def reset(self, env_ids=None):
        ids = env_ids if env_ids is not None else slice(None)
        self.ready[ids] = False
        self.load_frames[ids] = 0


class AttainedLoadExclusiveSupport:
    """Only pay release frames after the commanded two-foot load was reached."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self.ready = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.load_frames = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    def __call__(self, env: ManagerBasedRlEnv):
        t = env.episode_length_buf * env.step_dt
        loaded = ((t >= 1.8) & (t <= 2.2)
                  & (commanded_two_foot_load(env) > .5))
        self.load_frames = torch.where(loaded, self.load_frames + 1, 0)
        self.ready |= self.load_frames >= 10
        _, _, _, exclusive = release_contact_state(env)
        return (self.ready & exclusive & (t >= 2) & (t < 5)).float()

    def reset(self, env_ids=None):
        ids = env_ids if env_ids is not None else slice(None)
        self.ready[ids] = False
        self.load_frames[ids] = 0


class ReleaseDurationAndReturn:
    """Pay once at measured support durations and again for a clean return."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        n = env.num_envs
        self.current = torch.zeros(n, device=env.device)
        self.longest = torch.zeros(n, device=env.device)
        self.paid_half = torch.zeros(n, device=env.device, dtype=torch.bool)
        self.paid_one = torch.zeros_like(self.paid_half)
        self.paid_two = torch.zeros_like(self.paid_half)
        self.paid_return = torch.zeros_like(self.paid_half)
        self.ready = torch.zeros_like(self.paid_half)
        self.load_frames = torch.zeros(n, device=env.device, dtype=torch.int32)
        self.start_stance_xy = torch.zeros((n, 2), device=env.device)
        self.has_start = torch.zeros_like(self.paid_half)
        self.max_stance_move = torch.zeros(n, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv):
        t = env.episode_length_buf * env.step_dt
        loaded = ((t >= 1.8) & (t <= 2.2)
                  & (commanded_two_foot_load(env) > .5))
        self.load_frames = torch.where(loaded, self.load_frames + 1, 0)
        self.ready |= self.load_frames >= 10
        _, _, _, exclusive = release_contact_state(env)
        command = env.command_manager.get_term("twist")
        robot = env.scene["robot"]
        sites = robot.data.site_pos_w
        left_id = robot.site_names.index("left_sole_center")
        right_id = robot.site_names.index("right_sole_center")
        stance_xy = torch.where((command.stance_side > 0)[:, None],
                                sites[:, right_id, :2], sites[:, left_id, :2])
        self.start_stance_xy = torch.where(self.has_start[:, None],
                                           self.start_stance_xy, stance_xy)
        self.has_start[:] = True
        self.max_stance_move = torch.maximum(
            self.max_stance_move,
            (stance_xy - self.start_stance_xy).norm(dim=-1),
        )
        valid = (self.ready & (t >= 2) & (t < 5) & exclusive
                 & (self.max_stance_move <= .015))
        self.current = torch.where(valid, self.current + env.step_dt,
                                   torch.zeros_like(self.current))
        self.longest = torch.maximum(self.longest, self.current)
        half = (self.current >= .5) & ~self.paid_half
        one = (self.current >= 1.0) & ~self.paid_one
        two = (self.current >= 2.0) & ~self.paid_two
        returned = ((t >= 7.5) & (self.longest >= .5)
                    & (self.max_stance_move <= .015)
                    & (standing_two_foot_return(env) > .5) & ~self.paid_return)
        self.paid_half |= half
        self.paid_one |= one
        self.paid_two |= two
        self.paid_return |= returned
        return half.float() + 2 * one.float() + 4 * two.float() + returned.float()

    def reset(self, env_ids=None):
        ids = env_ids if env_ids is not None else slice(None)
        self.current[ids] = 0
        self.longest[ids] = 0
        self.paid_half[ids] = False
        self.paid_one[ids] = False
        self.paid_two[ids] = False
        self.paid_return[ids] = False
        self.ready[ids] = False
        self.load_frames[ids] = 0
        self.start_stance_xy[ids] = 0.0
        self.has_start[ids] = False
        self.max_stance_move[ids] = 0.0


class UnsafeReleaseFrame:
    """Penalize the same fall/contact/slip routes that invalidate native scoring."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self.start_stance_xy = torch.zeros((env.num_envs, 2), device=env.device)
        self.has_start = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def __call__(self, env: ManagerBasedRlEnv):
        command = env.command_manager.get_term("twist")
        robot = env.scene["robot"]
        sites = robot.data.site_pos_w
        left_id = robot.site_names.index("left_sole_center")
        right_id = robot.site_names.index("right_sole_center")
        stance_xy = torch.where((command.stance_side > 0)[:, None],
                                sites[:, right_id, :2], sites[:, left_id, :2])
        self.start_stance_xy = torch.where(self.has_start[:, None],
                                           self.start_stance_xy, stance_xy)
        self.has_start[:] = True
        quat = robot.data.root_link_pose_w[:, 3:7]
        upright = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
        sensors = env.scene.sensors
        bad = (sensors["nonsole_ground_contact"].data.found.any(dim=-1)
               | sensors["self_body_contact"].data.found.any(dim=-1)
               | (sensors["hand_ground_contact"].data.force.norm(dim=-1).amax(dim=-1) >= 1.0))
        invalid = ((robot.data.root_link_pos_w[:, 2] < .20)
                   | (upright < math.cos(math.radians(15)))
                   | ((stance_xy - self.start_stance_xy).norm(dim=-1) > .015)
                   | bad)
        t = env.episode_length_buf * env.step_dt
        return ((t >= 1.0) & invalid).float()

    def reset(self, env_ids=None):
        ids = env_ids if env_ids is not None else slice(None)
        self.start_stance_xy[ids] = 0.0
        self.has_start[ids] = False


class SupportHoldAndReturn:
    """Episode diagnostic for the evaluator's duration and return conditions."""

    def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self.current = torch.zeros(env.num_envs, device=env.device)
        self.longest = torch.zeros_like(self.current)

    def __call__(self, env: ManagerBasedRlEnv, left_cfg: SceneEntityCfg,
                 right_cfg: SceneEntityCfg):
        clear = commanded_clear_support(env, left_cfg, right_cfg) > 0.5
        self.current = torch.where(clear, self.current + env.step_dt,
                                   torch.zeros_like(self.current))
        self.longest = torch.maximum(self.longest, self.current)
        returned = standing_two_foot_return(env) > 0.5
        return ((self.longest >= 2.0) & returned &
                (env.episode_length_buf * env.step_dt >= 8.0)).float()

    def reset(self, env_ids=None):
        ids = env_ids if env_ids is not None else slice(None)
        self.current[ids] = 0.0
        self.longest[ids] = 0.0


class LoadHoldAndReturn:
    """Partial terminal metric; independent evaluator also checks slip and motor duty."""

    def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
        del cfg
        self.current = torch.zeros(env.num_envs, device=env.device)
        self.longest = torch.zeros_like(self.current)

    def __call__(self, env: ManagerBasedRlEnv):
        valid = commanded_two_foot_load(env) > .5
        self.current = torch.where(valid, self.current + env.step_dt,
                                   torch.zeros_like(self.current))
        self.longest = torch.maximum(self.longest, self.current)
        return ((self.longest >= 3.0 - env.step_dt) &
                (standing_two_foot_return(env) > .5) &
                (env.episode_length_buf * env.step_dt >= 8.0)).float()

    def reset(self, env_ids=None):
        ids = env_ids if env_ids is not None else slice(None)
        self.current[ids] = 0.0
        self.longest[ids] = 0.0


class StageALoadTerminalReward:
    """Pay once for sustained requested load and a clean two-foot return.

    This is a training proxy. The independent evaluator also scores the full
    sole patch, which this low-cost online term does not reconstruct.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        del cfg
        n = env.num_envs
        self.window_ok = torch.ones(n, dtype=torch.bool, device=env.device)
        self.window_frames = torch.zeros(n, dtype=torch.int32, device=env.device)
        self.rated_leg_frames = torch.zeros_like(self.window_frames)
        self.rated_arm_frames = torch.zeros_like(self.window_frames)
        self.bad_contact = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.start_stance_xy = torch.zeros((n, 2), device=env.device)
        self.has_start = torch.zeros_like(self.window_ok)
        self.max_stance_move = torch.zeros(n, device=env.device)
        self.paid = torch.zeros_like(self.window_ok)

    def __call__(self, env: ManagerBasedRlEnv, rated_leg_nm: tuple[float, ...],
                 rated_arm_nm: tuple[float, ...]) -> torch.Tensor:
        t = env.episode_length_buf * env.step_dt
        window = (t >= 2.0 - 1e-6) & (t < 5.0 - 1e-6)
        valid_load = commanded_two_foot_load(env) > .5
        self.window_ok &= ~window | valid_load
        self.window_frames += window.int()
        robot = env.scene["robot"]
        leg_rating = torch.as_tensor(rated_leg_nm, device=env.device)
        arm_rating = torch.as_tensor(rated_arm_nm, device=env.device)
        self.rated_leg_frames += (
            robot.data.actuator_force[:, :10].abs() >= .995 * leg_rating
        ).any(dim=1).int()
        self.rated_arm_frames += (
            robot.data.actuator_force[:, 10:20].abs() >= .995 * arm_rating
        ).any(dim=1).int()
        sensors = env.scene.sensors
        self.bad_contact |= (
            sensors["nonsole_ground_contact"].data.found.any(dim=-1)
            | sensors["self_body_contact"].data.found.any(dim=-1)
            | (sensors["hand_ground_contact"].data.force.norm(dim=-1).amax(dim=-1) >= 1.0)
        )
        command = env.command_manager.get_term("twist")
        right = command.stance_side > 0
        sites = robot.data.site_pos_w
        left_id = robot.site_names.index("left_sole_center")
        right_id = robot.site_names.index("right_sole_center")
        stance_xy = torch.where(right[:, None], sites[:, right_id, :2], sites[:, left_id, :2])
        self.start_stance_xy = torch.where(self.has_start[:, None], self.start_stance_xy, stance_xy)
        self.has_start[:] = True
        self.max_stance_move = torch.maximum(
            self.max_stance_move, (stance_xy - self.start_stance_xy).norm(dim=-1)
        )
        at_end = (t >= 8.0 - 1e-6) & ~self.paid
        quat = robot.data.root_link_pose_w[:, 3:7]
        upright = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
        final_stand = ((standing_two_foot_return(env) > .5)
                       & (robot.data.root_link_pos_w[:, 2] >= .20)
                       & (upright >= math.cos(math.radians(15))))
        steps = env.episode_length_buf.clamp_min(1)
        success = (at_end & self.window_ok & (self.window_frames >= 150)
                   & final_stand & ~self.bad_contact
                   & (self.max_stance_move <= .015)
                   & (self.rated_leg_frames.float() / steps <= .10)
                   & (self.rated_arm_frames.float() / steps <= .10))
        self.paid |= at_end
        return success.float()

    def reset(self, env_ids=None):
        ids = env_ids if env_ids is not None else slice(None)
        self.window_ok[ids] = True
        self.window_frames[ids] = 0
        self.rated_leg_frames[ids] = 0
        self.rated_arm_frames[ids] = 0
        self.bad_contact[ids] = False
        self.start_stance_xy[ids] = 0.0
        self.has_start[ids] = False
        self.max_stance_move[ids] = 0.0
        self.paid[ids] = False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--right-pose-json", type=Path, required=True)
    p.add_argument("--left-pose-json", type=Path, required=True)
    p.add_argument("--body-xml", type=Path, required=True)
    p.add_argument("--worlds", type=int, default=512)
    p.add_argument("--updates", type=int, default=12)
    p.add_argument("--seed", type=int, default=7144)
    p.add_argument("--domain-variation", action="store_true")
    p.add_argument("--lesson", choices=("release_transition",), default="release_transition")
    p.add_argument("--run-name", required=True)
    p.add_argument("--preflight-only", action="store_true",
                   help="Build and step one local CPU world; no PPO update")
    p.add_argument("--reference-residual", action="store_true",
                   help="Required seven-channel residual over the source load reference")
    p.add_argument("--preflight-full-zero", action="store_true",
                   help="Run a full local 400-step direct-zero control-clock check")
    p.add_argument("--preflight-ppo-act", action="store_true",
                   help="Sample the actual untrained RSL-RL PPO actor once; no update or score")
    p.add_argument("--preflight-ppo-episode", action="store_true",
                   help="Roll the initialized PPO actor for one unscored 8 s episode; no update")
    p.add_argument("--preflight-reward-controller", choices=("reference", "ppo"),
                   help="Record full local reward and physical channels for one 8 s episode")
    p.add_argument("--preflight-pulse-joint", choices=REFERENCE_RESIDUAL_JOINTS,
                   help="Apply one signed raw residual pulse during a reference reward episode")
    p.add_argument("--preflight-pulse-raw", type=float, default=0.0)
    p.add_argument("--residual-init-std", type=float, default=0.05,
                   help="Raw seven-channel Gaussian std; bracket exploration before a fit")
    p.add_argument("--preflight-side", choices=("left", "right"),
                   help="Select an actual reset-event side for a local preflight")
    p.add_argument("--preflight-out", type=Path,
                   help="Save the local preflight record as JSON")
    p.add_argument("--eval-checkpoint", type=Path,
                   help="Reload a saved V19 actor for one source-matched native visit")
    p.add_argument("--eval-device", default="cpu")
    p.add_argument("--eval-fixed-lag-steps", type=int)
    p.add_argument("--eval-video", type=Path)
    args = p.parse_args()
    if not args.reference_residual:
        p.error("V19 requires --reference-residual")
    if args.preflight_only and args.domain_variation and args.eval_device == "cpu":
        p.error("Local CPU preflight requires domain variation off")
    if args.domain_variation and not args.preflight_only:
        p.error("V19 pilot is nominal only; per-world effort-limit scoring must precede varied fit")
    if args.preflight_full_zero and not (args.reference_residual and args.preflight_only):
        p.error("--preflight-full-zero needs --reference-residual --preflight-only")
    if args.preflight_ppo_act and not (args.reference_residual and args.preflight_only):
        p.error("--preflight-ppo-act needs --reference-residual --preflight-only")
    if args.preflight_ppo_episode and not (args.reference_residual and args.preflight_only):
        p.error("--preflight-ppo-episode needs --reference-residual --preflight-only")
    if args.preflight_ppo_episode and args.preflight_ppo_act:
        p.error("Select one PPO preflight mode")
    if args.preflight_reward_controller and not (args.reference_residual and args.preflight_only):
        p.error("--preflight-reward-controller needs --reference-residual --preflight-only")
    if args.preflight_reward_controller and (args.preflight_ppo_episode or args.preflight_ppo_act):
        p.error("Select one full-episode preflight mode")
    if args.preflight_pulse_joint and args.preflight_reward_controller != "reference":
        p.error("A pulse needs --preflight-reward-controller reference")
    if args.preflight_pulse_joint and not 0 < abs(args.preflight_pulse_raw) <= 1:
        p.error("A pulse raw value must be in [-1,1] excluding zero")
    if not 0.02 <= args.residual_init_std <= 0.2:
        p.error("--residual-init-std must be in [0.02, 0.2]")
    if args.preflight_side and not (args.reference_residual and args.preflight_only):
        p.error("--preflight-side needs --reference-residual --preflight-only")
    if args.eval_checkpoint and (not args.preflight_only or args.preflight_reward_controller != "ppo"):
        p.error("Checkpoint replay requires --preflight-only --preflight-reward-controller ppo")
    if args.eval_video and not args.preflight_reward_controller:
        p.error("Video requires a full reward/controller episode")
    if args.eval_fixed_lag_steps is not None and (not args.domain_variation or
            not 0 <= args.eval_fixed_lag_steps <= 4):
        p.error("Fixed motor lag requires domain variation and 0..4 physics steps")
    body = args.body_xml.resolve()
    model = require_direct_frame(body)
    names = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    specs = {side: json.loads(path.read_text()) for side, path in
             (("right", args.right_pose_json), ("left", args.left_pose_json))}
    pose = {side: [specs[side]["joints"][name] for name in names]
            for side in ("right", "left")}
    z = specs["right"]["root_xyz_roll_pitch"][2]
    roll = specs["right"]["root_xyz_roll_pitch"][3]
    pitch = specs["right"]["root_xyz_roll_pitch"][4]
    cfg = load_env_cfg(ape.ELBOW_FLAMINGO_CYCLE_ARMS_TASK_ID, play=False)
    configure_frame50_motors(cfg, body, rail_v=12.0, delay_s=0.0)
    give_soles_contact_priority(cfg)
    if args.reference_residual:
        cfg.scene.entities["robot"].collisions[0].friction = {
            r"^(left|right)_sole$": (1.0,)
        }
    if args.domain_variation:
        robot = cfg.scene.entities["robot"]
        robot.articulation.actuators = tuple(
            replace(act, delay_min_lag=0, delay_max_lag=4)
            for act in robot.articulation.actuators
        )
        bodies = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                       for i in range(1, model.nbody))
        cfg.events["cove_link_density"] = EventTermCfg(
            mode="reset", func=dr.pseudo_inertia,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=bodies),
                    "alpha_range": (math.log(0.9) / 2, math.log(1.1) / 2)},
        )
        cfg.events["cove_sole_friction"] = EventTermCfg(
            mode="reset", func=dr.geom_friction,
            params={"asset_cfg": SceneEntityCfg(
                "robot", geom_names=("left_sole", "right_sole")),
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
    if args.eval_fixed_lag_steps is not None:
        robot = cfg.scene.entities["robot"]
        robot.articulation.actuators = tuple(
            replace(act, delay_min_lag=args.eval_fixed_lag_steps,
                    delay_max_lag=args.eval_fixed_lag_steps)
            for act in robot.articulation.actuators
        )
    cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
    clip_frame50_leg_targets(cfg, body, margin_rad=0.02)
    if args.reference_residual:
        prior = cfg.actions["joint_pos"]
        cfg.actions["joint_pos"] = ReleaseReferenceActionCfg(
            entity_name=prior.entity_name,
            actuator_names=prior.actuator_names,
            preserve_order=prior.preserve_order,
            scale=0.05,
            offset=0.0,
            clip=prior.clip,
            shift_roll_rad=0.35,
            reference_native_scale=prior.scale,
            reference_native_offset=prior.offset,
            reference_use_default_offset=prior.use_default_offset,
            residual_joint_names=REFERENCE_RESIDUAL_JOINTS,
            raw_residual_clip=1.0,
        )
        for group in ("actor", "critic"):
            terms = cfg.observations[group].terms
            terms["support_skill_time"] = ObservationTermCfg(func=support_skill_time)
            terms["support_latched_side"] = ObservationTermCfg(func=support_latched_side)
            terms["support_reference_error"] = ObservationTermCfg(
                func=support_reference_error,
                noise=deepcopy(terms["joint_pos"].noise) if group == "actor" else None,
            )
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, model.qpos0[2])
    cfg.seed = args.seed
    cfg.rewards["pose_track"].params.update({"pose_right": pose["right"],
                                              "pose_left": pose["left"]})
    cfg.rewards["gravity_track"].params.update({
        "gravity_right": (math.sin(pitch), -math.sin(roll) * math.cos(pitch),
                          -math.cos(roll) * math.cos(pitch)),
        "gravity_left": (math.sin(pitch), math.sin(roll) * math.cos(pitch),
                         -math.cos(roll) * math.cos(pitch)),
    })
    cfg.rewards["joint_torques_l2"].weight = -0.05
    # The source one-foot pose and COM targets reward an unreachable static
    # bridge. V19 instead pays attained load and actual release contact.
    for name in ("com_target", "stance_foot_grounded", "swing_foot_contact",
                 "swing_foot_clear", "pose_track", "gravity_track", "stillness",
                 "stance_side_tilt", "commanded_support"):
        cfg.rewards[name].weight = 0.0
    cfg.rewards["foot_slip"].weight = -0.5
    cfg.rewards["release_entry_load"] = RewardTermCfg(
        func=release_entry_load_reward, weight=1.0,
    )
    cfg.rewards["attained_load_release_progress"] = RewardTermCfg(
        func=AttainedLoadReleaseProgress, weight=2.0,
    )
    cfg.rewards["clear_exclusive_support"] = RewardTermCfg(
        func=AttainedLoadExclusiveSupport, weight=4.0,
    )
    cfg.rewards["release_duration_return"] = RewardTermCfg(
        func=ReleaseDurationAndReturn, weight=50.0,
    )
    cfg.rewards["unsafe_release_frame"] = RewardTermCfg(
        func=UnsafeReleaseFrame, weight=-15.0,
    )
    cfg.episode_length_s = 8.0
    if args.reference_residual:
        cfg.rewards["action_over_limit"].func = support_action_over_limit
        cfg.rewards["action_over_limit"].params = {"overshoot": 0.1}
    cfg.events["set_flamingo_cycle_state"].params.update({
        "pose_right": pose["right"], "pose_left": pose["left"],
        "base_roll": roll, "base_pitch": pitch,
        "z_min": z, "z_max": z + 0.002,
        "standing_z_min": float(model.qpos0[2]),
        "standing_z_max": float(model.qpos0[2] + (0.0 if args.reference_residual else 0.002)),
        "tilt_noise": math.radians(0.25 if args.reference_residual else 1.0),
        "joint_noise_std": 0.0025 if args.reference_residual else 0.005,
        "in_pose_prob": 0.0,
    })
    # Upstream random 2.5–5 s resampling can omit the return or hold phase.
    # Every episode now exposes the same 1–5 s request and 5–8 s return as
    # the independent evaluator, with one sampled stance side per episode.
    cfg.commands["twist"] = CoveSupportCycleCommandCfg(
        **{**vars(cfg.commands["twist"]),
           "resampling_time_range": (100.0, 100.0)},
    )
    # The saved 300-update run never reached the source 700-update transition.
    # Expose upright entries throughout and the final zero in-pose mix for
    # half of the planned per-world steps, independent of GPU world count.
    total_per_world = args.updates * 24
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
        {"step": 0, "weight": -0.1},
        {"step": int(total_per_world * 0.5), "weight": -0.5},
    ]
    cfg.curriculum["flamingo_com_std"].params["param_stages"] = [
        {"step": 0, "params": {"std": 0.03}},
        {"step": int(total_per_world * 0.5), "params": {"std": 0.02}},
    ]
    cfg.curriculum["in_pose_prob"].params["param_stages"] = [
        {"step": 0, "params": {"in_pose_prob": 0.0}},
    ]
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ape.ContactSensorCfg(
            name="thigh_torso_contact",
            primary=ape.ContactMatch(
                mode="geom",
                pattern=r"^(left_thigh_frame_budget|right_thigh_frame_budget)$",
                entity="robot"),
            secondary=ape.ContactMatch(
                mode="geom", pattern="torso_frame_compute_budget", entity="robot"),
            fields=("found", "force"), reduce="maxforce", num_slots=1,
        ),
        ape.ContactSensorCfg(
            name="self_body_contact",
            primary=ape.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=ape.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"), reduce="none", num_slots=1,
        ),
        ape.ContactSensorCfg(
            name="nonsole_ground_contact",
            primary=ape.ContactMatch(
                mode="geom", pattern=r"^(?!(?:left_sole|right_sole)$).*", entity="robot"),
            secondary=ape.ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="netforce", num_slots=1,
        ),
    )
    cfg.terminations["thigh_torso_contact"] = TerminationTermCfg(
        func=ape.mdp.illegal_contact,
        params={"sensor_name": "thigh_torso_contact", "force_threshold": 0.1},
    )
    rated = tuple(float(act.effort_limit)
                  for act in cfg.scene.entities["robot"].articulation.actuators[:10])
    cfg.rewards["rated_leg_frame_penalty"] = RewardTermCfg(
        func=rated_leg_frame, weight=-1.0, params={"rated_nm": rated},
    )
    cfg.metrics["commanded_clear_support_fraction"] = MetricsTermCfg(
        func=commanded_clear_support,
        params={"left_cfg": SceneEntityCfg("robot", site_names=("left_sole_center",)),
                "right_cfg": SceneEntityCfg("robot", site_names=("right_sole_center",)),
                "clearance_m": 0.025},
    )
    cfg.metrics["terminal_two_foot_stand"] = MetricsTermCfg(
        func=standing_two_foot_return, reduce="last",
    )
    cfg.metrics["commanded_two_foot_load_fraction"] = MetricsTermCfg(
        func=commanded_two_foot_load,
    )
    cfg.metrics["terminal_2s_hold_and_return_partial"] = MetricsTermCfg(
        func=SupportHoldAndReturn, reduce="last",
        params={"left_cfg": SceneEntityCfg("robot", site_names=("left_sole_center",)),
                "right_cfg": SceneEntityCfg("robot", site_names=("right_sole_center",))},
    )
    cfg.metrics["rated_leg_frame_fraction"] = MetricsTermCfg(
        func=rated_leg_frame, params={"rated_nm": rated},
    )
    agent = deepcopy(load_rl_cfg(ape.ELBOW_FLAMINGO_CYCLE_ARMS_TASK_ID))
    if agent.num_steps_per_env != 24:
        raise ValueError("Curriculum schedule assumes 24 rollout steps per world")
    agent.experiment_name = "ape_cove50_release_residual_v19"
    agent.actor.distribution_cfg["init_std"] = (
        args.residual_init_std if args.reference_residual else 0.5
    )
    task_id = "Mjlab-Cove50-Release-Residual-V19"
    if args.preflight_only:
        cfg.scene.num_envs = 1
        cfg.auto_reset = False
        if args.eval_video:
            cfg.viewer.origin_type = cfg.viewer.OriginType.WORLD
            cfg.viewer.lookat = (0.0, 0.0, 0.24)
            cfg.viewer.distance = 0.75
            cfg.viewer.azimuth = 45.0
            cfg.viewer.elevation = -14.0
            cfg.viewer.width = 640
            cfg.viewer.height = 480
        env = ManagerBasedRlEnv(
            cfg=cfg, device=args.eval_device,
            render_mode="rgb_array" if args.eval_video else None,
        )
        if args.eval_video:
            reveal_frame50_proxy_geometry_for_video(env.sim.mj_model)
        observation, _ = env.reset()
        if args.preflight_side:
            requested = 1.0 if args.preflight_side == "right" else -1.0
            for _ in range(32):
                if float(env.command_manager.get_term("twist").stance_side[0]) == requested:
                    break
                observation, _ = env.reset()
            else:
                raise RuntimeError("Requested side did not occur in 32 reset-event draws")
        first = {
            "schema": "robot-learning-lab.cove50-support-env-entry.v1",
            "body": str(body),
            "body_sha256": hashlib.sha256(body.read_bytes()).hexdigest(),
            "seed": args.seed,
            "lesson": args.lesson,
            "actor_shape": list(observation["actor"].shape),
            "critic_shape": list(observation["critic"].shape),
            "action_shape": list(env.action_space.shape),
            "runtime_integrator": int(env.sim.mj_model.opt.integrator),
            "configured_integrator": cfg.sim.mujoco.integrator,
            "actor_first_12": observation["actor"][0, :12].tolist(),
            "feet_found": env.scene.sensors["feet_ground_contact"].data.found[0].tolist(),
            "metrics": list(cfg.metrics),
            "command_schedule_s": {"stand": [0, 1], "support": [1, 5],
                                   "return_and_settle": [5, 8]},
            "curriculum": {name: term.params for name, term in cfg.curriculum.items()},
            "claim_limit": "One local CPU setup step; not CUDA score or learned skill.",
        }
        action_term = env.action_manager.get_term("joint_pos")
        first["reset_command"] = {
            "flag": float(env.command_manager.get_term("twist").flag[0]),
            "observed_side": float(env.command_manager.get_term("twist").side[0]),
            "latched_side": float(env.command_manager.get_term("twist").stance_side[0]),
            "alpha": float(env.command_manager.get_term("twist").alpha[0]),
            "skill_time_s": float(support_skill_time(env)[0, 0]),
        }
        if args.preflight_full_zero:
            from evaluate_support_reference_parity import authored_teacher_target

            term = env.command_manager.get_term("twist")
            side = "right" if float(term.stance_side[0]) > 0 else "left"
            zero_targets, zero_qpos, zero_qvel = [], [], []
            boundaries = []
            max_target_gap = 0.0
            early_termination_step = None
            for step in range(400):
                t = step * env.step_dt
                expected = authored_teacher_target(
                    t, side, "load_transfer", list(action_term.target_names),
                    args.eval_device, 0.35, 0.2
                )
                active = action_term.reference_native_scale.abs() > 1e-8
                native = torch.where(
                    active,
                    (expected - action_term.reference_native_offset) /
                    torch.where(active, action_term.reference_native_scale, 1.0),
                    0.0,
                )
                decoded = native * action_term.reference_native_scale + action_term.reference_native_offset
                if step in (0, 50, 100, 250, 350, 399):
                    boundaries.append({
                        "step": step, "skill_time_s": t,
                        "flag": float(term.flag[0]),
                        "observed_side": float(term.side[0]),
                        "latched_side": float(term.stance_side[0]),
                        "alpha": float(term.alpha[0]),
                    })
                _, _, terminated, time_out, _ = env.step(
                    torch.zeros(env.action_space.shape, device=env.device))
                gap = (decoded - action_term.last_absolute_target).abs().max().item()
                max_target_gap = max(max_target_gap, gap)
                zero_targets.append(action_term.last_absolute_target[0].cpu().numpy().copy())
                zero_qpos.append(env.sim.data.qpos[0].cpu().numpy().copy())
                zero_qvel.append(env.sim.data.qvel[0].cpu().numpy().copy())
                if bool(terminated[0]) or (bool(time_out[0]) and step < 399):
                    early_termination_step = step
                    break
            first["full_zero_preflight"] = {
                "side": side, "steps": len(zero_targets),
                "expected_steps": 400, "early_termination_step": early_termination_step,
                "max_decoded_teacher_target_gap_rad": max_target_gap,
                "control_boundaries": boundaries,
            }
            if args.preflight_out:
                args.preflight_out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.preflight_out.with_suffix(".npz"),
                                    joint_target=np.asarray(zero_targets),
                                    qpos=np.asarray(zero_qpos), qvel=np.asarray(zero_qvel),
                                    joint_names=np.asarray(action_term.target_names))
            env.reset()
        env.step(torch.zeros(env.action_space.shape, device=env.device))
        if args.reference_residual:
            action_term.process_actions(torch.full(env.action_space.shape, 2.0,
                                                   device=env.device))
            first["clip_preflight"] = {
                "raw": action_term.raw_action[0].tolist(),
                "bounded": action_term.last_bounded_action[0].tolist(),
                "target_correction_rad": action_term._processed_actions[0].tolist(),
            }
            sampled = torch.randn(env.action_space.shape, device=env.device) * agent.actor.distribution_cfg["init_std"]
            env.step(sampled)
            first["sampled_action_preflight"] = {
                "raw": sampled[0].tolist(),
                "bounded": action_term.last_bounded_action[0].tolist(),
                "joint_names": list(action_term.residual_joint_names),
                "target_correction_rad": action_term._processed_actions[0].tolist(),
                "absolute_target": action_term.last_absolute_target[0].tolist(),
                "native_reference_action": action_term.last_reference_native_action[0].tolist(),
                "init_std_raw": agent.actor.distribution_cfg["init_std"],
                "raw_clip": action_term.cfg.raw_residual_clip,
                "action_manager_current_raw": env.action_manager.action[0].tolist(),
                "action_manager_previous_raw": env.action_manager.prev_action[0].tolist(),
            }
        term = env.command_manager.get_term("twist")
        old_steps = env.episode_length_buf.clone()
        first["command_schedule_preflight"] = []
        for seconds in (0.0, 0.98, 1.0, 4.98, 5.0, 8.0):
            env.episode_length_buf[:] = round(seconds / env.step_dt)
            term._update_command()
            first["command_schedule_preflight"].append({
                "t_s": seconds,
                "flag": float(term.flag[0]),
                "observed_side": float(term.side[0]),
                "latched_side": float(term.stance_side[0]),
            })
        env.episode_length_buf[:] = old_steps
        sole = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM,
                                 "robot/left_sole")
        first["sole_friction_reset_draws"] = []
        first["stance_side_reset_draws"] = []
        for _ in range(16):
            env.reset()
            first["sole_friction_reset_draws"].append(
                float(env.sim.model.geom_friction[0, sole, 0])
            )
            first["stance_side_reset_draws"].append(float(term.stance_side[0]))
        if args.preflight_ppo_act or args.preflight_ppo_episode:
            # This deliberately executes PPO.act rather than synthesizing
            # Gaussian noise. It checks the initialized actor and 4D action
            # contract without taking an optimizer step or scoring a skill.
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            if args.preflight_side:
                requested = 1.0 if args.preflight_side == "right" else -1.0
                for _ in range(32):
                    if float(env.command_manager.get_term("twist").stance_side[0]) == requested:
                        break
                    wrapped.reset()
                else:
                    raise RuntimeError("PPO preflight side did not occur")
            torch.manual_seed(args.seed)
            runner_cfg = asdict(agent)
            runner_cfg["multi_gpu"] = None
            for name in ("actor", "critic"):
                for optional in ("cnn_cfg", "distribution_cfg"):
                    if runner_cfg[name].get(optional) is None:
                        runner_cfg[name].pop(optional, None)
                if runner_cfg[name].get("rnn_type") is None:
                    for optional in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                        runner_cfg[name].pop(optional, None)
            obs = wrapped.get_observations()
            alg = PPO.construct_algorithm(obs, wrapped, runner_cfg, args.eval_device)
            if args.preflight_ppo_act:
                sampled = alg.act(obs).detach()
                distribution = [value[0].detach().tolist()
                                for value in alg.transition.distribution_params]
                raw_reference = action_term.reference_target()[0].detach().clone()
                wrapped.step(sampled)
                first["ppo_act_preflight"] = {
                    "source": "rsl_rl.algorithms.PPO.act; initialized actor, no checkpoint or update",
                    "actor_observation_shape": list(obs["actor"].shape),
                    "action_shape": list(sampled.shape),
                    "sampled_raw_action": sampled[0].tolist(),
                    "distribution_params": distribution,
                    "action_manager_raw": env.action_manager.action[0].tolist(),
                    "bounded_raw_action": action_term.last_bounded_action[0].tolist(),
                    "native_target_correction_rad": (
                        action_term.last_absolute_target[0] - raw_reference
                    ).tolist(),
                    "latched_side": float(env.command_manager.get_term("twist").stance_side[0]),
                    "control_steps": 1,
                    "optimizer_updates": 0,
                }
            else:
                # This is a cheap exploration-tolerance check, not a native
                # skill score. Keep the same actor and exact action manager
                # for all 400 control steps, without optimizer updates.
                rows = []
                early_termination_step = None
                for step in range(400):
                    sampled = alg.act(obs).detach()
                    _, reward, terminated, time_out, _ = env.step(sampled)
                    obs = wrapped.get_observations()
                    rows.append({
                        "step": step + 1,
                        "time_s": round((step + 1) * env.step_dt, 3),
                        "reward": float(reward[0]),
                        "sampled_action": sampled[0].tolist(),
                        "load_valid": bool(commanded_two_foot_load(env)[0] > .5),
                        "rated_leg": bool(rated_leg_frame(env, rated)[0] > .5),
                        "returned_two_foot": bool(standing_two_foot_return(env)[0] > .5),
                        "terminated": bool(terminated[0]),
                        "time_out": bool(time_out[0]),
                    })
                    if bool(terminated[0]) or (bool(time_out[0]) and step < 399):
                        early_termination_step = step + 1
                        break
                hold = [row for row in rows if 2.0 <= row["time_s"] <= 5.0]
                first["ppo_episode_preflight"] = {
                    "source": "rsl_rl.algorithms.PPO.act; initialized actor, no checkpoint or update",
                    "side": "right" if float(env.command_manager.get_term("twist").stance_side[0]) > 0 else "left",
                    "residual_init_std": args.residual_init_std,
                    "steps": len(rows),
                    "early_termination_step": early_termination_step,
                    "mean_abs_raw_action": float(np.mean([np.mean(np.abs(row["sampled_action"])) for row in rows])),
                    "max_abs_raw_action": float(np.max([np.max(np.abs(row["sampled_action"])) for row in rows])),
                    "load_valid_frames_2_to_5_s": sum(row["load_valid"] for row in hold),
                    "observed_frames_2_to_5_s": len(hold),
                    "rated_leg_fraction": sum(row["rated_leg"] for row in rows) / len(rows),
                    "last_returned_two_foot": rows[-1]["returned_two_foot"],
                    "reward_sum": sum(row["reward"] for row in rows),
                    "rows": rows,
                    "optimizer_updates": 0,
                    "claim_limit": "Local CPU sampled-actor survival and load diagnostic only; no full physical gate or CUDA score.",
                }
        if args.preflight_reward_controller:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            if args.preflight_side:
                requested = 1.0 if args.preflight_side == "right" else -1.0
                for _ in range(32):
                    if float(env.command_manager.get_term("twist").stance_side[0]) == requested:
                        break
                    wrapped.reset()
                else:
                    raise RuntimeError("Reward preflight side did not occur")
            obs = wrapped.get_observations()
            initial_qpos = env.sim.data.qpos[0].cpu().numpy().copy()
            initial_qvel = env.sim.data.qvel[0].cpu().numpy().copy()
            state_hash = hashlib.sha256(initial_qpos.tobytes() + initial_qvel.tobytes()).hexdigest()
            domain_hash = hashlib.sha256()
            domain_fields = []
            for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat",
                          "geom_friction", "actuator_gainprm", "actuator_biasprm",
                          "actuator_forcerange"):
                value = getattr(env.sim.model, field, None)
                if value is not None:
                    item = (value[0] if isinstance(value, (TorchArray, torch.Tensor,
                                                         tuple, list)) else value)
                    array = (item.detach().cpu().numpy() if isinstance(item, torch.Tensor)
                             else np.asarray(item))
                    domain_hash.update(field.encode() + array.tobytes())
                    domain_fields.append(field)
            alg = None
            policy = None
            checkpoint_iteration = None
            if args.preflight_reward_controller == "ppo":
                torch.manual_seed(args.seed)
                runner_cfg = asdict(agent)
                runner_cfg["multi_gpu"] = None
                for name in ("actor", "critic"):
                    for optional in ("cnn_cfg", "distribution_cfg"):
                        if runner_cfg[name].get(optional) is None:
                            runner_cfg[name].pop(optional, None)
                    if runner_cfg[name].get("rnn_type") is None:
                        for optional in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                            runner_cfg[name].pop(optional, None)
                if args.eval_checkpoint:
                    runner = ape.VelocityOnPolicyRunner(wrapped, runner_cfg, device=args.eval_device)
                    runner.load(str(args.eval_checkpoint),
                                load_cfg={"actor": True, "iteration": True},
                                strict=True, map_location=args.eval_device)
                    checkpoint_iteration = int(runner.current_learning_iteration)
                    policy = runner.get_inference_policy(device=args.eval_device)
                else:
                    alg = PPO.construct_algorithm(obs, wrapped, runner_cfg, args.eval_device)
            robot = env.scene["robot"]
            sensors = env.scene.sensors
            lag_buffers = []
            seen_lag_buffers = set()
            for actuator in robot.actuators:
                buffer = getattr(actuator, "_delay_buffer", None)
                if buffer is not None and id(buffer) not in seen_lag_buffers:
                    seen_lag_buffers.add(id(buffer))
                    lag_buffers.append(buffer)
            lag_hash = hashlib.sha256()
            lag_trace = []
            side = "right" if float(env.command_manager.get_term("twist").stance_side[0]) > 0 else "left"
            stance_slot = 1 if side == "right" else 0
            arm_rated = torch.as_tensor(
                [act.effort_limit for act in cfg.scene.entities["robot"].articulation.actuators[10:20]],
                device=env.device,
            )
            rows, states, velocities = [], [], []
            writer = None
            if args.eval_video:
                args.eval_video.parent.mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(str(args.eval_video), fps=25, macro_block_size=1)
            for step in range(400):
                with torch.inference_mode():
                    action = (policy(obs).detach() if policy is not None else
                              alg.act(obs).detach() if alg is not None else
                              torch.zeros(env.action_space.shape, device=env.device))
                    if args.preflight_pulse_joint and 2.50 <= step * env.step_dt < 2.70:
                        channel = action_term.residual_joint_names.index(args.preflight_pulse_joint)
                        action[0, channel] = args.preflight_pulse_raw
                    _, reward, terminated, time_out, _ = env.step(action)
                lags = [int(buffer.current_lags[0].cpu()) for buffer in lag_buffers]
                if args.eval_fixed_lag_steps is not None and any(
                    lag != args.eval_fixed_lag_steps for lag in lags
                ):
                    raise RuntimeError("Observed motor lag differs from frozen fixed lag")
                lag_trace.append(lags)
                lag_hash.update(np.asarray(lags, dtype=np.int16).tobytes())
                obs = wrapped.get_observations()
                feet = sensors["feet_ground_contact"].data
                force = feet.force[0].norm(dim=-1)
                quat = robot.data.root_link_pose_w[0, 3:7]
                upright = 1.0 - 2.0 * (quat[1].square() + quat[2].square())
                reward_terms = {name: value[0] * env.step_dt
                                for name, value in env.reward_manager.get_active_iterable_terms(0)}
                row = {
                    "step": step + 1,
                    "time_s": round((step + 1) * env.step_dt, 3),
                    "raw_action": action[0].tolist(),
                    "reward": float(reward[0]),
                    "reward_terms": reward_terms,
                    "command_flag": float(env.command_manager.get_term("twist").flag[0]),
                    "feet_grounded": feet.found[0].bool().tolist(),
                    "stance_load_fraction": float(force[stance_slot] / force.sum().clamp_min(1.0)),
                    "pelvis_xy_m": robot.data.root_link_pos_w[0, :2].tolist(),
                    "pelvis_z_m": float(robot.data.root_link_pos_w[0, 2]),
                    "tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, float(upright))))),
                    "rated_leg": bool(rated_leg_frame(env, rated)[0] > .5),
                    "rated_arm": bool((robot.data.actuator_force[0, 10:20].abs() >= .995 * arm_rated).any()),
                    "nonsole_contact": bool(sensors["nonsole_ground_contact"].data.found[0].any()),
                    "self_contact": bool(sensors["self_body_contact"].data.found[0].any()),
                    "hand_force_n": float(sensors["hand_ground_contact"].data.force[0].norm(dim=-1).max()),
                    "returned_two_foot": bool(standing_two_foot_return(env)[0] > .5),
                    "terminated": bool(terminated[0]),
                    "time_out": bool(time_out[0]),
                }
                if abs(sum(reward_terms.values()) - row["reward"]) > 2e-5:
                    raise RuntimeError("Per-term reward and returned reward disagree")
                rows.append(row)
                states.append(env.sim.data.qpos[0].cpu().numpy().copy())
                velocities.append(env.sim.data.qvel[0].cpu().numpy().copy())
                if writer is not None and step % 2 == 0:
                    writer.append_data(env.render())
                if row["terminated"] or (row["time_out"] and step < 399):
                    break
            if writer is not None:
                writer.close()
            if args.preflight_out:
                args.preflight_out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    args.preflight_out.with_suffix(".npz"),
                    initial_qpos=initial_qpos, initial_qvel=initial_qvel,
                    qpos=np.asarray(states), qvel=np.asarray(velocities),
                )
            first["reward_episode_preflight"] = {
                "controller": args.preflight_reward_controller,
                "side": side,
                "seed": args.seed,
                "residual_init_std": args.residual_init_std if alg is not None else None,
                "pulse_joint": args.preflight_pulse_joint,
                "pulse_raw": args.preflight_pulse_raw if args.preflight_pulse_joint else None,
                "pulse_window_s": [2.50, 2.70] if args.preflight_pulse_joint else None,
                "initial_state_sha256": state_hash,
                "initial_domain_sha256": domain_hash.hexdigest(),
                "initial_domain_fields": domain_fields,
                "eval_device": str(env.device),
                "eval_cuda_device_name": (torch.cuda.get_device_name(env.device)
                                          if str(env.device).startswith("cuda") else None),
                "eval_fixed_lag_steps": args.eval_fixed_lag_steps,
                "lag_buffer_count": len(lag_buffers),
                "lag_sequence_sha256": lag_hash.hexdigest(),
                "lag_trace": lag_trace,
                "checkpoint_sha256": (hashlib.sha256(args.eval_checkpoint.read_bytes()).hexdigest()
                                      if args.eval_checkpoint else None),
                "checkpoint_iteration": checkpoint_iteration,
                "steps": len(rows),
                "reward_sum": sum(row["reward"] for row in rows),
                "reward_terms_sum": {name: sum(row["reward_terms"][name] for row in rows)
                                     for name in rows[0]["reward_terms"]},
                "rows": rows,
                "optimizer_updates": 0,
                "claim_limit": ("One native development replay; score full physical outcome from source trace."
                                if str(env.device).startswith("cuda") else
                                "Local CPU reward and physical prefit audit; no CUDA score."),
            }
        env.close()
        if args.preflight_out:
            args.preflight_out.parent.mkdir(parents=True, exist_ok=True)
            args.preflight_out.write_text(json.dumps(first, indent=2) + "\n")
        print(json.dumps(first, indent=2))
        return
    ape.register_mjlab_task(task_id=task_id, env_cfg=cfg, play_env_cfg=cfg,
                            rl_cfg=agent, runner_cls=ape.VelocityOnPolicyRunner)
    train = TrainConfig.from_task(task_id)
    train.env.scene.num_envs = args.worlds
    train.env.sim.mujoco.timestep = 0.002
    train.env.decimation = 10
    train.agent.max_iterations = args.updates
    train.agent.save_interval = 4
    train.agent.run_name = args.run_name
    train.agent.logger = "tensorboard"
    train.agent.resume = False
    launch_training(task_id, train)


if __name__ == "__main__":
    main()
