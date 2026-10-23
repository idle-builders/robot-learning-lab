"""Retarget retained stand, squat, and arm lessons to the 21-axis Cove body.

The source 22-axis PPO weights first pass through map_ppo_22_to_21.py.  Pose
and height goals here come from fresh Cove kinematics, not the 35 cm targets.
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
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from ape_walk import task as ape
from ape_walk.physics.frame50 import (configure_frame50_motors,
                                      clip_frame50_leg_targets,
                                      clip_frame50_shoulder_targets,
                                      give_soles_contact_priority,
                                      require_direct_frame)


TASKS = {
    "stand": ape.ELBOW_STAND_TASK_ID,
    "squat": ape.ELBOW_SQUAT_HEADING_OBS_TASK_ID,
    "arms": ape.ELBOW_ARM_TASK_ID,
    "independent-arms": ape.ELBOW_ARM_INDEPENDENT_TASK_ID,
}
STAND_Z = 0.2305
SQUAT_Z = 0.1794
# Flat soles at this pelvis height, with ankle pitch inside its +/-0.349 rad limit.
SQUAT_LEGS = {2: -1.16, 3: 1.46, 4: -0.30,
              7: -1.16, 8: 1.46, 9: -0.30}
SQUAT_COUNTERBALANCE_ARMS = {13: -1.0, 18: -1.0}
SQUAT_ACTION_SCALE = {2: 1.5, 3: 1.6, 4: 1.0,
                      7: 1.5, 8: 1.6, 9: 1.0,
                      13: 1.2, 18: 1.2}


class CoveSitStandBlendCommand(ape.ApeSitStandHeadingCommand):
    """Expose the actual slew target to the actor in the existing x command slot."""

    @property
    def command(self):
        observed = self.vel_command_b.clone()
        observed[:, 0] = self.alpha
        return observed


@dataclass(kw_only=True)
class CoveSitStandBlendCommandCfg(ape.ApeSitStandHeadingCommandCfg):
    class_type: type = CoveSitStandBlendCommand

    def build(self, env):
        return CoveSitStandBlendCommand(self, env)


def expose_squat_blend(cfg) -> None:
    current = cfg.commands["twist"]
    cfg.commands["twist"] = CoveSitStandBlendCommandCfg(
        **{**vars(current), "class_type": CoveSitStandBlendCommand}
    )


def zero_squat_action_history(env):
    """Preserve the 21-channel actor contract without unstable action feedback."""
    return torch.zeros_like(env.action_manager.action)


def mask_squat_action_history(cfg) -> None:
    cfg.observations["actor"].terms["actions"].func = zero_squat_action_history


def seated_rise_phase(env) -> torch.Tensor:
    """Expose the command slew without replacing the intended stand command."""
    return env.command_manager.get_term("twist").alpha[:, None]


def nonsole_body_on_terrain(env) -> torch.Tensor:
    """A fallen torso/limb touching terrain is a failure even without force."""
    return env.scene["nonsole_body_ground_contact"].data.found.any(dim=-1)


def hand_on_terrain(env) -> torch.Tensor:
    """Match the seated-rise score's zero-tolerance hand contact rule."""
    return env.scene["hand_ground_contact"].data.found.any(dim=-1)


def require_seated_contact_capacity(env) -> None:
    """Reject truncated MuJoCo-Warp contacts before using them for clearance."""
    capacity = env.sim.data.contact.dist.shape[0]
    active = int(env.sim.data.nacon[0])
    broadphase = int(env.sim.data.ncollision[0])
    if not hasattr(env, "_seated_contact_peak"):
        env._seated_contact_peak = [0, 0, capacity]
    env._seated_contact_peak[0] = max(env._seated_contact_peak[0], active)
    env._seated_contact_peak[1] = max(env._seated_contact_peak[1], broadphase)
    if active > capacity or broadphase > capacity:
        raise ValueError(
            f"Seated reset contact capacity overflow: active={active}, "
            f"broadphase={broadphase}, capacity={capacity}"
        )


def reset_seated_rise_clearance(env, env_ids, **kwargs) -> None:
    """Reject colliding seated draws and lift accepted poses clear of the floor.

    The upstream sampler adds joint noise *in place*. Restore the entry state
    and redraw base joint offsets before each attempt, making full joint poses
    independent rather than accumulating noise across retries.
    """
    max_attempts = 64
    initial_qpos = env.sim.data.qpos[env_ids].clone()
    initial_qvel = env.sim.data.qvel[env_ids].clone()
    pending_mask = torch.ones(len(env_ids), device=env.device, dtype=torch.bool)
    if not hasattr(env, "_seated_reset_accept_attempt_hist"):
        env._seated_reset_accept_attempt_hist = torch.zeros(
            max_attempts, device=env.device, dtype=torch.long
        )
        env._seated_reset_rejections = torch.zeros(4, device=env.device, dtype=torch.long)
        env._seated_reset_samples = {}
        env._seated_sole_gap_bins = torch.zeros(4, device=env.device, dtype=torch.long)
        env._seated_last_predicted_gap_m = torch.full(
            (env.num_envs,), float("nan"), device=env.device
        )
    terrain = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    robot = env.scene["robot"]
    joint_reset = env.cfg.events["reset_robot_joints"]
    sole_ids = [int(robot.indexing.geom_ids[robot.geom_names.index(f"{side}_sole")])
                for side in ("left", "right")]
    for attempt in range(max_attempts):
        pending = env_ids[pending_mask]
        env.sim.data.qpos[pending] = initial_qpos[pending_mask]
        env.sim.data.qvel[pending] = initial_qvel[pending_mask]
        joint_reset.func(env, pending, **joint_reset.params)
        ape.microduck_mdp.set_random_ground_state(env, pending, **kwargs)
        env.sim.forward()
        require_seated_contact_capacity(env)
        contact = env.sim.data.contact
        geoms = contact.geom
        active = torch.arange(contact.dist.shape[0], device=env.device) < env.sim.data.nacon[0]
        self_pair = (active & (geoms[:, 0] >= 0) & (geoms[:, 1] >= 0)
                     & (geoms[:, 0] != terrain) & (geoms[:, 1] != terrain))
        deep_self = self_pair & (contact.dist < -0.001)
        floor_pair = active & (((geoms[:, 0] == terrain) & (geoms[:, 1] >= 0)
                                & (geoms[:, 1] != terrain)) | \
                               ((geoms[:, 1] == terrain) & (geoms[:, 0] >= 0)
                                & (geoms[:, 0] != terrain)))
        sole_pairs = [floor_pair & (((geoms[:, 0] == terrain) & (geoms[:, 1] == sole)) |
                                    ((geoms[:, 1] == terrain) & (geoms[:, 0] == sole)))
                      for sole in sole_ids]
        floor_nonsole = floor_pair & ~sole_pairs[0] & ~sole_pairs[1]
        sole_smallest = []
        has_both_soles = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        for pair in sole_pairs:
            closest = torch.full((env.num_envs,), float("inf"), device=env.device)
            closest.scatter_reduce_(0, contact.worldid[pair].long(),
                                    contact.dist[pair], reduce="amin")
            sole_smallest.append(closest)
            has_both_soles &= torch.isfinite(closest)
        smallest = torch.minimum(sole_smallest[0], sole_smallest[1])
        invalid = torch.isin(pending, contact.worldid[deep_self].long())
        missing_floor = ~has_both_soles[pending]
        nonsole_floor = torch.isin(pending, contact.worldid[floor_nonsole].long())
        env._seated_reset_rejections[0] += torch.count_nonzero(invalid)
        env._seated_reset_rejections[1] += torch.count_nonzero(missing_floor)
        env._seated_reset_rejections[2] += torch.count_nonzero(nonsole_floor)
        invalid |= missing_floor | nonsole_floor
        candidates = pending[~invalid]
        lift_by_world = torch.zeros(env.num_envs, device=env.device)
        if len(candidates):
            # V11 found up to 2.1 mm batched/offline contact disagreement.
            # The 3.5 mm clearance leaves the feet free to settle on step one.
            lift = (-smallest[candidates] + 0.0035).clamp_min(0.0)
            lift_by_world[candidates] = lift
            env.sim.data.qpos[candidates, 2] += lift
            env.sim.forward()
            require_seated_contact_capacity(env)
            contact = env.sim.data.contact
            geoms = contact.geom
            active = torch.arange(contact.dist.shape[0], device=env.device) < env.sim.data.nacon[0]
            self_pair = (active & (geoms[:, 0] >= 0) & (geoms[:, 1] >= 0)
                         & (geoms[:, 0] != terrain) & (geoms[:, 1] != terrain))
            floor_pair = active & (((geoms[:, 0] == terrain) & (geoms[:, 1] >= 0)
                                    & (geoms[:, 1] != terrain)) | \
                                   ((geoms[:, 1] == terrain) & (geoms[:, 0] >= 0)
                                    & (geoms[:, 0] != terrain)))
            deep_post = (self_pair | floor_pair) & (contact.dist < -0.001)
            post_invalid = torch.isin(candidates, contact.worldid[deep_post].long())
            env._seated_reset_rejections[3] += torch.count_nonzero(post_invalid)
            invalid[~invalid] = post_invalid
        accepted = pending[~invalid]
        env._seated_reset_accept_attempt_hist[attempt] += len(accepted)
        if len(accepted):
            predicted_gaps = torch.maximum(
                sole_smallest[0][accepted], sole_smallest[1][accepted]
            ) + lift_by_world[accepted]
            env._seated_last_predicted_gap_m[accepted] = predicted_gaps
            env._seated_sole_gap_bins[0] += (predicted_gaps <= 0.005).sum()
            env._seated_sole_gap_bins[1] += ((predicted_gaps > 0.005)
                                             & (predicted_gaps <= 0.010)).sum()
            env._seated_sole_gap_bins[2] += ((predicted_gaps > 0.010)
                                             & (predicted_gaps <= 0.015)).sum()
            env._seated_sole_gap_bins[3] += (predicted_gaps > 0.015).sum()
        phase = getattr(env, "_seated_reset_phase", "full_reset")
        samples = env._seated_reset_samples.setdefault(f"{phase}:attempt{attempt + 1}", [])
        for world in accepted[:max(0, 4 - len(samples))].tolist():
            samples.append({
                "world": world,
                "attempt": attempt + 1,
                "phase": phase,
                "predicted_sole_floor_gaps_m": [
                    float(x[world] + lift_by_world[world]) for x in sole_smallest
                ],
                "qpos": env.sim.data.qpos[world].tolist(),
                "qvel": env.sim.data.qvel[world].tolist(),
            })
        pending_mask[pending_mask.clone()] = invalid
        if not pending_mask.any():
            return
    failed = env_ids[pending_mask]
    contact = env.sim.data.contact
    active = torch.arange(contact.dist.shape[0], device=env.device) < env.sim.data.nacon[0]
    deep = active & (contact.dist < -0.001) & torch.isin(contact.worldid.long(), failed.long())
    examples = []
    for world, pair, depth in zip(contact.worldid[deep][:8].tolist(),
                                  contact.geom[deep][:8].tolist(),
                                  contact.dist[deep][:8].tolist()):
        examples.append({
            "world": world,
            "geoms": [mujoco.mj_id2name(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, g)
                      for g in pair],
            "depth_m": depth,
        })
    raise ValueError(
        f"Seated rise reset exhausted {max_attempts} independent clearance draws: "
        f"worlds={failed[:8].tolist()} of {len(env_ids)}, "
        f"rejections_self_missing_sole_nonsole_floor_post_lift={env._seated_reset_rejections.tolist()}, "
        f"contacts={examples}"
    )


def configure_seated_rise_stage(cfg) -> None:
    """Bounded seated-only task matching the native reference entry family.

    This is an environment preflight, not approval to launch PPO. In particular
    the complete pass/fall/no-op reward audit is still required.
    """
    command = cfg.commands["twist"]
    command.sit_prob = 0.0
    command.resampling_time_range = (100.0, 100.0)
    command.ramp_s = 2.0
    reset = cfg.events["set_ground_state"].params
    reset.update({
        "face_down_prob": 0.0, "face_up_prob": 0.0,
        "sitting_prob": 1.0, "standing_prob": 0.0,
        "sitting_z_min": SQUAT_Z, "sitting_z_max": SQUAT_Z + 0.004,
        "sitting_joint_noise_std": 0.02,
        "sitting_tilt_max": math.radians(4),
    })
    cfg.events["set_ground_state"].func = reset_seated_rise_clearance
    for group in ("actor", "critic"):
        cfg.observations[group].terms["rise_phase"] = ObservationTermCfg(
            func=seated_rise_phase,
        )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ape.ContactSensorCfg(
            name="nonsole_body_ground_contact",
            primary=ape.ContactMatch(
                mode="geom",
                pattern=(
                    r"^pelvis_frame_budget$",
                    r"^torso_frame_compute_budget$",
                    r"^(left|right)_(thigh|shin|upper_arm|forearm)_frame_budget$",
                ),
                entity="robot",
            ),
            secondary=ape.ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="none", num_slots=1,
        ),
    )
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(20),
                "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",))},
    )
    cfg.terminations["root_too_low"] = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.15, "asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.terminations["nonsole_body_ground"] = TerminationTermCfg(
        func=nonsole_body_on_terrain,
    )
    cfg.terminations["hand_contact"] = TerminationTermCfg(
        func=hand_on_terrain,
    )


def build_seated_rise_env_cfg(body: Path, *, variance: bool = True,
                              model: mujoco.MjModel | None = None):
    """Build the same seated task for training and native replay."""
    model = model if model is not None else require_direct_frame(body)
    cfg = load_env_cfg(TASKS["squat"], play=False)
    retarget(cfg, body, model, "squat", variance=variance)
    configure_seated_rise_stage(cfg)
    return cfg


def reset_recovery_near_floor(env, env_ids, **kwargs) -> None:
    """Place sampled prone poses at first floor contact before the first step."""
    ape.microduck_mdp.set_random_ground_state(env, env_ids, **kwargs)
    qpos = env.sim.data.qpos
    quat = qpos[env_ids, 3:7]
    up_z = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
    prone_ids = env_ids[up_z < 0.5]
    if len(prone_ids) == 0:
        return
    env.sim.forward()
    contact = env.sim.data.contact
    terrain = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    geoms = contact.geom
    floor_pair = ((geoms[:, 0] == terrain) & (geoms[:, 1] > 0)) | \
                 ((geoms[:, 1] == terrain) & (geoms[:, 0] > 0))
    smallest = torch.full((env.num_envs,), float("inf"), device=env.device)
    smallest.scatter_reduce_(0, contact.worldid[floor_pair].long(),
                             contact.dist[floor_pair], reduce="amin")
    if bool(torch.isinf(smallest[prone_ids]).any()):
        raise ValueError("Prone reset has no floor contact; cannot place near floor")
    qpos[prone_ids, 2] += (-smallest[prone_ids] + 0.0005).clamp_min(0.0)
    env.sim.forward()


def remap_arm_indices(cfg) -> None:
    remap = {14: 13, 19: 18}
    for term in cfg.rewards.values():
        params = term.params
        if "sit_overrides" in params:
            params["sit_overrides"] = {
                remap.get(k, k): v for k, v in params["sit_overrides"].items()
            }
        if "joint_indices" in params:
            params["joint_indices"] = [remap.get(k, k)
                                        for k in params["joint_indices"]]


def retarget(cfg, body: Path, model: mujoco.MjModel, lesson: str,
             *, variance: bool = True, effort_weight: float = 0.0) -> None:
    lesson_sit_z = STAND_Z if lesson in ("arms", "independent-arms") else SQUAT_Z
    cfg.scene.entities["robot"].spec_fn = partial(ape._ape_spec, body)
    cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
    give_soles_contact_priority(cfg)
    configure_frame50_motors(cfg, body, rail_v=12.0, delay_s=0.0)
    if variance:
        cfg.scene.entities["robot"].articulation.actuators = tuple(
            replace(act, delay_min_lag=0, delay_max_lag=4)
            for act in cfg.scene.entities["robot"].articulation.actuators
        )
    cfg.scene.entities["robot"].init_state.pos = (0.0, 0.0, STAND_Z)
    pose = cfg.scene.entities["robot"].init_state.joint_pos
    pose[r".*shoulder_pitch.*"] = 0.0
    pose[r".*elbow_pitch.*"] = 0.0
    cfg.actions["joint_pos"].scale.pop("^head_pitch$", None)
    if lesson == "squat":
        # The 35 cm action scale clips before Cove's legal -1.16/1.46 rad
        # squat targets. Keep the new targets reachable inside +/-1 raw action.
        for side in ("left", "right"):
            cfg.actions["joint_pos"].scale[f"^{side}_hip_pitch$"] = 1.5
            cfg.actions["joint_pos"].scale[f"^{side}_knee$"] = 1.6
            cfg.actions["joint_pos"].scale[f"^{side}_shoulder_pitch$"] = 1.2

    for command in cfg.commands.values():
        if hasattr(command, "stand_z"):
            command.stand_z = STAND_Z
        if hasattr(command, "sit_z"):
            command.sit_z = lesson_sit_z
    reset = cfg.events["set_ground_state"].params
    reset["standing_z_min"], reset["standing_z_max"] = STAND_Z, STAND_Z + 0.004
    reset["sitting_z_min"], reset["sitting_z_max"] = SQUAT_Z, SQUAT_Z + 0.004
    reset["sitting_joint_overrides"] = (
        SQUAT_LEGS | SQUAT_COUNTERBALANCE_ARMS
        if lesson == "squat" else SQUAT_LEGS
    )
    reset["sitting_joint_noise_std"] = 0.04
    reset["sitting_tilt_max"] = math.radians(4)
    for term in cfg.rewards.values():
        for key, value in (("stand_z", STAND_Z), ("sit_z", lesson_sit_z)):
            if key in term.params:
                term.params[key] = value
        if "sit_overrides" in term.params and lesson in ("stand", "squat"):
            term.params["sit_overrides"] = SQUAT_LEGS
    if lesson == "squat":
        cfg.rewards["counterbalance_arms"] = RewardTermCfg(
            func=ape.microduck_mdp.posture_pose_match,
            weight=2.0,
            params={"command_name": "twist",
                    "sit_overrides": SQUAT_COUNTERBALANCE_ARMS,
                    "joint_indices": list(SQUAT_COUNTERBALANCE_ARMS),
                    "std": 0.4},
        )
    if lesson in ("arms", "independent-arms"):
        remap_arm_indices(cfg)
    if effort_weight:
        if effort_weight >= 0:
            raise ValueError("Effort penalty must have a negative weight")
        cfg.rewards["joint_torques_l2"] = RewardTermCfg(
            func=ape.microduck_mdp.joint_torques_l2,
            weight=effort_weight,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

    if not variance:
        return
    body_names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                       for i in range(1, model.nbody))
    cfg.events["cove_link_density"] = EventTermCfg(
        mode="reset", func=dr.pseudo_inertia,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=body_names),
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
    parser.add_argument("--lesson", choices=TASKS, required=True)
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=256)
    parser.add_argument("--updates", type=int, default=150)
    parser.add_argument("--resume-run")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--effort-weight", type=float, default=0.0)
    parser.add_argument("--shoulder-target-margin-rad", type=float)
    parser.add_argument("--leg-target-margin-rad", type=float)
    parser.add_argument("--terminate-on-fall", action="store_true")
    parser.add_argument("--expose-squat-blend", action="store_true")
    parser.add_argument("--mask-action-history", action="store_true")
    parser.add_argument("--recovery-resets", action="store_true",
                        help="For stand, sample prone, seated and upright recovery entries")
    parser.add_argument("--seated-rise-stage", action="store_true",
                        help="Separate seated-only squat-to-stand environment preflight; paid fit needs reward review")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Compile and step entries on --preflight-device; do not train")
    parser.add_argument("--preflight-device", default="cpu",
                        help="Device for --preflight-only; use cuda:0 before paid updates")
    parser.add_argument("--preflight-worlds", type=int, default=4,
                        help="Worlds in a reset/first-step preflight")
    parser.add_argument("--preflight-resets", type=int, default=16,
                        help="Reset/first-step cycles in a preflight")
    parser.add_argument("--preflight-compact", action="store_true",
                        help="Aggregate large reset stress instead of serializing every entry")
    parser.add_argument("--preflight-auto-reset-steps", type=int, default=0,
                        help="Exercise partial automatic resets with initial-policy-scale actions")
    parser.add_argument("--preflight-auto-reset-min-accepted", type=int, default=0,
                        help="Require at least this many accepted entries before stress completes")
    parser.add_argument("--preflight-action-std", type=float, default=0.2,
                        help="Reset-stress random action std; V23 initial PPO logged 0.20")
    args = parser.parse_args()
    if args.fresh and (args.resume_run or args.resume_checkpoint):
        parser.error("--fresh and resume checkpoint are mutually exclusive")
    if not args.fresh and not (args.resume_run and args.resume_checkpoint):
        parser.error("--resume-run and --resume-checkpoint are required unless --fresh")
    if args.seated_rise_stage and (args.lesson != "squat" or args.recovery_resets
                                   or args.terminate_on_fall or args.expose_squat_blend):
        parser.error("Seated rise is a separate squat stage; no prone resets or blend-slot replacement")
    body = args.body_xml.resolve()
    model = require_direct_frame(body)
    if args.seated_rise_stage:
        cfg = build_seated_rise_env_cfg(body, model=model)
    else:
        cfg = load_env_cfg(TASKS[args.lesson], play=False)
        retarget(cfg, body, model, args.lesson,
                 effort_weight=args.effort_weight)
    if args.recovery_resets:
        if args.lesson != "stand":
            parser.error("Recovery resets require the stand lesson")
        cfg.events["set_ground_state"].params.update({
            "face_down_prob": 0.25,
            "face_up_prob": 0.25,
            "sitting_prob": 0.25,
            "standing_prob": 0.25,
            "prone_z_min": 0.020,
            "prone_z_max": 0.020,
            "sitting_z_min": 0.188,
            "sitting_z_max": 0.194,
            "face_up_roll_max": math.radians(60),
        })
        cfg.events["set_ground_state"].func = reset_recovery_near_floor
    if args.expose_squat_blend:
        if args.lesson != "squat":
            parser.error("Blend observation is configured for the squat lesson")
        expose_squat_blend(cfg)
    if args.mask_action_history:
        if args.lesson != "squat":
            parser.error("Action-history mask is configured for the squat lesson")
        mask_squat_action_history(cfg)
    if args.shoulder_target_margin_rad is not None:
        if args.lesson not in ("arms", "independent-arms"):
            parser.error("Shoulder target clipping requires an arm lesson")
        clip_frame50_shoulder_targets(cfg, body,
                                      margin_rad=args.shoulder_target_margin_rad)
    if args.leg_target_margin_rad is not None:
        if args.lesson not in ("stand", "squat"):
            parser.error("Leg target clipping requires a posture lesson")
        clip_frame50_leg_targets(cfg, body,
                                 margin_rad=args.leg_target_margin_rad)
    if args.terminate_on_fall:
        if args.lesson != "squat":
            parser.error("Fall termination is configured for the squat lesson")
        cfg.terminations["fell_over"] = TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": 0.7,
                    "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",))},
        )
        cfg.terminations["hand_contact"] = TerminationTermCfg(
            func=mdp.illegal_contact,
            params={"sensor_name": "hand_ground_contact", "force_threshold": 1.0},
        )
    agent = deepcopy(load_rl_cfg(TASKS[args.lesson]))
    agent.experiment_name = ("ape_cove50_seated_rise" if args.seated_rise_stage
                             else f"ape_cove50_{args.lesson.replace('-', '_')}")
    task_id = "Mjlab-Cove50-seated-rise" if args.seated_rise_stage else f"Mjlab-Cove50-{args.lesson}"
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
    train.agent.resume = not args.fresh
    if not args.fresh:
        train.agent.load_run = args.resume_run
        train.agent.load_checkpoint = args.resume_checkpoint
    if args.preflight_only:
        train.env.scene.num_envs = args.preflight_worlds
        train.env.auto_reset = args.preflight_auto_reset_steps > 0
        train.env.seed = 99
        local = ManagerBasedRlEnv(cfg=train.env, device=args.preflight_device)
        observation, _ = local.reset()
        report = {
            "schema": "robot-learning-lab.cove50-posture-entry.v1",
            "lesson": args.lesson,
            "device": args.preflight_device,
            "preflight_worlds": args.preflight_worlds,
            "preflight_resets": args.preflight_resets,
            "preflight_compact": args.preflight_compact,
            "preflight_auto_reset_steps_requested": args.preflight_auto_reset_steps,
            "preflight_auto_reset_min_accepted": args.preflight_auto_reset_min_accepted,
            "preflight_action_std": args.preflight_action_std,
            "task": task_id,
            "recovery_resets": args.recovery_resets,
            "seated_rise_stage": args.seated_rise_stage,
            "terminations": list(train.env.terminations),
            "reward_terms": {name: term.weight for name, term in train.env.rewards.items()},
            "actor_observation_terms": list(train.env.observations["actor"].terms),
            "actor_observation_corruption": train.env.observations["actor"].enable_corruption,
            "command_class": train.env.commands["twist"].class_type.__name__,
            "motor_delay_lag_range_steps": sorted(set(
                (act.delay_min_lag, act.delay_max_lag)
                for act in train.env.scene.entities["robot"].articulation.actuators
            )),
            "body": str(body),
            "actor_shape": list(observation["actor"].shape),
            "critic_shape": list(observation["critic"].shape),
            "action_shape": list(local.action_space.shape),
            "runtime_integrator": int(local.sim.mj_model.opt.integrator),
            "configured_integrator": train.env.sim.mujoco.integrator,
            "actor_first_12": observation["actor"][0, :12].tolist(),
            "actor_last_4": observation["actor"][0, -4:].tolist(),
            "first_command": local.command_manager.get_command("twist")[0].tolist(),
            "first_rise_phase": float(local.command_manager.get_term("twist").alpha[0]),
            "reset_buckets": [],
            "claim_limit": "Reset and first-step setup entries, not learned recovery or policy score.",
        }
        if args.seated_rise_stage:
            assert torch.allclose(observation["actor"][:, -1],
                                  local.command_manager.get_term("twist").alpha)
        probe = mujoco.MjData(local.sim.mj_model)
        robot = local.scene["robot"]
        sole_ids = [int(robot.indexing.geom_ids[robot.geom_names.index(f"{side}_sole")])
                    for side in ("left", "right")]
        torso_id = int(robot.indexing.body_ids[robot.body_names.index("torso")])
        if args.seated_rise_stage:
            report["sampled_physics"] = []
            report["first_step_terminations"] = []
            report["entry_actor_nonfinite_count"] = 0
            report["first_step_termination_count"] = 0
            report["first_step_named_termination_counts"] = {
                name: 0 for name in local.termination_manager.active_terms
            }
        for reset_index in range(args.preflight_resets):
            observation, _ = local.reset()
            if args.seated_rise_stage:
                report["entry_actor_nonfinite_count"] += int((
                    ~torch.isfinite(observation["actor"])
                ).sum())
            qpos = local.sim.data.qpos[:, :7].clone()
            if args.seated_rise_stage and not args.preflight_compact:
                for world in range(local.num_envs):
                    report["sampled_physics"].append({
                        "torso_mass_kg": float(local.sim.model.body_mass[world, torso_id]),
                        "sole_slide_friction": [
                            float(local.sim.model.geom_friction[world, geom, 0])
                            for geom in sole_ids
                        ],
                    })
            for index, state in enumerate(qpos if not args.preflight_compact else ()):
                w, x, y, _ = state[3:7].tolist()
                up_z = 1.0 - 2.0 * (x * x + y * y)
                if up_z < 0.5:
                    bucket = "prone"
                else:
                    bucket = "seated" if state[2] < 0.205 else "upright"
                probe.qpos[:] = local.sim.data.qpos[index].cpu().numpy()
                probe.qvel[:] = local.sim.data.qvel[index].cpu().numpy()
                mujoco.mj_forward(local.sim.mj_model, probe)
                closest_contact = min(probe.contact, key=lambda contact: contact.dist,
                                      default=None)
                terrain_id = mujoco.mj_name2id(local.sim.mj_model,
                    mujoco.mjtObj.mjOBJ_GEOM, "terrain")
                floor_contacts = [contact for contact in probe.contact
                                  if terrain_id in (contact.geom1, contact.geom2)]
                closest_floor = min(floor_contacts, key=lambda contact: contact.dist,
                                    default=None)
                closest = float(closest_contact.dist) if closest_contact is not None else 0.0
                names = ([mujoco.mj_id2name(local.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                          for geom in (closest_contact.geom1, closest_contact.geom2)]
                         if closest_contact is not None else [])
                report["reset_buckets"].append({
                    "bucket": bucket,
                    "root_height_m": float(state[2]),
                    "up_z": up_z,
                    "quat_wxy": [w, x, y],
                    "deepest_contact_distance_m": closest,
                    "deepest_contact_geoms": names,
                    "deepest_terrain_contact_distance_m": (
                        float(closest_floor.dist) if closest_floor is not None else 0.0
                    ),
                    "has_self_contact": any(
                        terrain_id not in (contact.geom1, contact.geom2)
                        for contact in probe.contact
                    ),
                })
            _, _, terminated, truncated, _ = local.step(
                torch.zeros(local.action_space.shape, device=args.preflight_device)
            )
            if reset_index == 0:
                report["first_zero_action_processed_target"] = (
                    local.action_manager.get_term("joint_pos")._processed_actions[0].tolist()
                )
            if args.seated_rise_stage:
                report["first_step_termination_count"] += int(terminated.sum() + truncated.sum())
                for name in local.termination_manager.active_terms:
                    report["first_step_named_termination_counts"][name] += int(
                        local.termination_manager.get_term(name).sum()
                    )
                if not args.preflight_compact:
                    report["first_step_terminations"].append({
                        "terminated": [bool(x) for x in terminated.tolist()],
                        "truncated": [bool(x) for x in truncated.tolist()],
                        "named_terms": {
                            name: [bool(x) for x in local.termination_manager.get_term(name).tolist()]
                            for name in local.termination_manager.active_terms
                        },
                    })
        if args.preflight_auto_reset_steps:
            local._seated_reset_phase = "auto_partial"
            previous_done = torch.ones(local.num_envs, device=args.preflight_device,
                                       dtype=torch.bool)
            auto_dones = torch.zeros((), device=args.preflight_device, dtype=torch.long)
            immediate_dones = torch.zeros_like(auto_dones)
            actor_nonfinite = torch.zeros_like(auto_dones)
            reward_nonfinite = torch.zeros_like(auto_dones)
            qpos_nonfinite = torch.zeros_like(auto_dones)
            target_nonfinite = torch.zeros_like(auto_dones)
            auto_named = {name: torch.zeros_like(auto_dones)
                          for name in local.termination_manager.active_terms}
            max_steps = args.preflight_auto_reset_steps + 1200
            step_count = 0
            while step_count < max_steps:
                actions = (torch.randn(local.action_space.shape, device=args.preflight_device)
                           * args.preflight_action_std)
                actions[previous_done] = 0.0
                observation, rewards, terminated, truncated, _ = local.step(actions)
                dones = terminated | truncated
                auto_dones += dones.sum()
                immediate_dones += (dones & previous_done).sum()
                actor_nonfinite += (~torch.isfinite(observation["actor"])).sum()
                reward_nonfinite += (~torch.isfinite(rewards)).sum()
                qpos_nonfinite += (~torch.isfinite(local.sim.data.qpos)).sum()
                target_nonfinite += (~torch.isfinite(
                    local.action_manager.get_term("joint_pos")._processed_actions
                )).sum()
                for name in auto_named:
                    auto_named[name] += local.termination_manager.get_term(name).sum()
                previous_done = dones
                step_count += 1
                if (step_count >= args.preflight_auto_reset_steps
                    and int(local._seated_reset_accept_attempt_hist.sum())
                    >= args.preflight_auto_reset_min_accepted):
                    break
            report["auto_reset_stress"] = {
                "steps_per_world": step_count,
                "partial_reset_events": int(auto_dones),
                "first_step_zero_action_terminations": int(immediate_dones),
                "actor_nonfinite_count": int(actor_nonfinite),
                "reward_nonfinite_count": int(reward_nonfinite),
                "qpos_nonfinite_count": int(qpos_nonfinite),
                "processed_target_nonfinite_count": int(target_nonfinite),
                "named_termination_counts": {k: int(v) for k, v in auto_named.items()},
                "min_accepted_reached": (int(local._seated_reset_accept_attempt_hist.sum())
                                         >= args.preflight_auto_reset_min_accepted),
            }
        if args.seated_rise_stage:
            report["reset_accept_attempt_hist"] = (
                local._seated_reset_accept_attempt_hist.tolist()
            )
            report["reset_rejections_self_missing_sole_nonsole_floor_post_lift"] = (
                local._seated_reset_rejections.tolist()
            )
            report["reset_accepted_entries"] = sum(report["reset_accept_attempt_hist"])
            report["contact_peak_active_broadphase_capacity"] = local._seated_contact_peak
            report["max_sole_floor_gap_bins_le5_le10_le15_gt15mm"] = (
                local._seated_sole_gap_bins.tolist()
            )
            report["accepted_source_state_samples_by_attempt"] = local._seated_reset_samples
        local.close()
        print(json.dumps(report, indent=2))
        return
    launch_training(task_id, train)


if __name__ == "__main__":
    main()
