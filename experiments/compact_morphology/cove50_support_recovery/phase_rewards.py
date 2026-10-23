"""Phase-aware stand/one-foot shaping for the selected Cove body."""

from __future__ import annotations

import torch
import mujoco
import numpy as np
from mjlab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg

from ape_walk import task as ape
from .phase_actions import phase_blend


def _phase(env, lift_s, hold_s, lower_s, staged, stage="all"):
    return phase_blend(env, lift_s, hold_s, lower_s,
                       stage if staged else "all")


def phase_command(env, lift_s, hold_s, lower_s, staged, stance):
    stance_phase = _phase(env, lift_s, hold_s, lower_s, staged, "stance")
    swing = _phase(env, lift_s, hold_s, lower_s, staged, "swing")
    side = 1.0 if stance == "right" else -1.0
    return torch.stack((stance_phase, side * swing if staged else
                        side * torch.ones_like(stance_phase),
                        torch.zeros_like(stance_phase)), dim=-1)


def phase_pose(env, target, std, arm_index, arm_bump, swing_indices,
               lift_s, hold_s, lower_s, staged, stance):
    del stance
    robot = env.scene["robot"]
    phase = _phase(env, lift_s, hold_s, lower_s, staged, "stance")[:, None]
    end = torch.as_tensor(target, device=env.device, dtype=robot.data.joint_pos.dtype)[None, :]
    home = robot.data.default_joint_pos
    desired = home + phase * (end - home)
    if staged:
        swing = _phase(env, lift_s, hold_s, lower_s, staged, "swing")[:, None]
        desired[:, swing_indices] = (home[:, swing_indices] + swing *
                                     (end[:, swing_indices] - home[:, swing_indices]))
    if arm_bump:
        desired[:, arm_index] += arm_bump * torch.sin(torch.pi * phase[:, 0])
    return torch.exp(-((robot.data.joint_pos - desired) / std) ** 2).mean(dim=-1)


def phase_gravity(env, target, std, lift_s, hold_s, lower_s, staged, stance):
    del stance
    phase = _phase(env, lift_s, hold_s, lower_s, staged, "stance")[:, None]
    g = env.scene["robot"].data.projected_gravity_b
    end = torch.as_tensor(target, device=env.device, dtype=g.dtype)[None, :]
    home = torch.tensor((0.0, 0.0, -1.0), device=env.device, dtype=g.dtype)[None, :]
    desired = torch.nn.functional.normalize(home + phase * (end - home), dim=-1)
    return torch.exp(-((g - desired) ** 2).sum(dim=-1) / std**2)


def endpoint_com_offset(model, pose_spec, stance="right"):
    """Whole-body CoM offset from the stance sole in the screened endpoint."""
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[:3] = pose_spec["root_xyz_roll_pitch"][:3]
    quat = np.empty(4)
    mujoco.mju_euler2Quat(
        quat, np.array((*pose_spec["root_xyz_roll_pitch"][3:], 0.0)), "xyz")
    data.qpos[3:7] = quat
    for i in range(model.nu):
        joint = model.actuator_trnid[i, 0]
        data.qpos[model.jnt_qposadr[joint]] = pose_spec["joints"][model.joint(joint).name]
    mujoco.mj_forward(model, data)
    offset = (data.subtree_com[model.body("pelvis").id, :2] -
              data.site_xpos[model.site(f"{stance}_sole_center").id, :2])
    return tuple(float(x) for x in offset)


def phase_com(env, std, com_offset_xy, lift_s, hold_s, lower_s, staged, stance):
    robot = env.scene["robot"]
    swing_side = "left" if stance == "right" else "right"
    support = robot.data.site_pos_w[:, robot.site_names.index(f"{stance}_sole_center"), :2]
    swing_foot = robot.data.site_pos_w[:, robot.site_names.index(f"{swing_side}_sole_center"), :2]
    phase = _phase(env, lift_s, hold_s, lower_s, staged, "stance")[:, None]
    offset = torch.as_tensor(com_offset_xy, device=env.device, dtype=support.dtype)
    desired = 0.5 * (swing_foot + support) * (1.0 - phase) + (support + offset) * phase
    actual = env.sim.data.subtree_com[:, robot.indexing.root_body_id, :2]
    return torch.exp(-((actual - desired) ** 2).sum(dim=-1) / std**2)


def phase_swing_height(env, height, std, lift_s, hold_s, lower_s, staged,
                       stance, linear):
    robot = env.scene["robot"]
    swing = "left" if stance == "right" else "right"
    site = robot.data.site_pos_w[:, robot.site_names.index(f"{swing}_sole_center"), 2]
    z = site - env.scene.terrain.env_origins[:, 2]
    desired = height * _phase(env, lift_s, hold_s, lower_s, staged, "swing")
    if linear:
        return torch.clamp(1.0 - torch.abs(z - desired) / height, min=0.0)
    return torch.exp(-((z - desired) / std) ** 2)


def phase_contacts(env, lift_s, hold_s, lower_s, staged, stance):
    feet = env.scene.sensors["feet_ground_contact"].data.found.float()
    phase = _phase(env, lift_s, hold_s, lower_s, staged, "swing")
    stance_idx = 1 if stance == "right" else 0
    swing_idx = 1 - stance_idx
    return feet[:, stance_idx] * ((1.0 - phase) * feet[:, swing_idx] +
                                  phase * (1.0 - feet[:, swing_idx]))


def single_support_success(env, lift_s, hold_s, lower_s, staged, stance):
    phase = _phase(env, lift_s, hold_s, lower_s, staged, "swing")
    feet = env.scene.sensors["feet_ground_contact"].data.found.bool()
    robot = env.scene["robot"]
    swing = "left" if stance == "right" else "right"
    stance_idx = 1 if stance == "right" else 0
    swing_idx = 1 - stance_idx
    height = robot.data.site_pos_w[:, robot.site_names.index(f"{swing}_sole_center"), 2]
    return ((phase > 0.95) & feet[:, stance_idx] & ~feet[:, swing_idx] &
            (height - env.scene.terrain.env_origins[:, 2] > 0.025)).float()


def configure_phase_rewards(cfg, pose, gravity, *, lift_s, hold_s, lower_s,
                            arm_index, arm_bump, swing_indices, staged,
                            com_offset_xy, stance, urgent_support=False):
    schedule = {"lift_s": lift_s, "hold_s": hold_s, "lower_s": lower_s,
                "staged": staged, "stance": stance}
    for group in ("actor", "critic"):
        cfg.observations[group].terms["command"] = ObservationTermCfg(
            func=phase_command, params=schedule)
    for term in ("com_over_stance_foot", "stance_foot_grounded",
                 "swing_foot_clear", "swing_foot_touch", "pose_flamingo",
                 "gravity_flamingo"):
        cfg.rewards.pop(term)
    cfg.rewards["phase_pose"] = RewardTermCfg(
        func=phase_pose, weight=1.5,
        params={**schedule, "target": pose, "std": 0.5,
                "arm_index": arm_index, "arm_bump": arm_bump,
                "swing_indices": swing_indices})
    cfg.rewards["phase_gravity"] = RewardTermCfg(
        func=phase_gravity, weight=2.0,
        params={**schedule, "target": gravity, "std": 0.15})
    cfg.rewards["phase_com"] = RewardTermCfg(
        func=phase_com, weight=3.0,
        params={**schedule, "std": 0.04,
                "com_offset_xy": com_offset_xy})
    cfg.rewards["phase_swing_height"] = RewardTermCfg(
        func=phase_swing_height, weight=4.0 if urgent_support else 1.5,
        params={**schedule, "height": 0.06, "std": 0.03,
                "linear": urgent_support})
    cfg.rewards["phase_contacts"] = RewardTermCfg(
        func=phase_contacts, weight=3.0 if urgent_support else 1.5,
        params=schedule)
    cfg.rewards["single_support_success"] = RewardTermCfg(
        func=single_support_success, weight=10.0 if urgent_support else 3.0,
        params=schedule)
    cfg.rewards["foot_slip"].weight = -0.5
    cfg.curriculum.pop("flamingo_com_std", None)
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        ape.ContactSensorCfg(
            name="all_self_contact",
            primary=ape.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            secondary=ape.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
            fields=("found", "force"), reduce="none", num_slots=1),
    )
    cfg.terminations["all_self_contact"] = TerminationTermCfg(
        func=ape.mdp.illegal_contact,
        params={"sensor_name": "all_self_contact", "force_threshold": 0.1})
