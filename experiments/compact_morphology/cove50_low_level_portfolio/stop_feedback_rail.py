"""Reset-time 10–12 V variation for Cove's explicit MJLab DC servos.

The saved Frame50 motor configuration is built at 12 V. The source catalog
scales Feetech continuous torque, stall torque and no-load speed linearly with
rail voltage; XL330 axes use their fixed 5 V rating. This event applies that
same relationship independently in each native physics world.
"""

from __future__ import annotations

import torch

from ape_walk.physics.frame50 import ROBOTIS_SMALL, part_for_joint


def sample_cove_rail(env, env_ids, low_v: float = 10.0, high_v: float = 12.0):
    if not 10.0 <= low_v <= high_v <= 12.0:
        raise ValueError("Cove rail interval must stay within the sourced 10–12 V cases")
    ids = (torch.arange(env.num_envs, device=env.device)[env_ids]
           if isinstance(env_ids, slice) else
           torch.arange(env.num_envs, device=env.device)
           if env_ids is None else env_ids)
    if len(ids) == 0:
        return
    rail = low_v + (high_v - low_v) * torch.rand(len(ids), device=env.device)
    if not hasattr(env, "stop_feedback_rail_v"):
        env.stop_feedback_rail_v = torch.zeros(env.num_envs, device=env.device)
        env.stop_feedback_rail_min = torch.full((), float("inf"), device=env.device)
        env.stop_feedback_rail_max = torch.full((), float("-inf"), device=env.device)
        env.stop_feedback_rail_samples = torch.zeros((), dtype=torch.long, device=env.device)
        env.stop_feedback_leg_rating_min = torch.full((), float("inf"), device=env.device)
        env.stop_feedback_leg_rating_max = torch.full((), float("-inf"), device=env.device)
    env.stop_feedback_rail_v[ids] = rail
    env.stop_feedback_rail_min = torch.minimum(env.stop_feedback_rail_min, rail.amin())
    env.stop_feedback_rail_max = torch.maximum(env.stop_feedback_rail_max, rail.amax())
    env.stop_feedback_rail_samples += len(ids)
    scale = (rail / 12.0)[:, None]
    for index, actuator in enumerate(env.scene["robot"].actuators):
        if len(actuator.target_names) != 1:
            raise ValueError("Cove motor rail expects one joint per DC actuator")
        if part_for_joint(actuator.target_names[0]) == ROBOTIS_SMALL:
            continue
        for field in ("force_limit", "saturation_effort", "velocity_limit_motor", "_vel_at_effort_lim"):
            if getattr(actuator, field, None) is None:
                raise ValueError(f"Cove DC actuator is missing {field}")
        actuator.force_limit[ids] = actuator.cfg.effort_limit * scale
        actuator.saturation_effort[ids] = actuator.cfg.saturation_effort * scale
        actuator.velocity_limit_motor[ids] = actuator.cfg.velocity_limit * scale
        actuator._vel_at_effort_lim[ids] = (actuator.velocity_limit_motor[ids]
                                            * (1.0 + actuator.force_limit[ids]
                                               / actuator.saturation_effort[ids]))
        if index < 10:
            env.stop_feedback_leg_rating_min = torch.minimum(
                env.stop_feedback_leg_rating_min, actuator.force_limit[ids].amin())
            env.stop_feedback_leg_rating_max = torch.maximum(
                env.stop_feedback_leg_rating_max, actuator.force_limit[ids].amax())
