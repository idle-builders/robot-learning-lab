# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A small tray-slide task on the pinned i4h G1 scene and controller."""

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import torch
from arena.tasks.tray_pick_and_place import I4HEventsCfg, TrayPickPlaceTask
from isaaclab.envs.mimic_env_cfg import SubTaskConfig
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass


def _reset_slide_state(env, env_ids: torch.Tensor) -> None:
    if hasattr(env, "_slide_start_pos"):
        env._slide_start_pos[env_ids] = torch.nan
        env._slide_peak_z[env_ids] = -torch.inf
        env._slide_settle_count[env_ids] = 0


def tray_slid_on_support(
    env,
    object_cfg: SceneEntityCfg,
    min_slide_m: float = 0.15,
    max_lift_m: float = 0.08,
    max_speed_mps: float = 0.08,
    settle_steps: int = 20,
) -> torch.Tensor:
    tray = env.scene[object_cfg.name]
    pos = tray.data.root_pos_w - env.scene.env_origins
    if not hasattr(env, "_slide_start_pos"):
        env._slide_start_pos = torch.full_like(pos, torch.nan)
        env._slide_peak_z = torch.full((env.num_envs,), -torch.inf, device=env.device)
        env._slide_settle_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    first_observation = torch.isnan(env._slide_start_pos[:, 0])
    env._slide_start_pos = torch.where(first_observation[:, None], pos, env._slide_start_pos)
    env._slide_peak_z = torch.maximum(env._slide_peak_z, pos[:, 2])
    start = env._slide_start_pos
    distance = torch.linalg.norm(pos[:, :2] - start[:, :2], dim=-1)
    velocity = getattr(tray.data, "root_lin_vel_w", None)
    if velocity is None:
        velocity = tray.data.root_vel_w[:, :3]
    speed = torch.linalg.norm(velocity, dim=-1)
    ready = (
        (distance >= min_slide_m)
        & (torch.abs(pos[:, 2] - start[:, 2]) <= max_lift_m)
        & (env._slide_peak_z <= start[:, 2] + max_lift_m)
        & (speed <= max_speed_mps)
    )
    env._slide_settle_count = torch.where(
        ready,
        env._slide_settle_count + 1,
        torch.zeros_like(env._slide_settle_count),
    )
    return env._slide_settle_count >= settle_steps


@configclass
class SlideTerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out)
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg = MISSING


@configclass
class SlideEventsCfg(I4HEventsCfg):
    reset_slide_state: EventTermCfg = MISSING

    def __init__(self, pick_up_object, destination_cart):
        super().__init__(pick_up_object, destination_cart)
        self.reset_slide_state = EventTermCfg(func=_reset_slide_state, mode="reset")


class TidySlideTrayTask(TrayPickPlaceTask):
    """Only the task predicate and Mimic boundaries differ from the i4h tray task."""

    def get_events_cfg(self):
        return SlideEventsCfg(self.pick_up_object, self.destination_bin)

    def get_termination_cfg(self):
        success = TerminationTermCfg(
            func=tray_slid_on_support,
            params={"object_cfg": SceneEntityCfg(self.pick_up_object.name)},
        )
        dropped = TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.pick_up_object.name),
            },
        )
        return SlideTerminationsCfg(success=success, object_dropped=dropped)

    def get_mimic_env_cfg(self, embodiment_name: str):
        cfg = super().get_mimic_env_cfg(embodiment_name)
        cfg.datagen_config.name = "g1_tidy_slide_tray_D0"

        def one_segment():
            return SubTaskConfig(
                object_ref=self.pick_up_object.name,
                first_subtask_start_offset_range=(0, 0),
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.001,
                num_interpolation_steps=0,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )

        cfg.subtask_configs = {part: [one_segment()] for part in ("left", "right", "body")}
        return cfg
