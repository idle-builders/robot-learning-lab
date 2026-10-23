# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""i4h tray scene with a slide goal; G1 and controller are unchanged."""

from __future__ import annotations

import argparse

from arena.environments.core.humanoid_base import HumanoidEnvironmentBase


class TidySlideTrayEnvironment(HumanoidEnvironmentBase):
    name = "tidy_slide_tray"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        super().add_cli_args(parser)

    def get_env(self, args: argparse.Namespace):
        from arena.tasks.tidy_slide_tray import TidySlideTrayTask
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("pre_op")()
        tray = self.asset_registry.get_asset_by_name("surgical_tray")()
        cart = self.asset_registry.get_asset_by_name("cart")()
        embodiment = self.asset_registry.get_asset_by_name(args.embodiment)(enable_cameras=args.enable_cameras)

        background.set_initial_pose(Pose(position_xyz=(4.0, 0.0, -0.8), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        tray.set_initial_pose(Pose(position_xyz=(-1.15, -1.6, -0.08), rotation_wxyz=(0.707, 0.0, 0.0, 0.707)))
        cart.set_initial_pose(Pose(position_xyz=(0.35, -1.65, -0.7875), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        embodiment.set_initial_pose(Pose(position_xyz=(-0.5, -1.62, 0.0), rotation_wxyz=(0.0, 0.0, 0.0, 1.0)))

        self._maybe_patch_locomanip_mimic(args, embodiment)

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[background, tray, cart]),
            task=TidySlideTrayTask(
                tray,
                cart,
                background,
                episode_length_s=30.0,
                task_description="Slide the tray along the counter without lifting it.",
            ),
            teleop_device=self._resolve_teleop_device(args.teleop_device),
        )
