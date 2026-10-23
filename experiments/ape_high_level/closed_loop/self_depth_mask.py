"""Predict camera-visible self pixels from the actor's own articulated model.

The predicted image uses only known Cove geometry, joint/head encoders and
camera calibration. An environment surface in front of the predicted self
surface remains an obstacle observation. Physical model/optic error has not
been qualified; this is a simulation diagnostic, not a motion safety gate.
"""

from __future__ import annotations

import mujoco
import numpy as np


class SelfDepthMask:
    def __init__(self, model: mujoco.MjModel, *, camera: str,
                 width: int, height: int,
                 agreement_m: float = .025) -> None:
        self.model = model
        self.camera = camera
        self.agreement_m = agreement_m
        self.data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, width=width, height=height)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = 0
        # The selected model assigns its CAD and shell visuals to 1 and 2;
        # the floor, props and walls are group 0. Collision proxies in 4 do
        # not render in the measured camera image.
        self.option.geomgroup[1] = 1
        self.option.geomgroup[2] = 1

    def apply(self, measured_depth_m: np.ndarray,
              actor_qpos: np.ndarray,
              *, max_depth_m: float = 1.4) -> tuple[np.ndarray, dict]:
        self.data.qpos[:] = actor_qpos
        self.data.qvel[:] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self.camera,
                                   scene_option=self.option)
        predicted = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        measured = np.asarray(measured_depth_m)
        predicted_self = np.isfinite(predicted) & (predicted < max_depth_m)
        valid_measured = np.isfinite(measured) & (measured > .08)
        agreement = np.abs(measured - predicted) <= self.agreement_m
        mask = predicted_self & valid_measured & agreement
        filtered = measured.copy()
        filtered[mask] = np.nan
        closer_surface = (predicted_self & valid_measured &
                          (measured < predicted - self.agreement_m))
        return filtered, {"predicted_self_pixels": int(predicted_self.sum()),
                          "masked_self_pixels": int(mask.sum()),
                          "closer_surface_pixels_preserved": int(
                              closer_surface.sum())}

    def close(self) -> None:
        self.renderer.close()
