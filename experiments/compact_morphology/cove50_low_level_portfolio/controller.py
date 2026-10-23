"""Small action router for retained Direct Frame low-level actors.

This reproduces the historical native walk/stop/posture/arms controller's
observation adapters and action arbitration. It is intentionally independent
of policy loading and simulation so the same contract can be checked against
saved native traces before a new evaluator uses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Mode = Literal["walk", "stop", "posture", "arms"]


@dataclass(frozen=True)
class LowLevelCommand:
    mode: Mode
    forward_m_s: float = 0.0
    yaw_rad_s: float = 0.0
    squat: bool = False
    raise_arms: bool = False


class PortfolioRouter:
    """Stateful 50 Hz observation/action interface for one physical world."""

    def __init__(self, target_names: list[str], posture_action_scale: np.ndarray):
        if len(target_names) != 21:
            raise ValueError("expected 21 target joints")
        expected_legs = [f"{side}_{joint}" for side in ("left", "right") for joint in
                         ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")]
        if target_names[:10] != expected_legs:
            raise ValueError("leg action order differs from saved actor contract")
        self.shoulders = [target_names.index(f"{side}_shoulder_pitch") for side in ("left", "right")]
        self.scale = np.asarray(posture_action_scale, dtype=np.float32)
        if self.scale.shape != (21,) or np.any(self.scale[:10] <= 0):
            raise ValueError("invalid posture action scale")
        self.reset()

    def reset(self) -> None:
        self.last_gait_action = np.zeros(21, dtype=np.float32)
        self.previous_arm_flag = np.float32(0.0)
        self.filtered: np.ndarray | None = None

    def arm_observation(self, actor_obs: np.ndarray) -> np.ndarray:
        out = self._observation(actor_obs)
        out[69] = self.previous_arm_flag
        return out

    def gait_observation(self, actor_obs: np.ndarray, command: LowLevelCommand) -> np.ndarray:
        if command.mode not in ("walk", "stop"):
            raise ValueError("gait adapter requires walk or stop")
        out = self._observation(actor_obs)
        out[48:69] = self.last_gait_action
        out[69:72] = 0.0
        out[69] = command.forward_m_s if command.mode == "walk" else 0.0
        out[71] = command.yaw_rad_s if command.mode == "walk" else 0.0
        return out

    def route(self, command: LowLevelCommand, squat_action: np.ndarray,
              arm_action: np.ndarray, gait_action: np.ndarray | None = None) -> np.ndarray:
        squat_action = self._action(squat_action)
        arm_action = self._action(arm_action)
        merged = np.clip(squat_action, -1.0, 1.0).copy()
        if command.mode in ("walk", "stop"):
            if gait_action is None:
                raise ValueError("gait action required for walk or stop")
            gait_action = self._action(gait_action)
            self.last_gait_action = gait_action.copy()
            merged[:] = 0.0
            merged[:10] = gait_action[:10] * np.float32(0.3) / self.scale[:10]
        elif command.mode == "arms":
            merged[:10] = 0.0
            merged[self.shoulders] = arm_action[self.shoulders]
        elif command.mode != "posture":
            raise ValueError(f"unknown low-level mode: {command.mode}")
        if command.mode in ("walk", "stop") or self.filtered is None:
            self.filtered = merged.copy()
        else:
            self.filtered = np.float32(0.2) * merged + np.float32(0.8) * self.filtered
        self.previous_arm_flag = np.float32(command.raise_arms)
        return self.filtered.copy()

    @staticmethod
    def _observation(value: np.ndarray) -> np.ndarray:
        out = np.asarray(value, dtype=np.float32)
        if out.shape != (72,):
            raise ValueError("expected 72 actor observations")
        return out.copy()

    @staticmethod
    def _action(value: np.ndarray) -> np.ndarray:
        out = np.asarray(value, dtype=np.float32)
        if out.shape != (21,):
            raise ValueError("expected 21 actor actions")
        return out
