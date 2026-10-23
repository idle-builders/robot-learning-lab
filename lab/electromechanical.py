"""Backend-independent electromechanical actuator reference model.

This is deliberately numeric and dependency-light so the Isaac worker can be
checked against it without importing Isaac.  It models a shared pack RC state
for one actuator group and uses the four-quadrant linear DC motor envelope
documented by the pinned Isaac Lab DCMotor implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ElectromechanicalConfig:
    dt_s: float = 1.0 / 500.0
    delay_steps: int = 10
    source_voltage_v: float = 12.6
    source_resistance_ohm: float = 0.100
    bulk_capacitance_f: float = 0.0022
    motor_min_voltage_v: float = 9.0
    nominal_voltage_v: float = 12.0
    stall_effort_nm: float = 2.941995
    no_load_speed_rad_s: float = 4.717106
    stall_current_a: float = 2.7
    idle_current_a: float = 0.03
    logic_power_w: float = 3.0 / 0.85
    friction_nm: float = 0.0


class ElectromechanicalPack:
    """One shared pack and 16-joint actuator-group reference model."""

    def __init__(self, kp: Any, kd: Any, effort_limit: Any, velocity_limit: Any, cfg: ElectromechanicalConfig | None = None):
        self.cfg = cfg or ElectromechanicalConfig()
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.effort_limit = np.asarray(effort_limit, dtype=float)
        self.velocity_limit = np.asarray(velocity_limit, dtype=float)
        if not (self.kp.shape == self.kd.shape == self.effort_limit.shape == self.velocity_limit.shape):
            raise ValueError("kp, kd, effort_limit and velocity_limit must have the same shape")
        if self.kp.ndim != 1 or self.kp.size != 16:
            raise ValueError("the sealed Pip actuator group must contain exactly 16 joints")
        if self.cfg.delay_steps < 0 or self.cfg.dt_s <= 0 or self.cfg.source_resistance_ohm <= 0 or self.cfg.bulk_capacitance_f <= 0:
            raise ValueError("invalid discrete pack parameters")
        self._queue = np.zeros((self.cfg.delay_steps + 1, 16, 3), dtype=float)
        self._voltage_v = self.cfg.source_voltage_v
        self.energy_j = 0.0

    @property
    def voltage_v(self) -> float:
        return float(self._voltage_v)

    def reset(self, voltage_v: float | None = None) -> None:
        self._queue.fill(0.0)
        self._voltage_v = self.cfg.source_voltage_v if voltage_v is None else float(voltage_v)
        self.energy_j = 0.0

    def _torque_window(self, joint_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Four-quadrant form: braking torque remains available at either sign
        # of speed; only motoring in the direction of motion loses headroom.
        voltage_scale = np.clip(self._voltage_v / self.cfg.nominal_voltage_v, 0.0, 1.0)
        stall = self.cfg.stall_effort_nm * voltage_scale
        speed = max(self.cfg.no_load_speed_rad_s * voltage_scale, 1e-9)
        speed_at_limit = speed * (1.0 + self.effort_limit / max(stall, 1e-9))
        bounded_vel = np.clip(joint_vel, -speed_at_limit, speed_at_limit)
        top = stall * (1.0 - bounded_vel / speed)
        bottom = stall * (-1.0 - bounded_vel / speed)
        upper = np.minimum(top, self.effort_limit)
        lower = np.maximum(bottom, -self.effort_limit)
        return np.maximum(upper, lower), np.minimum(lower, upper)

    def step(self, joint_pos: Any, joint_vel: Any, desired_pos: Any, desired_vel: Any, feedforward: Any | None = None) -> dict[str, np.ndarray | float]:
        q = np.asarray(joint_pos, dtype=float)
        dq = np.asarray(joint_vel, dtype=float)
        qd = np.asarray(desired_pos, dtype=float)
        dqd = np.asarray(desired_vel, dtype=float)
        ff = np.zeros(16) if feedforward is None else np.asarray(feedforward, dtype=float)
        if any(x.shape != (16,) for x in (q, dq, qd, dqd, ff)):
            raise ValueError("all joint vectors must have shape (16,)")
        self._queue[:-1] = self._queue[1:]
        self._queue[-1] = np.stack((qd, dqd, ff), axis=1)
        delayed_qd, delayed_dqd, delayed_ff = self._queue[0].T
        computed = self.kp * (delayed_qd - q) + self.kd * (delayed_dqd - dq) + delayed_ff
        computed -= self.cfg.friction_nm * np.sign(dq)
        upper, lower = self._torque_window(dq)
        applied = np.clip(computed, lower, upper)
        current_a = self.cfg.idle_current_a + self.cfg.stall_current_a * np.abs(applied) / self.cfg.stall_effort_nm
        current_a = np.minimum(current_a, self.cfg.idle_current_a + self.cfg.stall_current_a)
        total_current = float(np.sum(current_a) + self.cfg.logic_power_w / max(self._voltage_v, 1e-9))
        v_inf = self.cfg.source_voltage_v - self.cfg.source_resistance_ohm * total_current
        alpha = math.exp(-self.cfg.dt_s / (self.cfg.source_resistance_ohm * self.cfg.bulk_capacitance_f))
        self._voltage_v = v_inf + (self._voltage_v - v_inf) * alpha
        self.energy_j += self._voltage_v * total_current * self.cfg.dt_s
        return {
            "computed_effort_nm": computed,
            "applied_effort_nm": applied,
            "current_a": current_a,
            "total_current_a": total_current,
            "voltage_v": self._voltage_v,
            "energy_j": self.energy_j,
            "below_motor_min_voltage": bool(self._voltage_v < self.cfg.motor_min_voltage_v),
        }
