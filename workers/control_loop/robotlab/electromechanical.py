"""Isaac Lab adapter for the shared-pack electromechanical model.

The worker image owns Isaac Lab.  This module remains importable without it so
the backend-independent reference and tests can run on the host.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised only inside the Isaac worker image
    import torch
    from isaaclab.actuators import DCMotor, DCMotorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.types import ArticulationActions
except ImportError:  # host environment intentionally has no Isaac installation
    torch = None
    DCMotor = None
    DCMotorCfg = None
    configclass = None
    ArticulationActions = object


if DCMotor is not None:  # pragma: no cover
    @configclass
    class ElectromechanicalDCMotorCfg(DCMotorCfg):
        class_type: type = "{DIR}.electromechanical:ElectromechanicalDCMotor"
        delay_steps: int = 10
        source_voltage_v: float = 12.6
        source_resistance_ohm: float = 0.100
        bulk_capacitance_f: float = 0.0022
        nominal_voltage_v: float = 12.0
        no_load_speed_rad_s: float = 4.717106
        stall_current_a: float = 2.7
        idle_current_a: float = 0.03
        logic_power_w: float = 3.0 / 0.85
        dt_s: float = 1.0 / 500.0


    class ElectromechanicalDCMotor(DCMotor):
        """DCMotor with one shared RC bus and bounded command delay."""

        cfg: ElectromechanicalDCMotorCfg

        def __init__(self, cfg, *args, **kwargs):
            super().__init__(cfg, *args, **kwargs)
            # The retained Newton EA backend names these actuator_*; AGILE's
            # pinned PhysX 3.0 beta2 uses effort_limit / velocity_limit. Resolve
            # native tensors once; keep the shared electrical law identical.
            self._electrical_effort_limit = (self.actuator_effort_limit
                if hasattr(self, 'actuator_effort_limit') else self.effort_limit)
            self._electrical_velocity_limit = (self.actuator_velocity_limit
                if hasattr(self, 'actuator_velocity_limit') else self.velocity_limit)
            if cfg.delay_steps < 0:
                raise ValueError("delay_steps must be non-negative")
            self._delay = torch.zeros((cfg.delay_steps + 1, self._num_envs, self.num_joints, 3), device=self._device)
            self._voltage = torch.full((self._num_envs, 1), cfg.source_voltage_v, device=self._device)
            self.current = torch.zeros_like(self.computed_effort)
            self.energy_j = torch.zeros((self._num_envs, 1), device=self._device)
            self._initialized = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

        def reset(self, env_ids):
            super().reset(env_ids)
            if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
                ids = torch.arange(self._num_envs, device=self._device)
            else:
                ids = torch.as_tensor(env_ids, device=self._device).reshape(-1)
            self._delay[:, ids] = 0.0
            self._voltage[ids] = self.cfg.source_voltage_v
            self.current[ids] = 0.0
            self.energy_j[ids] = 0.0
            self._initialized[ids] = False

        def _clip_effort(self, effort):
            scale = torch.clamp(self._voltage / self.cfg.nominal_voltage_v, min=0.0, max=1.0)
            stall = self._saturation_effort * scale
            base_speed = torch.minimum(self._electrical_velocity_limit, torch.as_tensor(self.cfg.no_load_speed_rad_s, device=self._device))
            speed = torch.clamp(base_speed * scale, min=1e-9)
            speed_at_limit = speed * (1.0 + self._electrical_effort_limit / torch.clamp(stall, min=1e-9))
            vel = torch.clamp(self._joint_vel, min=-speed_at_limit, max=speed_at_limit)
            top = torch.minimum(stall * (1.0 - vel / speed), self._electrical_effort_limit)
            bottom = torch.maximum(stall * (-1.0 - vel / speed), -self._electrical_effort_limit)
            lower = torch.minimum(bottom, top)
            upper = torch.maximum(bottom, top)
            return torch.clamp(effort, min=lower, max=upper)

        def compute(self, control_action: ArticulationActions, joint_pos, joint_vel):
            if any(value is None for value in (control_action.joint_positions, control_action.joint_velocities, control_action.joint_efforts)):
                raise ValueError("ElectromechanicalDCMotor requires position, velocity, and effort action tensors")
            fresh = ~self._initialized
            if torch.any(fresh):
                self._delay[:, fresh, :, 0] = control_action.joint_positions[fresh]
                self._delay[:, fresh, :, 1] = control_action.joint_velocities[fresh]
                self._delay[:, fresh, :, 2] = control_action.joint_efforts[fresh]
                self._initialized[fresh] = True
            self._delay[:-1] = self._delay[1:].clone()
            self._delay[-1, :, :, 0] = control_action.joint_positions
            self._delay[-1, :, :, 1] = control_action.joint_velocities
            self._delay[-1, :, :, 2] = control_action.joint_efforts
            control_action.joint_positions = self._delay[0, :, :, 0]
            control_action.joint_velocities = self._delay[0, :, :, 1]
            control_action.joint_efforts = self._delay[0, :, :, 2]
            result = super().compute(control_action, joint_pos, joint_vel)
            self.current = self.cfg.idle_current_a + self.cfg.stall_current_a * torch.abs(self.applied_effort) / self._saturation_effort
            self.current = torch.clamp(self.current, max=self.cfg.idle_current_a + self.cfg.stall_current_a)
            total_current = self.current.sum(dim=1, keepdim=True) + self.cfg.logic_power_w / torch.clamp(self._voltage, min=1e-6)
            self.total_current = total_current
            v_inf = self.cfg.source_voltage_v - self.cfg.source_resistance_ohm * total_current
            alpha = torch.exp(torch.tensor(-self.cfg.dt_s / (self.cfg.source_resistance_ohm * self.cfg.bulk_capacitance_f), device=self._device))
            self._voltage = v_inf + (self._voltage - v_inf) * alpha
            self.energy_j += self._voltage * total_current * self.cfg.dt_s
            return result

        def telemetry(self, env_index=0):
            return dict(bus_voltage_v=float(self._voltage[env_index,0]),
                estimated_current_a=float(self.total_current[env_index,0]),
                estimated_joint_current_a=self.current[env_index].tolist(),
                computed_effort_nm=self.computed_effort[env_index].tolist(),
                applied_effort_nm=self.applied_effort[env_index].tolist(),
                estimated_energy_j=float(self.energy_j[env_index,0]),
                delay_s=self.cfg.delay_steps*self.cfg.dt_s)
else:
    ElectromechanicalDCMotorCfg = None
    ElectromechanicalDCMotor = None

if ElectromechanicalDCMotor is not None:  # pragma: no cover
    ElectromechanicalDCMotorCfg.class_type = ElectromechanicalDCMotor
