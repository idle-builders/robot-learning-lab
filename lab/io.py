"""Transport-independent joint IO contract and a deterministic simulator adapter."""

from dataclasses import dataclass
from typing import Protocol
import mujoco
import numpy as np
from .control import Supervisor


@dataclass(frozen=True)
class JointCommand:
    sequence: int
    targets_rad: tuple[float, ...]


@dataclass(frozen=True)
class MixedJointCommand:
    """Joint order: continuous wheel targets in rad/s, other targets in rad."""

    sequence: int
    targets: tuple[float, ...]


@dataclass(frozen=True)
class IMUSample:
    time_s: float
    gyro_rad_s: tuple[float, float, float]
    specific_force_m_s2: tuple[float, float, float]


@dataclass(frozen=True)
class JointState:
    time_s: float
    names: tuple[str, ...]
    position_rad: tuple[float, ...]
    velocity_rad_s: tuple[float, ...]
    effort_nm: tuple[float, ...]
    supervisor_status: int


class RobotIO(Protocol):
    def reset(self) -> JointState: ...
    def command(self, command: JointCommand | MixedJointCommand) -> bool: ...
    def advance(self, physics_steps: int) -> JointState: ...
    def observe(self) -> JointState: ...


class SimRobot:
    """No wall-clock sleeping or network. Free and supported models share this API."""

    def __init__(
        self, robot, model_path, library, velocity_wheels=False, wheel_velocity_gain=0.2
    ):
        self.robot = robot
        self.library = library
        self.velocity_wheels = velocity_wheels
        self.wheel_velocity_gain = wheel_velocity_gain
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.names = tuple(j.name for j in robot.joints)
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in self.names])
        self.vadr = np.array([self.model.joint(n).dofadr[0] for n in self.names])
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.command_receipts = []
        self.active_command = None
        self.scene_data = None
        self.supervisor = Supervisor(
            self.robot, self.library, self.velocity_wheels, self.wheel_velocity_gain
        )
        mujoco.mj_forward(self.model, self.data)
        self.imu_time_s = 0.0
        return self.observe()

    def command(self, command):
        expected = MixedJointCommand if self.velocity_wheels else JointCommand
        if not isinstance(command, expected):
            raise TypeError(f"this simulator requires {expected.__name__}")
        accepted = self.supervisor.command(
            command.sequence,
            round(self.data.time * 1000),
            command.targets if self.velocity_wheels else command.targets_rad,
        )
        receipt = {
            "sequence": command.sequence,
            "received_time_s": float(self.data.time),
            "accepted": bool(accepted),
            "targets": list(
                command.targets if self.velocity_wheels else command.targets_rad
            ),
            "first_control_tick_s": None,
        }
        self.command_receipts.append(receipt)
        if accepted:
            self.active_command = receipt
        return accepted

    def observe_scene(self):
        """Fresh post-integration body poses without changing controller/sensor data."""
        if self.scene_data is None:
            self.scene_data = mujoco.MjData(self.model)
        snapshot = self.scene_data
        snapshot.qpos[:] = self.data.qpos
        snapshot.mocap_pos[:] = self.data.mocap_pos
        snapshot.mocap_quat[:] = self.data.mocap_quat
        mujoco.mj_kinematics(self.model, snapshot)
        return {
            "time_s": float(self.data.time),
            "source": "privileged_simulator_truth",
            "bodies": {
                self.model.body(i).name: {
                    "position_m": snapshot.xpos[i].tolist(),
                    "quaternion_xyzw": [
                        *snapshot.xquat[i, 1:].tolist(),
                        float(snapshot.xquat[i, 0]),
                    ],
                }
                for i in range(self.model.nbody)
                if self.model.body(i).name
            },
        }

    def observe_imu(self):
        return IMUSample(
            self.imu_time_s,
            tuple(self.data.sensor("imu_gyro").data),
            tuple(self.data.sensor("imu_accel").data),
        )

    def advance(self, physics_steps):
        if not isinstance(physics_steps, int) or physics_steps < 1:
            raise ValueError("positive integer steps required")
        c = self.robot.config
        for _ in range(physics_steps):
            current = c.power.electronics_current_a + sum(
                c.actuators[j.family].idle_current_a
                + abs(u) / c.actuators[j.family].effective_nm_per_a
                for j, u in zip(self.robot.joints, self.data.ctrl)
            )
            voltage = (
                c.power.battery_nominal_v - current * c.power.source_resistance_ohm
            )
            status, u = self.supervisor.tick(
                round(self.data.time * 1000),
                voltage,
                self.data.qpos[self.qadr],
                self.data.qvel[self.vadr],
            )
            if (
                status == 1
                and self.active_command is not None
                and self.active_command["first_control_tick_s"] is None
            ):
                self.active_command["first_control_tick_s"] = float(self.data.time)
            self.data.ctrl[:] = u
            # mj_step leaves sensors at the pre-integration sample time.
            self.imu_time_s = float(self.data.time)
            mujoco.mj_step(self.model, self.data)
        return self.observe()

    def observe(self):
        return JointState(
            float(self.data.time),
            self.names,
            tuple(self.data.qpos[self.qadr]),
            tuple(self.data.qvel[self.vadr]),
            tuple(self.data.ctrl),
            self.supervisor.state.status,
        )
