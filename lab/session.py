"""RLB1 simulation adapter using the embedded decoder and ActuatorGuard.

The guard is shared; effort generation remains the existing ideal servo model.
No serial port, hardware calibration, clock synchronization or motor is implied.
"""

import ctypes as ct
import secrets
import numpy as np

from .control import MAX, Array, State, Supervisor
from .hardware import hardware_map
from .io import SimRobot
from .transport import encode_command


class JointMap(ct.Structure):
    _fields_ = [
        ("id", ct.c_uint8),
        ("bus", ct.c_uint8),
        ("mode", ct.c_uint8),
        ("direction", ct.c_int8),
        ("zero", ct.c_uint16),
        ("lower", ct.c_double),
        ("upper", ct.c_double),
        ("speed", ct.c_double),
    ]


class Guard(ct.Structure):
    _fields_ = [
        ("guard", State),
        ("session", ct.c_uint64),
        ("last_sequence", ct.c_uint64),
        ("sequence_seen", ct.c_uint8),
        ("armed", ct.c_uint8),
        ("calibrated", ct.c_uint8),
        ("sent", Array),
    ]


class WireCommand(ct.Structure):
    _fields_ = [
        ("type", ct.c_uint8),
        ("count", ct.c_uint8),
        ("sequence", ct.c_uint64),
        ("session", ct.c_uint64),
        ("value", ct.c_float * MAX),
    ]


class Receiver(ct.Structure):
    _fields_ = [
        ("data", ct.c_uint8 * (28 + 4 * MAX)),
        ("used", ct.c_size_t),
        ("expected", ct.c_size_t),
    ]


class SessionSupervisor(Supervisor):
    def __init__(
        self,
        robot,
        library,
        *,
        session,
        simulated_calibration=False,
        wheel_velocity_gain=0.2,
    ):
        if type(session) is not int or not 0 < session < 2**64:
            raise ValueError("session must be a nonzero uint64")
        super().__init__(
            robot,
            library,
            velocity_wheels=True,
            wheel_velocity_gain=wheel_velocity_gain,
        )
        self.mapping = hardware_map(robot)
        self.map = (JointMap * self.n)(
            *(
                JointMap(**{k: row[k] for k, _ in JointMap._fields_})
                for row in self.mapping["joints"]
            )
        )
        self.guard = Guard()
        self.receiver = Receiver()
        self.arm_switch, self.stop_ok = False, True
        self.decoded, self.rejected = 0, 0
        self.lib.actuator_init.argtypes = [
            ct.POINTER(Guard),
            ct.POINTER(JointMap),
            ct.c_uint,
            ct.c_uint,
            ct.c_double,
            ct.c_uint64,
            ct.c_int,
        ]
        self.lib.actuator_command.argtypes = [
            ct.POINTER(Guard),
            ct.POINTER(JointMap),
            ct.POINTER(WireCommand),
            ct.c_uint64,
            ct.c_int,
            ct.c_int,
        ]
        self.lib.actuator_check.argtypes = [
            ct.POINTER(Guard),
            ct.c_uint64,
            ct.c_double,
            ct.c_int,
            ct.c_int,
        ]
        self.lib.wire_push.argtypes = [
            ct.POINTER(Receiver),
            ct.c_uint8,
            ct.POINTER(WireCommand),
        ]
        c = robot.config
        if not self.lib.actuator_init(
            ct.byref(self.guard),
            self.map,
            self.n,
            c.control.timeout_ms,
            c.power.minimum_bus_v,
            session,
            simulated_calibration,
        ):
            raise ValueError("embedded guard rejected actuator map")
        # Keep the embedded guard's bounds/lifecycle. Supply only the simulation
        # effort law coefficients, which actuator_rows on hardware does not use.
        self.guard.guard.kp, self.guard.guard.kd = self.state.kp, self.state.kd
        self.guard.guard.effort = self.state.effort
        self.state = self.guard.guard

    @property
    def session(self):
        return self.guard.session

    def receive(self, data, now_ms):
        if (
            not isinstance(data, bytes)
            or type(now_ms) is not int
            or not 0 <= now_ms < 2**64
        ):
            raise ValueError("bytes and uint64 receive clock required")
        results = []
        for byte in data:
            command = WireCommand()
            if self.lib.wire_push(ct.byref(self.receiver), byte, ct.byref(command)):
                accepted = bool(
                    self.lib.actuator_command(
                        ct.byref(self.guard),
                        self.map,
                        ct.byref(command),
                        now_ms,
                        self.arm_switch,
                        self.stop_ok,
                    )
                )
                self.decoded += 1
                self.rejected += not accepted
                results.append(accepted)
        return results

    def send(self, kind, sequence, now_ms, targets=()):
        return self.receive(
            encode_command(kind, sequence, self.session, targets), now_ms
        ) == [True]

    def command(self, seq, received_ms, targets):
        return self.send("command", seq, received_ms, targets)

    def tick(self, now_ms, voltage, q, dq, estop=False):
        if type(now_ms) is not int or not 0 <= now_ms < 2**64:
            raise ValueError("uint64 clock required")
        self.lib.actuator_check(
            ct.byref(self.guard),
            now_ms,
            voltage,
            self.arm_switch,
            self.stop_ok and not estop,
        )
        # supervisor_check alone cannot express the maintained-arm gate.
        if not self.guard.armed:
            return self.state.status, np.zeros(self.n)
        status, effort = super().tick(now_ms, voltage, q, dq, estop)
        if status in (2, 3, 4):
            self.guard.armed = 0
        return status, effort


class SessionSimRobot(SimRobot):
    """Explicit ARM/STOP plus mixed commands; reset creates a new boot session."""

    def __init__(
        self,
        robot,
        model_path,
        library,
        *,
        simulated_calibration=False,
        wheel_velocity_gain=0.2,
    ):
        self.simulated_calibration = simulated_calibration
        self.boot_session = secrets.randbits(64) or 1
        super().__init__(
            robot,
            model_path,
            library,
            velocity_wheels=True,
            wheel_velocity_gain=wheel_velocity_gain,
        )

    def reset(self):
        super().reset()
        self.boot_session = self.boot_session % (2**64 - 1) + 1
        self.supervisor = SessionSupervisor(
            self.robot,
            self.library,
            session=self.boot_session,
            simulated_calibration=self.simulated_calibration,
            wheel_velocity_gain=self.wheel_velocity_gain,
        )
        return self.observe()

    def arm(self, sequence):
        return self.supervisor.send("arm", sequence, round(self.data.time * 1000))

    def stop(self, sequence):
        return self.supervisor.send("stop", sequence, round(self.data.time * 1000))

    def receive(self, data):
        return self.supervisor.receive(data, round(self.data.time * 1000))
