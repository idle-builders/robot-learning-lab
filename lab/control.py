"""Same native C supervisor runs in the physics loop and host fault tests."""

import ctypes as ct
from pathlib import Path
import subprocess
import sys
import numpy as np

MAX = 32
Array = ct.c_double * MAX


class State(ct.Structure):
    _fields_ = (
        [(n, ct.c_uint32) for n in ("n", "timeout_ms", "status", "has_command")]
        + [(n, ct.c_uint64) for n in ("sequence", "received_ms")]
        + [(n, ct.c_double) for n in ("minimum_voltage", "kp", "kd")]
        + [(n, Array) for n in ("lower", "upper", "effort", "speed", "target")]
    )


def build_supervisor(output):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    dest = path / ("supervisor.dylib" if sys.platform == "darwin" else "supervisor.so")
    command = [
        "cc",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-O2",
        "-fPIC",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "firmware/supervisor.c",
        "firmware/actuator.c",
        "firmware/wire.c",
        "firmware/imu.c",
        "-o",
        str(dest),
    ]
    if sys.platform != "darwin":
        command.append("-lm")
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (path / "firmware-build.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    return dest.resolve()


class Supervisor:
    def __init__(self, robot, library, velocity_wheels=False, wheel_velocity_gain=0.2):
        self.n = len(robot.joints)
        self.state = State()
        self.lib = ct.CDLL(str(library))
        ptr = ct.POINTER(ct.c_double)
        self.lib.supervisor_init.argtypes = [
            ct.POINTER(State),
            ct.c_uint32,
            ct.c_uint32,
            ct.c_double,
            ct.c_double,
            ct.c_double,
            ptr,
            ptr,
            ptr,
            ptr,
        ]
        self.lib.supervisor_command.argtypes = [
            ct.POINTER(State),
            ct.c_uint64,
            ct.c_uint64,
            ptr,
        ]
        self.lib.supervisor_tick.argtypes = [
            ct.POINTER(State),
            ct.c_uint64,
            ct.c_double,
            ct.c_int,
            ptr,
            ptr,
            ptr,
        ]
        self.mixed_tick = getattr(self.lib, "supervisor_tick_mixed", None)
        if velocity_wheels and self.mixed_tick is None:
            raise ValueError("rebuild the native supervisor for wheel-speed commands")
        if self.mixed_tick is not None:
            self.mixed_tick.argtypes = self.lib.supervisor_tick.argtypes[:-1] + [
                ct.POINTER(ct.c_uint32),
                ct.c_double,
                ptr,
            ]
        self.velocity_wheels = velocity_wheels
        self.wheel_velocity_gain = wheel_velocity_gain
        self.modes = (ct.c_uint32 * MAX)(
            *[int(velocity_wheels and j.continuous) for j in robot.joints]
        )
        lo = Array(*[(-1e9 if j.continuous else j.limits[0]) for j in robot.joints])
        hi = Array(*[(1e9 if j.continuous else j.limits[1]) for j in robot.joints])
        effort = Array(
            *[robot.config.actuators[j.family].effort_limit_nm for j in robot.joints]
        )
        speed = Array(
            *[
                robot.config.actuators[j.family].velocity_limit_rad_s
                for j in robot.joints
            ]
        )
        if velocity_wheels:
            for i, j in enumerate(robot.joints):
                if j.continuous:
                    lo[i], hi[i] = -speed[i], speed[i]
        c = robot.config
        if not self.lib.supervisor_init(
            ct.byref(self.state),
            self.n,
            c.control.timeout_ms,
            c.power.minimum_bus_v,
            c.control.kp,
            c.control.kd,
            lo,
            hi,
            effort,
            speed,
        ):
            raise ValueError("native supervisor rejected configuration")

    def command(self, seq, received_ms, targets):
        if len(targets) != self.n or seq < 0 or received_ms < 0:
            raise ValueError("invalid command shape or clock")
        return bool(
            self.lib.supervisor_command(
                ct.byref(self.state), seq, received_ms, Array(*targets)
            )
        )

    def tick(self, now_ms, voltage, q, dq, estop=False):
        if len(q) != self.n or len(dq) != self.n or now_ms < 0:
            raise ValueError("invalid state shape or clock")
        out = Array()
        arguments = (
            ct.byref(self.state),
            now_ms,
            voltage,
            estop,
            Array(*q),
            Array(*dq),
        )
        status = (
            self.mixed_tick(*arguments, self.modes, self.wheel_velocity_gain, out)
            if self.velocity_wheels
            else self.lib.supervisor_tick(*arguments, out)
        )
        return status, np.array(list(out)[: self.n])
