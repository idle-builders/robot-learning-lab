"""Sensor-based sagittal balance experiments for the articulated wheeled concepts."""

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import sys
import traceback
from typing import Literal
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import yaml
from pydantic import Field, model_validator

from .artifact_store import snapshot_file, snapshot_tree
from .config import RobotConfig, Strict, load_config
from .robot import make_robot
from .export import export_all
from .control import build_supervisor
from .io import SimRobot, MixedJointCommand, IMUSample
from .imu_sim import DriverIMU, SensorModel


def _apply_mass_model(robot, mass_model):
    if mass_model is not None:
        from .mass_model import apply_mass_model

        apply_mass_model(robot, Path(mass_model))


class BalanceSettings(Strict):
    pitch_gain_m_s: float = Field(default=4, ge=0)
    pitch_rate_gain_m: float = Field(default=0.2, ge=0)
    position_gain_s_inv: float = Field(default=1, ge=0)
    velocity_gain: float = Field(default=2.5, ge=0)
    complementary_tau_s: float = Field(default=0.3, gt=0)
    wheel_velocity_gain_nm_s: float = Field(default=0.2, gt=0)
    pitch_trim: Literal["zero", "assembly_mass", "observed_assembly_mass"] = "zero"
    pitch_initialization: Literal["zero", "stationary_imu"] = "zero"
    arm_rest_targets_rad: dict[str, float] = Field(default_factory=dict)


class BalanceCase(Strict):
    name: str = Field(pattern=r"^[a-z0-9-]+$")
    duration_s: float = Field(default=15, gt=1, le=60)
    initial_pitch_rad: float = Field(default=0.03, ge=-0.2, le=0.2)
    floor_friction: float = Field(default=0.8, gt=0, le=2)
    mass_scale: float = Field(default=1, gt=0, le=2)
    wheel_gain_scale: float = Field(default=1, gt=0, le=5)
    command_delay_ms: int = Field(default=0, ge=0, le=500)
    gyro_bias_rad_s: float = Field(default=0, ge=-0.1, le=0.1)
    gyro_noise_std_rad_s: float = Field(default=0, ge=0, le=0.1)
    accel_noise_std_m_s2: float = Field(default=0, ge=0, le=1)
    push_force_n: float = Field(default=0, ge=-5, le=5)
    push_start_s: float = Field(default=3, ge=0)
    push_duration_s: float = Field(default=0.1, gt=0, le=2)
    waypoints: list[tuple[float, float]] = Field(default_factory=lambda: [(0, 0)])
    controller_enabled: bool = True
    seed: int = Field(default=0, ge=0)
    sensor: SensorModel = Field(default_factory=SensorModel)
    tilt_limit_rad: float = Field(default=0.25, gt=0, le=0.6)
    final_position_limit_m: float = Field(default=0.025, gt=0)
    final_pitch_limit_rad: float = Field(default=0.025, gt=0)
    final_speed_limit_m_s: float = Field(default=0.02, gt=0)
    path_error_limit_m: float = Field(default=0.04, gt=0)

    @model_validator(mode="after")
    def validate_timeline(self):
        if not self.waypoints or self.waypoints[0] != (0, 0):
            raise ValueError("waypoints must start at time=0, position=0")
        if any(b[0] <= a[0] for a, b in zip(self.waypoints, self.waypoints[1:])):
            raise ValueError("waypoint times must increase")
        if self.waypoints[-1][0] > self.duration_s - 2:
            raise ValueError("reserve two seconds of final hold")
        if (
            self.push_force_n
            and self.push_start_s + self.push_duration_s > self.duration_s
        ):
            raise ValueError("push must occur within the episode")
        return self


@dataclass
class BalanceObservation:
    pitch_rad: float
    pitch_rate_rad_s: float
    position_m: float
    velocity_m_s: float


class SagittalObserver:
    """Planar complementary IMU filter + no-slip, leg-compensated wheel odometry.

    Never reads MuJoCo's root pose/velocity. Assumes roll/yaw remain small,
    encoder zeros are known, and gyro/accelerometer axes agree with the torso.
    """

    def __init__(self, robot, settings):
        self.robot, self.settings = robot, settings
        names = [j.name for j in robot.joints]
        self.wheels = [i for i, j in enumerate(robot.joints) if j.continuous]
        self.hips = [names.index(s + "_thigh_joint") for s in ("left", "right")]
        self.knees = [names.index(s + "_shin_joint") for s in ("left", "right")]
        self.pitch = 0.0
        self.last_time = None

    def update(self, state, imu):
        t = imu.time_s
        if (
            not math.isfinite(t)
            or not math.isfinite(state.time_s)
            or not -1e-9 <= state.time_s - t <= 0.1
        ):
            raise ValueError("IMU sample time is invalid, future or stale")
        dt = 0 if self.last_time is None else t - self.last_time
        if dt < 0 or dt > 0.1:
            raise ValueError("IMU sample time regressed or became stale")
        gyro = np.asarray(imu.gyro_rad_s)
        acc = np.asarray(imu.specific_force_m_s2)
        q, dq = np.asarray(state.position_rad), np.asarray(state.velocity_rad_s)
        if not all(np.isfinite(x).all() for x in (gyro, acc, q, dq)):
            raise ValueError("nonfinite observer input")
        if (
            self.last_time is None
            and self.settings.pitch_initialization == "stationary_imu"
        ):
            # Explicit held-start assumption, using only the first sensor sample.
            # Norm/gyro checks cannot distinguish gravity from every sustained
            # linear acceleration; they do not prove physical stationarity.
            if not 7 < np.linalg.norm(acc) < 12 or np.linalg.norm(gyro) > 0.05:
                raise ValueError(
                    "stationary IMU initialization requires gravity and low rotation"
                )
            self.pitch = math.atan2(-acc[0], acc[2])
        self.pitch += float(gyro[1]) * dt
        # Reject freefall/large acceleration. This is not a full 3D estimator.
        if 7 < np.linalg.norm(acc) < 12:
            alpha = 1 - math.exp(-dt / self.settings.complementary_tau_s)
            self.pitch += alpha * (math.atan2(-acc[0], acc[2]) - self.pitch)
        self.last_time = t
        h, k, w = [float(np.mean(q[i])) for i in (self.hips, self.knees, self.wheels)]
        hd, kd, wd = [
            float(np.mean(dq[i])) for i in (self.hips, self.knees, self.wheels)
        ]
        p, pd = self.pitch, float(gyro[1])
        d = self.robot.config.dimensions
        x = (
            d.wheel_radius_m * (w + h + k + p)
            + d.hip_drop_m * math.sin(p)
            + d.thigh_m * math.sin(p + h)
            + d.shin_m * math.sin(p + h + k)
        )
        v = (
            d.wheel_radius_m * (wd + hd + kd + pd)
            + d.hip_drop_m * math.cos(p) * pd
            + d.thigh_m * math.cos(p + h) * (pd + hd)
            + d.shin_m * math.cos(p + h + k) * (pd + hd + kd)
        )
        return BalanceObservation(p, pd, x, v)


def validate_arm_rest(robot, targets):
    """Validate fixed arm commands; leg changes need a separate ground fixture."""
    if not targets:
        return {}
    arms = {j.name for j in robot.joints if j.family == "arm"}
    if set(targets) - arms:
        raise ValueError("balance rest targets must name arm-family joints only")
    robot.forward_kinematics(targets)  # Shared finite-value and joint-limit checks.
    return dict(targets)


def neutral_mass_trim(robot, arm_rest_targets_rad=None):
    """Static sagittal equilibrium for the controller's declared joint posture.

    Uses declared link masses/COMs and kinematics, never simulator truth.
    This fixed trim does not compensate moving arms, a payload or wheel slip.
    """
    if robot.mass_model is None:
        raise ValueError("assembly-mass trim requires an explicit mass model")
    rest = validate_arm_rest(robot, arm_rest_targets_rad or {})
    frames = robot.forward_kinematics(rest) if rest else robot.forward_kinematics()
    return dict(
        **_mass_trim_from_frames(robot, frames, 1e-9),
        mode="assembly_mass",
        posture="declared arm rest targets; all other joint positions zero",
        arm_rest_targets_rad=rest,
        scope="assumed static mass distribution; no live COM/payload compensation",
    )


def observed_mass_trim(robot, state):
    """Quasi-static lean from current encoders and assumed mass, without root truth."""
    if robot.mass_model is None:
        raise ValueError("observed assembly-mass trim requires an explicit mass model")
    names = tuple(j.name for j in robot.joints)
    if (
        state is None
        or tuple(state.names) != names
        or len(state.position_rad) != len(names)
    ):
        raise ValueError(
            "observed trim requires a complete joint state in generated order"
        )
    if not math.isfinite(state.time_s):
        raise ValueError("observed trim requires finite acquisition time")
    positions = dict(zip(names, state.position_rad))
    # Preserve measured excursions; command limits remain enforced separately.
    frames = robot.forward_kinematics(positions, enforce_limits=False)
    return dict(
        **_mass_trim_from_frames(robot, frames, 0.002),
        mode="observed_assembly_mass",
        posture="observed encoder positions in generated order",
        joint_positions_rad=positions,
        observation_time_s=state.time_s,
        scope="quasi-static assumed mass; no payload identification, slip or inertial-force compensation",
    )


def _mass_trim_from_frames(robot, frames, axle_tolerance_m):
    wheels = np.array([frames[s + "_wheel"][:3, 3] for s in ("left", "right")])
    axle_mismatch = float(np.linalg.norm(wheels[0, [0, 2]] - wheels[1, [0, 2]]))
    if axle_mismatch > axle_tolerance_m:
        raise ValueError("static trim requires aligned wheel axes")
    weighted = np.zeros(3)
    total = 0.0
    for link in robot.links:
        if link.mass_properties is None:
            raise ValueError("static trim requires every link COM")
        local = np.asarray(link.mass_properties["center_m"], dtype=float)
        if (
            local.shape != (3,)
            or not np.isfinite(local).all()
            or not 0 < link.mass < float("inf")
        ):
            raise ValueError("invalid mass/COM in static trim")
        frame = frames[link.name]
        weighted += link.mass * (frame[:3, :3] @ local + frame[:3, 3])
        total += link.mass
    if not 0 < total < float("inf"):
        raise ValueError("static trim requires finite positive total mass")
    center = weighted / total
    axis = wheels.mean(axis=0)
    offset = center - axis
    if not np.isfinite(offset).all() or offset[2] <= 0:
        raise ValueError("static trim requires COM above the wheel axis")
    return dict(
        pitch_rad=-math.atan2(offset[0], offset[2]),
        center_world_m=center.tolist(),
        wheel_axis_world_m=axis.tolist(),
        total_mass_kg=total,
        mass_model=robot.mass_model,
        frame_basis="shared upright-root FK frame, not observed world pose",
        axle_mismatch_m=axle_mismatch,
        axle_tolerance_m=axle_tolerance_m,
    )


class WheelBalancer:
    def __init__(self, robot, settings):
        if robot.config.morphology != "wheeled":
            raise ValueError("wheel balancer requires articulated wheeled legs")
        self.robot, self.settings = robot, settings
        self.arm_rest_targets_rad = validate_arm_rest(
            robot, settings.arm_rest_targets_rad
        )
        self.trim = (
            neutral_mass_trim(robot, self.arm_rest_targets_rad)
            if settings.pitch_trim != "zero"
            else dict(mode="zero", pitch_rad=0.0)
        )

    def command(
        self,
        sequence,
        observation,
        reference_position=0.0,
        reference_velocity=0.0,
        *,
        joint_state=None,
    ):
        s, o = self.settings, observation
        if s.pitch_trim == "observed_assembly_mass":
            self.trim = observed_mass_trim(self.robot, joint_state)
        speed = (
            s.pitch_gain_m_s * (o.pitch_rad - self.trim["pitch_rad"])
            + s.pitch_rate_gain_m * o.pitch_rate_rad_s
            + s.position_gain_s_inv * (o.position_m - reference_position)
            + s.velocity_gain * (o.velocity_m_s - reference_velocity)
            + reference_velocity
        )
        if not math.isfinite(speed):
            raise ValueError("nonfinite balance demand")
        values = []
        for j in self.robot.joints:
            a = self.robot.config.actuators[j.family]
            values.append(
                float(
                    np.clip(
                        speed / self.robot.config.dimensions.wheel_radius_m,
                        -a.velocity_limit_rad_s,
                        a.velocity_limit_rad_s,
                    )
                )
                if j.continuous
                else self.arm_rest_targets_rad.get(j.name, 0.0)
            )
        return MixedJointCommand(sequence, tuple(values))


def reference(case, time_s):
    for a, b in zip(case.waypoints, case.waypoints[1:]):
        if time_s < b[0]:
            velocity = (b[1] - a[1]) / (b[0] - a[0])
            return a[1] + (time_s - a[0]) * velocity, velocity
    return case.waypoints[-1][1], 0.0


def scale_inertials(xml, scale):
    for inertial in xml.findall(".//inertial"):
        inertial.set("mass", str(float(inertial.attrib["mass"]) * scale))
        inertia_key = (
            "fullinertia" if "fullinertia" in inertial.attrib else "diaginertia"
        )
        inertial.set(
            inertia_key,
            " ".join(
                str(float(v) * scale) for v in inertial.attrib[inertia_key].split()
            ),
        )


def evaluate(robot, model_path, library, settings, case, output, render=False):
    controller = WheelBalancer(robot, settings)
    output.mkdir(parents=True, exist_ok=False)
    (output / "case.json").write_text(case.model_dump_json(indent=2))
    xml = ET.parse(model_path)
    for mesh in xml.findall("./asset/mesh"):
        mesh.set("file", str((model_path.parent / mesh.attrib["file"]).absolute()))
    floor = xml.find("./worldbody/geom[@name='floor']")
    floor.set("friction", f"{case.floor_friction} .01 .001")
    floor.set("priority", "1")
    scale_inertials(xml, case.mass_scale)
    xml.write(output / "model.xml")
    driver_mode = case.sensor.mode == "lsm6dso"
    gain = settings.wheel_velocity_gain_nm_s * case.wheel_gain_scale
    if driver_mode:
        from .session import SessionSimRobot

        sim = SessionSimRobot(
            robot,
            output / "model.xml",
            library,
            simulated_calibration=True,
            wheel_velocity_gain=gain,
        )
        sim.supervisor.arm_switch = True
    else:
        sim = SimRobot(robot, output / "model.xml", library, True, gain)
    a = case.initial_pitch_rad
    sim.data.qpos[3:7] = [math.cos(a / 2), 0, math.sin(a / 2), 0]
    for name, target in controller.arm_rest_targets_rad.items():
        sim.data.qpos[sim.model.joint(name).qposadr[0]] = target
    mujoco.mj_forward(sim.model, sim.data)
    (output / "initial-state.json").write_text(
        json.dumps(
            dict(
                scope="held startup pose; primitive collision shapes only",
                arm_rest_targets_rad=controller.arm_rest_targets_rad,
                position_rad=list(sim.observe().position_rad),
                contacts=[
                    dict(
                        geom1=sim.model.geom(c.geom1).name,
                        geom2=sim.model.geom(c.geom2).name,
                        distance_m=float(c.dist),
                    )
                    for c in sim.data.contact
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    sensor = None
    if driver_mode:
        sensor = DriverIMU(
            library,
            case.sensor,
            seed=case.seed,
            gyro_bias=case.gyro_bias_rad_s,
            gyro_noise=case.gyro_noise_std_rad_s,
            accel_noise=case.accel_noise_std_m_s2,
            robot=robot,
        )
        # Sensor is held stationary during startup; release starts physics time 0.
        if not np.allclose(
            sim.model.site("imu").pos,
            sensor.config["sensor_frame"]["allocated_xyz_m"],
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError(
                "simulation IMU site differs from selected sensor allocation"
            )
        startup = sensor.warmup(
            sim.data.site("imu").xmat.reshape(3, 3).T @ np.array([0, 0, 9.80665])
        )
        if not startup["valid"]:
            result = dict(
                status="fail",
                case=case.name,
                duration_s=0,
                termination="imu_startup_invalid",
                sensor=sensor.summary(),
                motion_released=False,
                input=case.model_dump(),
                scope="held sensor startup; actuator guard never armed",
            )
            (output / "trajectory.jsonl").write_text("")
            (output / "imu-polls.jsonl").write_text(
                "".join(json.dumps(p) + "\n" for p in sensor.polls)
            )
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            return result, []
        if not sim.arm(0):
            raise ValueError("simulated calibration/arm failed")
    observer = SagittalObserver(robot, settings)
    rng = np.random.default_rng(case.seed)
    stride = robot.config.control.physics_hz // robot.config.control.command_hz
    trace, pending, frames = [], [], []
    torso_id = sim.model.body("torso").id
    wheel_indices = [i for i, j in enumerate(robot.joints) if j.continuous]
    termination = None
    sensor_fault = None
    for seq in range(round(case.duration_s * robot.config.control.command_hz)):
        state, imu = sim.observe(), sim.observe_imu()
        if sensor:
            health = sensor.feed(imu, state.time_s)
            if not health["valid"]:
                termination = (
                    "imu_fault"
                    if health["status"] == 4
                    else "imu_saturated"
                    if health["saturated"]
                    else "imu_stale"
                )
                # This is a simulation abort policy, not a safe physical fall strategy.
                if not sim.stop(seq + 1):
                    raise ValueError("sensor-fault STOP rejected")
                detected_time = state.time_s
                state = sim.advance(1)
                sensor_fault = {
                    **health,
                    "detected_time_s": detected_time,
                    "stop_applied_time_s": state.time_s,
                    "stop_status": state.supervisor_status,
                    "stop_effort_nm": list(state.effort_nm),
                }
                if trace:
                    trace[-1]["sensor_fault"] = sensor_fault
                break
            controller_imu = sensor.observation()
            observation = observer.update(state, controller_imu)
        else:
            g = np.array(imu.gyro_rad_s) + rng.normal(0, case.gyro_noise_std_rad_s, 3)
            g[1] += case.gyro_bias_rad_s
            acc = np.array(imu.specific_force_m_s2) + rng.normal(
                0, case.accel_noise_std_m_s2, 3
            )
            controller_imu = IMUSample(imu.time_s, tuple(g), tuple(acc))
            observation = observer.update(state, controller_imu)
        issued_time_s = state.time_s
        receipt_start = len(sim.command_receipts)
        xr, vr = reference(case, state.time_s)
        command = (
            controller.command(
                seq + int(driver_mode), observation, xr, vr, joint_state=state
            )
            if case.controller_enabled
            else MixedJointCommand(
                seq + int(driver_mode),
                tuple(
                    controller.arm_rest_targets_rad.get(j.name, 0.0)
                    for j in robot.joints
                ),
            )
        )
        pending.append((state.time_s + case.command_delay_ms / 1000, command))
        for _ in range(stride):
            while pending and pending[0][0] <= sim.data.time + 1e-10:
                _, ready = pending.pop(0)
                if not sim.command(ready):
                    termination = "command_rejected"
                    break
            if termination:
                break
            sim.data.xfrc_applied[torso_id, :] = 0
            if (
                case.push_start_s
                <= sim.data.time
                < case.push_start_s + case.push_duration_s
            ):
                sim.data.xfrc_applied[torso_id, 0] = case.push_force_n
            state = sim.advance(1)
            if sensor:
                sensor.feed(sim.observe_imu(), state.time_s)
        # Truth used only for independent evaluation and rendering, never control.
        matrix = np.zeros(9)
        mujoco.mju_quat2Mat(matrix, sim.data.qpos[3:7])
        matrix = matrix.reshape(3, 3)
        pitch = math.atan2(matrix[0, 2], matrix[2, 2])
        tilt = math.acos(float(np.clip(matrix[2, 2], -1, 1)))
        current = robot.config.power.electronics_current_a + sum(
            robot.config.actuators[j.family].idle_current_a
            + abs(u) / robot.config.actuators[j.family].effective_nm_per_a
            for j, u in zip(robot.joints, state.effort_nm)
        )
        row = dict(
            time_s=state.time_s,
            scene_state=sim.observe_scene(),
            controller_imu={
                **asdict(controller_imu),
                "timestamp_basis": "poll_start_proxy"
                if sensor
                else "sensor_evaluation",
            },
            control_timing={
                "issued_time_s": issued_time_s,
                "sequence": command.sequence,
                "estimator_update_time_s": issued_time_s,
                "receipts": [dict(r) for r in sim.command_receipts[receipt_start:]],
            },
            pitch_rad=pitch,
            tilt_rad=tilt,
            estimated_pitch_rad=observation.pitch_rad,
            estimated_position_m=observation.position_m,
            estimated_velocity_m_s=observation.velocity_m_s,
            pitch_reference_rad=controller.trim["pitch_rad"],
            trim_observation_time_s=controller.trim.get("observation_time_s"),
            trim_axle_mismatch_m=controller.trim.get("axle_mismatch_m"),
            truth_x_m=float(sim.data.qpos[0]),
            truth_velocity_m_s=float(sim.data.qvel[0]),
            reference_x_m=xr,
            reference_velocity_m_s=vr,
            base_height_m=float(sim.data.qpos[2]),
            targets=list(command.targets),
            position_rad=list(state.position_rad),
            velocity_rad_s=list(state.velocity_rad_s),
            effort_nm=list(state.effort_nm),
            supervisor_status=state.supervisor_status,
            estimated_bus_current_a=float(current),
        )
        trace.append(row)
        if sensor:
            row["imu"] = sensor.health()
        if render and seq % 5 == 0:
            frames.append(sim.data.qpos.copy())
        if not np.isfinite(sim.data.qpos).all():
            termination = "nonfinite_state"
        elif state.supervisor_status in (2, 3, 4):
            termination = "supervisor_stop"
        elif tilt > case.tilt_limit_rad:
            termination = "tilt_limit"
        elif sim.data.qpos[2] < 0.7 * robot.base_height:
            termination = "height_limit"
        if termination:
            break
    if not trace:
        result = dict(
            status="fail",
            case=case.name,
            duration_s=state.time_s,
            termination=termination,
            sensor_fault=sensor_fault,
            sensor=sensor.summary() if sensor else None,
            input=case.model_dump(),
            scope="sensor invalid before first motion command",
        )
        (output / "trajectory.jsonl").write_text("")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result, []
    final = trace[-1]
    hold = [s for s in trace if s["time_s"] > case.duration_s - 2]
    final_error = abs(final["truth_x_m"] - case.waypoints[-1][1])
    path_error = max(abs(s["truth_x_m"] - s["reference_x_m"]) for s in trace)
    passed = (
        termination is None
        and len(trace) == round(case.duration_s * robot.config.control.command_hz)
        and bool(hold)
    )
    passed = (
        passed
        and final_error <= case.final_position_limit_m
        and max(abs(s["pitch_rad"]) for s in hold) <= case.final_pitch_limit_rad
    )
    passed = (
        passed
        and max(abs(s["truth_velocity_m_s"]) for s in hold)
        <= case.final_speed_limit_m_s
    )
    passed = passed and (
        len(case.waypoints) == 1 or path_error <= case.path_error_limit_m
    )
    result = dict(
        status="pass" if passed else "fail",
        case=case.name,
        duration_s=final["time_s"],
        termination=termination,
        final_position_error_m=final_error,
        max_path_error_m=path_error,
        max_tilt_rad=max(s["tilt_rad"] for s in trace),
        max_position_excursion_m=max(abs(s["truth_x_m"]) for s in trace),
        final_pitch_rad=final["pitch_rad"],
        max_hold_speed_m_s=max(
            (abs(s["truth_velocity_m_s"]) for s in hold), default=None
        ),
        max_estimator_pitch_error_rad=max(
            abs(s["pitch_rad"] - s["estimated_pitch_rad"]) for s in trace
        ),
        peak_wheel_speed_rad_s=max(
            abs(s["velocity_rad_s"][i]) for s in trace for i in wheel_indices
        ),
        peak_estimated_current_a=max(s["estimated_bus_current_a"] for s in trace),
        input=case.model_dump(),
        joint_order=list(sim.names),
        scope="free-base sagittal controller using simulated IMU and encoder measurements",
        pitch_trim=controller.trim,
    )
    if sensor:
        result["sensor"] = sensor.summary()
        result["sensor_fault"] = sensor_fault
        result["command_path"] = {
            "type": "RLB1 decoder and ActuatorGuard",
            "simulated_calibration": True,
            "decoded_frames": sim.supervisor.decoded,
            "rejected_frames": sim.supervisor.rejected,
            "hardware_map_sha256": sim.supervisor.mapping["map_hash"],
        }
        (output / "imu-polls.jsonl").write_text(
            "".join(json.dumps(p) + "\n" for p in sensor.polls)
        )
    (output / "trajectory.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in trace)
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if render:
        render_episode(sim, frames, output)
    return result, trace


def render_episode(sim, frames, output):
    from PIL import Image

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.04, 0, 0.21]
    camera.distance = 0.9
    camera.azimuth = 135
    camera.elevation = -15
    images = []
    with mujoco.Renderer(sim.model, height=480, width=640) as renderer:
        for q in frames:
            sim.data.qpos[:] = q
            mujoco.mj_forward(sim.model, sim.data)
            renderer.update_scene(sim.data, camera=camera)
            images.append(Image.fromarray(renderer.render()))
    if images:
        images[0].save(
            output / "episode.gif",
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
        )
        images[-1].save(output / "final.png")


def run_suite(variant, model_run, suite_path, output, mass_model=None):
    robot = make_robot(load_config(variant))
    mass_model = Path(mass_model) if mass_model is not None else None
    _apply_mass_model(robot, mass_model)
    retained_config = RobotConfig.model_validate_json(
        (model_run / "resolved-config.json").read_text()
    ).model_dump(mode="json")
    if retained_config != json.loads(robot.config.model_dump_json()):
        raise ValueError("model-run resolved configuration differs from variant")
    old = json.loads((model_run / "interfaces.json").read_text())
    if old != json.loads(json.dumps(robot.interface())):
        raise ValueError(
            "model-run interfaces differ from variant; regenerate the workbench first"
        )
    suite = yaml.safe_load(suite_path.read_text())
    settings = BalanceSettings.model_validate(suite["controller"])
    cases = [BalanceCase.model_validate(c) for c in suite["cases"]]
    if len({c.name for c in cases}) != len(cases):
        raise ValueError("duplicate case name")
    output.mkdir(parents=True, exist_ok=False)
    if (model_run / "meshes").exists():
        snapshot_tree(model_run / "meshes", output / "meshes")
    export_all(robot, output)
    library = build_supervisor(output)
    sources = [
        variant,
        suite_path,
        Path("robot/base.yaml"),
        *map(
            Path,
            [
                "lab/balance.py",
                "lab/balance_review.py",
                "lab/control.py",
                "lab/io.py",
                "lab/export.py",
                "lab/robot.py",
                "lab/config.py",
                "lab/imu_sim.py",
                "lab/imu.py",
                "lab/session.py",
                "lab/hardware.py",
                "lab/transport.py",
                "electronics/imu.yaml",
                "electronics/controller.yaml",
                "electronics/carrier/parts.json",
                "firmware/supervisor.c",
                "firmware/supervisor.h",
                "firmware/imu.c",
                "firmware/imu.h",
                "firmware/actuator.c",
                "firmware/actuator.h",
                "firmware/wire.c",
                "firmware/wire.h",
            ],
        ),
    ]
    sources += [
        model_run / "resolved-config.json",
        model_run / "interfaces.json",
        model_run / "report.json",
    ]
    sources += list((model_run / "meshes").glob("*.stl"))
    if mass_model is not None:
        sources += [mass_model, Path("lab/mass_model.py")]
    for p in sources:
        p = Path(p).absolute()
        rel = p.relative_to(Path.cwd().resolve())
        dest = output / "source" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(p, dest)
    report = dict(
        schema_version=1,
        status="running",
        variant=robot.config.name,
        mass_model=robot.mass_model,
        controller=settings.model_dump(),
        pitch_trim=WheelBalancer(robot, settings).trim,
        resolved_config_sha256=hashlib.sha256(
            (output / "resolved-config.json").read_bytes()
        ).hexdigest(),
        source_hashes={
            str(Path(p).absolute().relative_to(Path.cwd().resolve())): hashlib.sha256(
                Path(p).read_bytes()
            ).hexdigest()
            for p in sources
        },
        versions={"mujoco": mujoco.__version__, "python": sys.version},
        cases=[],
        limitations=[
            "unmeasured actuator response, friction/backlash, allocated inertia and primitive contact shapes",
            "planar estimator assumes small roll/yaw and no wheel slip; no external position sensing",
            "no physical IMU, balance firmware deployment, rough-terrain or whole-body manipulation evidence",
            "ideal-sensor cases retain legacy lifecycle; LSM6DSO cases exercise RLB1/ActuatorGuard with explicit simulated calibration",
            "LSM6DSO cases emulate sampled registers and bus timing; no physical mounting/calibration, measured scheduling or silicon qualification",
            "sensor-invalid STOP is a simulation abort, not a qualified physical stopping/fall strategy",
        ],
    )
    for case in cases:
        try:
            result, trace = evaluate(
                robot,
                output / "robot.xml",
                library,
                settings,
                case,
                output / case.name,
                render=case.name == "drive-return",
            )
            from .electrical import power_transient

            result["power"] = (
                power_transient(robot, trace, output / case.name)
                if trace
                else {"status": "not_run", "reason": "no motion/load trace"}
            )
            (output / case.name / "result.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
        except Exception as exc:
            (output / f"{case.name}-error.log").write_text(traceback.format_exc())
            result = dict(case=case.name, status="fail", error=str(exc))
        report["cases"].append(result)
        (output / "balance-report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = (
        "pass"
        if all(
            c["status"] == "pass" and c.get("power", {}).get("status") == "pass"
            for c in report["cases"]
        )
        else "fail"
    )
    (output / "balance-report.json").write_text(json.dumps(report, indent=2) + "\n")
    from .balance_review import write_review

    write_review(report, output)
    print(json.dumps({c["case"]: c["status"] for c in report["cases"]}, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("variant", type=Path)
    p.add_argument("--model-run", type=Path, required=True)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mass-model", type=Path)
    a = p.parse_args()
    report = run_suite(a.variant, a.model_run, a.suite, a.output, a.mass_model)
    sys.exit(int(report["status"] != "pass"))
