"""Selected IMU contract and checks against the unchanged carrier interface."""

from pathlib import Path
import json
import yaml
import numpy as np


def load_imu(path=Path("electronics/imu.yaml"), *, robot=None):
    config = yaml.safe_load(Path(path).read_text())
    fixed = {
        "schema_version": 1,
        "part": "LSM6DSOTR",
        "bus": "i2c0",
        "address_7bit": 0x6A,
        "odr_hz": 208,
        "accel_full_scale_g": 4,
        "gyro_full_scale_dps": 500,
        "supply_v": 3.3,
        "pins": {"sda": 12, "scl": 13, "irq": 14},
        "connector_pins": {"V3": 1, "GND": 2, "SDA": 3, "SCL": 4, "INT1": 5},
        "carrier_connector": "J7",
        "telemetry_frame": "sensor",
        "use": "telemetry_only_supported_rig",
    }
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(f"IMU driver/board contract mismatch: {key}")
    for key, limits in {
        "bus_hz": (100000, 400000),
        "poll_interval_us": (1000, 20000),
        "transfer_timeout_us": (2000, 5000),
        "freshness_us": (5000, 100000),
    }.items():
        if (
            type(config.get(key)) is not int
            or not limits[0] <= config[key] <= limits[1]
        ):
            raise ValueError(f"invalid IMU timing allocation: {key}")
    if config["bus_hz"] not in (100000, 400000):
        raise ValueError("only standard/fast I2C modes are modeled")
    controller = yaml.safe_load(Path("electronics/controller.yaml").read_text())
    for signal, gpio in config["pins"].items():
        if controller["pins"][f"imu_{signal}_reserved"] != gpio:
            raise ValueError("IMU/controller pin mismatch")
    parts = json.loads(Path("electronics/carrier/parts.json").read_text())
    if config["pullups"] != {
        "location": "carrier",
        "ohm": parts["R4k7"]["resistance_ohm"],
    }:
        raise ValueError("IMU pullups differ from carrier parts")
    frame = config["sensor_frame"]
    matrix = np.asarray(frame["torso_from_sensor_rotation"], dtype=float)
    position = np.asarray(frame["allocated_xyz_m"], dtype=float)
    if (
        frame["parent"] != "torso"
        or matrix.shape != (3, 3)
        or position.shape != (3,)
        or not np.isfinite(position).all()
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8, rtol=0)
        or not np.isclose(np.linalg.det(matrix), 1, atol=1e-8, rtol=0)
    ):
        raise ValueError(
            "IMU frame requires a finite proper rotation and torso position"
        )
    if robot is not None and robot.config.dimensions.imu_mount_enabled:
        config["sensor_frame"] = {
            "parent": "torso",
            "allocated_xyz_m": list(robot.imu_position),
            "torso_from_sensor_rotation": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
            "status": "nominal_mount_with_assumed_axis_alignment_not_calibrated",
            "source": "variant board pose; package centre 1.6 + 0.83/2 mm above substrate bottom",
            "axis_assumption": "sensor axes assumed parallel to STEP board axes; package-to-die mapping and physical sign tests unqualified",
        }
    return config


def bus_budget(config, capacitance_pf):
    """First-order NXP UM10204 rise-time estimate; not measured harness evidence."""
    if capacitance_pf <= 0:
        raise ValueError("positive bus capacitance required")
    resistance = config["pullups"]["ohm"]
    rise_ns = 0.8473 * resistance * capacitance_pf * 1e-3
    limit_ns = 1000 if config["bus_hz"] == 100000 else 300
    # Register-pointer write + repeated read: 3 address/register bytes plus data,
    # each with ACK/NACK. Two transfers: one status byte and twelve output bytes.
    wire_us = (4 + 15) * 9 / config["bus_hz"] * 1e6
    return {
        "assumed_capacitance_pf": capacitance_pf,
        "rise_ns": rise_ns,
        "rise_limit_ns": limit_ns,
        "rise_status": "pass" if rise_ns <= limit_ns else "fail",
        "status_and_sample_wire_lower_bound_us": wire_us,
        "scope": "RC and wire-time allocation; no board/cable/clock-stretch measurements",
    }
