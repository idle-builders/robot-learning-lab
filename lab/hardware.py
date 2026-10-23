"""Shared candidate actuator/pin map for embedded builds and session simulation."""

import hashlib
import json
import math
from pathlib import Path
import yaml
from .components import check_components


def hardware_map(robot, calibrations=None):
    if check_components(robot)["status"] != "pass":
        raise ValueError(
            "select compatible sourced actuator parts before building hardware"
        )
    board = yaml.safe_load(Path("electronics/controller.yaml").read_text())
    frame_timeout = board.get("host_frame_timeout_ms")
    if type(frame_timeout) is not int or not 1 <= frame_timeout <= robot.config.control.timeout_ms:
        raise ValueError("host frame timeout must be a positive integer within the command deadline")
    settle = board.get("motor_power_settle_ms")
    if type(settle) is not int or not 1 <= settle < robot.config.control.timeout_ms:
        raise ValueError("motor power settle time must be a positive integer below the command deadline")
    pins = board["pins"]
    if len(set(pins.values())) != len(pins) or any(
        not 0 <= p <= 28 for p in pins.values()
    ):
        raise ValueError("duplicate or invalid GPIO assignment")
    records = []
    if calibrations is not None and (
        not isinstance(calibrations, dict)
        or set(calibrations) != set(j.name for j in robot.joints)
    ):
        raise ValueError("calibration must match the complete joint-name set")
    for i, j in enumerate(robot.joints):
        cal = calibrations[j.name] if calibrations else {"zero": 2048, "direction": 1}
        if (
            set(cal) != {"zero", "direction"}
            or type(cal["direction"]) is not int
            or cal["direction"] not in [-1, 1]
            or type(cal["zero"]) is not int
            or not 0 <= cal["zero"] <= 4095
        ):
            raise ValueError(f"invalid calibration: {j.name}")
        speed = robot.config.actuators[j.family].velocity_limit_rad_s
        if not 1 <= speed * 4096 / math.tau <= 32767:
            raise ValueError(f"speed cannot be represented by the servo: {j.name}")
        if not j.continuous and any(
            not 0 <= cal["zero"] + cal["direction"] * angle * 4096 / math.tau <= 4095
            for angle in j.limits
        ):
            raise ValueError(f"calibrated range leaves encoder travel: {j.name}")
        records.append(
            dict(
                name=j.name,
                id=i + 1,
                bus=int(j.name.startswith("right_")),
                mode=int(j.continuous),
                **cal,
                lower=j.limits[0],
                upper=j.limits[1],
                speed=speed,
                command_unit="rad/s" if j.continuous else "rad",
            )
        )
    mapping = {
        "schema_version": 1,
        "robot": robot.config.name,
        "calibrated": bool(calibrations),
        "joints": records,
        "board": board,
        "scope": "supported-rig target; no physical validation, Wi-Fi or balance controller",
    }
    digest = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    mapping["map_hash"] = digest
    return mapping
