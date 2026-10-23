#!/usr/bin/env python3
"""Generate a variant map and cross-compile a local Pico 2 W bench image."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.config import load_config
from lab.robot import make_robot
from lab.hardware import hardware_map
from lab.imu import load_imu
from lab.artifact_store import snapshot_file


def generate(variant, output, calibration=None):
    robot = make_robot(load_config(Path(variant)))
    calibrations = json.loads(Path(calibration).read_text()) if calibration else None
    mapping = hardware_map(robot, calibrations)
    imu = load_imu(robot=robot)
    imu_digest = hashlib.sha256(json.dumps(imu, sort_keys=True).encode()).hexdigest()
    records, board, digest = mapping["joints"], mapping["board"], mapping["map_hash"]
    pins = board["pins"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "hardware-map.json").write_text(json.dumps(mapping, indent=2))
    (output / "imu-config.json").write_text(json.dumps(imu, indent=2))
    (output / "resolved-config.json").write_text(robot.config.model_dump_json(indent=2))
    if calibration:
        shutil.copyfile(calibration, output / "calibration.json")
    p = robot.config.power
    divider = board["voltage_divider"]
    lines = [
        "#pragma once",
        '#include "actuator.h"',
        f'#define ROBOT_NAME "{robot.config.name}"',
        f'#define ROBOT_MAP_HASH "{digest}"',
        f"#define ROBOT_JOINT_COUNT {len(records)}",
        f"#define ROBOT_CALIBRATED {int(bool(calibrations))}",
        f"#define COMMAND_TIMEOUT_MS {robot.config.control.timeout_ms}",
        f"#define HOST_FRAME_TIMEOUT_MS {board['host_frame_timeout_ms']}",
        f"#define MOTOR_POWER_SETTLE_MS {board['motor_power_settle_ms']}",
        f"#define MINIMUM_BUS_V {p.minimum_bus_v}",
        f"#define COMMAND_HZ {board['command_hz']}",
        f"#define SERVO_BAUD {board['servo_baud']}",
        f"#define SERVO_TIMEOUT_US {board['feedback_timeout_us']}",
        f'#define IMU_CONFIG_HASH "{imu_digest}"',
        f"#define IMU_ADDRESS {imu['address_7bit']}",
        f"#define IMU_BUS_HZ {imu['bus_hz']}",
        f"#define IMU_POLL_US {imu['poll_interval_us']}",
        f"#define IMU_TRANSFER_TIMEOUT_US {imu['transfer_timeout_us']}",
        f"#define IMU_FRESHNESS_US {imu['freshness_us']}",
        f"#define ADC_REFERENCE_V {divider['adc_reference_v']}",
        f"#define DIVIDER_RATIO {(divider['high_ohm'] + divider['low_ohm']) / divider['low_ohm']}",
    ]
    lines += [f"#define PIN_{name.upper()} {pin}" for name, pin in pins.items()]
    lines += ["static const JointMap robot_map[ROBOT_JOINT_COUNT]={"]
    for r in records:
        lines.append(
            "  {"
            + ",".join(
                str(r[k])
                for k in [
                    "id",
                    "bus",
                    "mode",
                    "direction",
                    "zero",
                    "lower",
                    "upper",
                    "speed",
                ]
            )
            + "},"
        )
    lines += ["};"]
    (output / "robot_map.h").write_text("\n".join(lines) + "\n")
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--calibration",
        help="actual measured per-joint zero/direction JSON; absent means arm is disabled",
    )
    args = parser.parse_args()
    root = Path.cwd()
    output = Path(args.output).resolve()
    mapping = generate(args.variant, output, args.calibration)
    sdk = root / ".tools/pico-sdk"
    compiler = root / ".tools/arm-gnu-toolchain-14.3.rel1-darwin-arm64-arm-none-eabi"
    cmake = "/opt/homebrew/bin/cmake"
    env = dict(
        os.environ,
        PATH="/opt/homebrew/bin:"
        + str(compiler / "bin")
        + ":"
        + os.environ.get("PATH", ""),
    )
    commands = [
        [
            cmake,
            "-S",
            str(root / "firmware"),
            "-B",
            str(output / "build"),
            "-G",
            "Ninja",
            f"-DPICO_SDK_PATH={sdk}",
            f"-DPICO_TOOLCHAIN_PATH={compiler}",
            "-DPICO_BOARD=pico2_w",
            f"-DROBOT_GENERATED_DIR={output}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        [cmake, "--build", str(output / "build"), "--parallel", "4"],
    ]
    report = {
        "status": "not_run",
        "map_hash": mapping["map_hash"],
        "imu_config_sha256": hashlib.sha256(
            json.dumps(
                json.loads((output / "imu-config.json").read_text()), sort_keys=True
            ).encode()
        ).hexdigest(),
        "target": "pico2_w",
        "physical_hardware": "not_run",
        "steps": [],
    }
    paths = [
        *Path("firmware").glob("*"),
        Path("tools/build_firmware.py"),
        Path("electronics/controller.yaml"),
        Path("electronics/imu.yaml"),
        Path("electronics/carrier/parts.json"),
        Path(args.variant),
        Path("robot/base.yaml"),
        Path("robot/actuator-catalog.yaml"),
        Path("lab/config.py"),
        Path("lab/components.py"),
        Path("lab/hardware.py"),
        Path("lab/imu.py"),
        Path("lab/robot.py"),
        Path("lab/styling.py"),
    ]
    report["source_hashes"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()
    }
    for path in report["source_hashes"]:
        dest = output / "source" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(path, dest)
    try:
        report["sdk_commit"] = subprocess.check_output(
            ["git", "-C", str(sdk), "rev-parse", "HEAD"], text=True
        ).strip()
        report["compiler"] = subprocess.check_output(
            [str(compiler / "bin/arm-none-eabi-gcc"), "--version"], text=True
        ).splitlines()[0]
        for i, command in enumerate(commands):
            proc = subprocess.run(
                command, env=env, capture_output=True, text=True, timeout=300
            )
            (output / f"build-{i}.log").write_text(proc.stdout + proc.stderr)
            report["steps"].append(
                {
                    "command": command,
                    "exit_code": proc.returncode,
                    "log": f"build-{i}.log",
                }
            )
            proc.check_returncode()
        for suffix in ["elf", "bin", "uf2"]:
            shutil.copyfile(
                output / "build" / f"robot_controller.{suffix}",
                output / f"robot_controller.{suffix}",
            )
        size = subprocess.check_output(
            [
                str(compiler / "bin/arm-none-eabi-size"),
                str(output / "robot_controller.elf"),
            ],
            text=True,
        )
        report.update(
            status="pass",
            image_sizes_bytes={
                p.name: p.stat().st_size for p in output.glob("robot_controller.*")
            },
            elf_size=size,
        )
    except Exception as exc:
        report.update(status="fail", error=str(exc))
    (output / "build-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
