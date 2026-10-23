"""Make a small, reversible Zeroth-01 motor-rating variant for the upstream walker.

The builder's CAD-derived MJCF and identified controller remain unchanged. Only
the torque clamp and no-load planner speed in the actuator JSONs
are limited to Feetech's published 12 V figures. This is a screening envelope,
not an identified torque-speed/thermal model or proof of a safe real-world gait.
"""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="builder's sim/assets/zbot-cad")
    parser.add_argument("--catalog", type=Path, default=Path("robot/actuator-catalog.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torque-limit", choices=("stall", "rated"), default="stall")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    catalog = yaml.safe_load(args.catalog.read_text())["parts"]
    motors = {
        "feetech_sts3250.json": catalog["feetech-sts3250-c001"],
        "feetech_sts3215_12v.json": catalog["feetech-st3215-c018"],
    }
    output.mkdir(parents=True)
    (output / "actuators").mkdir()
    for name in ("robot.mjcf", "metadata.json", "meshes"):
        (output / name).symlink_to(source / name, target_is_directory=name == "meshes")

    manifest = {
        "source_mjcf_sha256": digest(source / "robot.mjcf"),
        "source_metadata_sha256": digest(source / "metadata.json"),
        "catalog_sha256": digest(args.catalog),
        "envelope": f"manufacturer 12 V {args.torque_limit} torque and no-load speed",
        "actuators": {},
    }
    for filename, rating in motors.items():
        original_path = source / "actuators" / filename
        params = json.loads(original_path.read_text())
        original = {key: params[key] for key in ("max_torque", "max_velocity")}
        params["max_torque"] = rating[f"{args.torque_limit}_effort_nm"]
        params["max_velocity"] = rating["no_load_speed_rad_s"]
        out = output / "actuators" / filename
        out.write_text(json.dumps(params, indent=2) + "\n")
        manifest["actuators"][filename] = {
            "source_sha256": digest(original_path),
            "original_fit": original,
            "rating_source": rating["source"],
            "rated_torque_nm": rating["rated_effort_nm"],
            "variant": {key: params[key] for key in original},
            "variant_sha256": digest(out),
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)
    for filename, info in manifest["actuators"].items():
        print(filename, info["variant"], "rated torque", info["rated_torque_nm"])


if __name__ == "__main__":
    main()
