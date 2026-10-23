"""Retain four raw peak-hold states for bounded arm/body CAD checks."""

import argparse
import json
from pathlib import Path
from lab.config import RobotConfig
from lab.robot import make_robot
from tools.check_camera_support import sha, write


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    bindings = {}

    def read(p):
        bindings[str(p)] = sha(p)
        return json.loads(p.read_text())

    assembly = Path("artifacts/scout-shin-cover-composed-v2")
    robot = make_robot(
        RobotConfig.model_validate(read(assembly / "resolved-config.json"))
    )
    baseline = read(assembly / "interfaces.json")
    samples = []
    for profile, kind, case in [
        ("lower", "bounds", "drive-return"),
        ("nominal", "normal", "drive-return"),
        ("upper", "bounds", "drive-return"),
        ("nominal", "normal", "noisy-biased-imu"),
    ]:
        folder = Path(f"artifacts/scout-covered-rest-{profile}-{kind}-v1")
        report = read(folder / "balance-report.json")
        interface = read(folder / "interfaces.json")
        # Mass fields differ; the complete kinematic contract must agree.
        for key in ["joint_order", "joints", "frame_convention", "units"]:
            if interface[key] != baseline[key]:
                raise ValueError(f"Kinematic contract differs: {key}")
        c = next(c for c in report["cases"] if c["case"] == case)
        trace = folder / case / "trajectory.jsonl"
        bindings[str(trace)] = sha(trace)
        rows = [
            (i, json.loads(line))
            for i, line in enumerate(trace.read_text().splitlines(), 1)
            if line.strip()
        ]
        line, row = max(
            ((i, r) for i, r in rows if r["time_s"] > c["input"]["duration_s"] - 2),
            key=lambda x: abs(x[1]["pitch_rad"]),
        )
        q = dict(zip(interface["joint_order"], row["position_rad"], strict=True))
        robot.forward_kinematics(q, enforce_limits=False)
        excursions = {
            j.name: dict(value_rad=q[j.name], limits_rad=j.limits)
            for j in robot.joints
            if not j.continuous and not j.limits[0] <= q[j.name] <= j.limits[1]
        }
        samples.append(
            dict(
                name=f"{profile}-{case}-peak-hold",
                pose_rad=q,
                time_s=row["time_s"],
                pitch_rad=row["pitch_rad"],
                trace=str(trace),
                line=line,
                joint_limit_excursions=excursions,
                selection="maximum absolute torso pitch in final two seconds",
            )
        )
    write(
        output / "samples.json",
        dict(
            joint_order=baseline["joint_order"],
            samples=samples,
            source_bindings=bindings,
            scope="Raw joint states, root-neutral self-clearance only; no ground, scene or complete trajectory claim",
        ),
    )
    (output / "sampler.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
