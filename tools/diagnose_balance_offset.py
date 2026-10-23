"""Retain the measured-posture/static-trim diagnosis from existing episodes."""

import json
from pathlib import Path
import argparse
from lab.balance import observed_mass_trim
from lab.config import load_config
from lab.io import JointState
from lab.mass_model import apply_mass_model
from lab.robot import make_robot
from tools.check_camera_support import sha, write


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    rows = []

    def load(path):
        sources[str(path)] = sha(path)
        return json.loads(path.read_text())

    for name in ("pip", "scout"):
        variant = Path(f"variants/{name}-distal-grip.yaml")
        robot = make_robot(load_config(variant))
        sources[str(variant)] = sha(variant)
        mass = Path(
            f"artifacts/{name}-integral-head-mass-v1/nominal/mass-properties.json"
        )
        apply_mass_model(robot, mass)
        sources[str(mass)] = sha(mass)
        folder = Path(f"artifacts/{name}-composed-parked-nominal-balance-v1")
        report = load(folder / "balance-report.json")
        for case in ("tilt-recovery", "drive-return"):
            path = folder / case / "trajectory.jsonl"
            sources[str(path)] = sha(path)
            trace = [json.loads(x) for x in path.read_text().splitlines()]
            last = trace[-1]
            result = next(c for c in report["cases"] if c["case"] == case)
            state = JointState(
                last["time_s"],
                tuple(result["joint_order"]),
                tuple(last["position_rad"]),
                tuple(last["velocity_rad_s"]),
                tuple(last["effort_nm"]),
                last["supervisor_status"],
            )
            observed = observed_mass_trim(robot, state)
            fixed = result["pitch_trim"]["pitch_rad"]
            rows.append(
                dict(
                    robot=name,
                    case=case,
                    fixed_trim_rad=fixed,
                    observed_posture_trim_rad=observed["pitch_rad"],
                    estimated_pitch_rad=last["estimated_pitch_rad"],
                    truth_pitch_rad=last["pitch_rad"],
                    estimated_position_m=last["estimated_position_m"],
                    truth_position_m=last["truth_x_m"],
                    initial_estimated_position_m=trace[0]["estimated_position_m"],
                    initial_recorded_truth_position_m=trace[0]["truth_x_m"],
                    axle_mismatch_m=observed["axle_mismatch_m"],
                )
            )
    sources[str(Path(__file__).relative_to(Path.cwd()))] = sha(Path(__file__))
    sources["lab/balance.py"] = sha(Path("lab/balance.py"))
    write(
        output / "report.json",
        dict(
            status="partial",
            scope="Retrospective diagnosis, not closed-loop or physical qualification. End-row encoders are post-step; the stored estimator was evaluated one command period earlier.",
            source_bindings=sources,
            cases=rows,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
