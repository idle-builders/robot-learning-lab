"""Collect varied native side-reach teachers and retain failed cases too.

The teacher is an authored joint-target trajectory, not a learned policy.
The MJLab actor's ee_to_cube term uses simulator truth; a later deployment must
supply a measured target-relative estimate with the same frame and units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
PROBE = Path(__file__).with_name("probe_side_reach_native.py")
OUT = ROOT / "artifacts/ape-cove50-side-reach-20261009/teacher-dataset-v1"
GEOMETRY = ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json"
NOMINAL = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
HEAVY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-heavy-lowgrip.xml"
REFERENCE = ROOT / "artifacts/ape-cove50-side-reach-20261009/action-contract-v1.npz"


CASES = {
    # Development placements deliberately vary robot-to-toy translation, supply,
    # contact body and command delay. Toy offsets do not alter the floor-toy size.
    "train-center-nominal12": ("train", False, 12, 0, 0, 0, 0, 0, 2144),
    "train-start-xminus5": ("train", False, 12, 0, -.005, 0, 0, 0, 2145),
    "train-start-xplus5": ("train", False, 12, 0, .005, 0, 0, 0, 2146),
    "train-start-yminus5": ("train", False, 12, 0, 0, -.005, 0, 0, 2147),
    "train-start-yplus5": ("train", False, 12, 0, 0, .005, 0, 0, 2148),
    "train-toy-xminus4": ("train", False, 12, 0, 0, 0, -.004, 0, 2149),
    "train-toy-xplus4": ("train", False, 12, 0, 0, 0, .004, 0, 2150),
    "train-toy-yminus4": ("train", False, 12, 0, 0, 0, 0, -.004, 2151),
    "train-toy-yplus4": ("train", False, 12, 0, 0, 0, 0, .004, 2152),
    "train-lowrail11": ("train", False, 11, 10, .002, -.002, 0, 0, 2153),
    "train-heavy12": ("train", True, 12, 0, -.002, .002, 0, 0, 2154),
    "train-heavy11delay": ("train", True, 11, 10, 0, 0, 0, 0, 2155),
    # Reserved combinations are not used for teacher fitting or tuning.
    "held-toy-diagonal-a": ("held", False, 12, 0, 0, 0, -.006, .006, 9011),
    "held-toy-diagonal-b": ("held", False, 11, 10, 0, 0, .006, -.006, 9012),
    "held-start-diagonal": ("held", False, 12, 0, -.006, .006, 0, 0, 9013),
    "held-heavy11-shift": ("held", True, 11, 10, .004, -.004, 0, 0, 9014),
    "held-heavy10-boundary": ("held", True, 10, 10, 0, 0, 0, 0, 9015),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(name: str) -> dict:
    split, heavy, rail, delay, sx, sy, tx, ty, seed = CASES[name]
    body = HEAVY if heavy else NOMINAL
    case_out = OUT / f"{name}.json"
    data_out = OUT / f"{name}.npz"
    command = [sys.executable, str(PROBE), "--geometry", str(GEOMETRY),
               "--body-xml", str(body), "--out", str(case_out),
               "--dataset-out", str(data_out), "--seed", str(seed),
               "--xy-correction-reference", str(REFERENCE),
               "--rail-v", str(rail), "--delay-ms", str(delay),
               "--start-x-offset-m", str(.07 + sx),
               "--start-y-offset-m", str(sy),
               "--toy-x-offset-m", str(tx), "--toy-y-offset-m", str(ty)]
    with (OUT / f"{name}.log").open("w") as log:
        subprocess.run(command, check=True, cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT)
    report = json.loads(case_out.read_text())
    accepted = (report["survived"] and
                report["longest_continuous_finger_toy_contact_s"] >= 1.0 and
                report["both_feet_grounded_fraction"] >= .99 and
                report["rated_leg_fraction"] <= .1 and
                report["max_hand_ground_force_n"] < 1.0)
    row = {"case": name, "split": split, "accepted_teacher": accepted,
           "body": "heavy_lowgrip" if heavy else "nominal", "rail_v": rail,
           "delay_ms": delay, "start_xy_delta_m": [sx, sy],
           "toy_xy_delta_m": [tx, ty], "seed": seed,
           "contact_s": report["longest_continuous_finger_toy_contact_s"],
           "survived": report["survived"],
           "both_feet_fraction": report["both_feet_grounded_fraction"],
           "rated_leg_fraction": report["rated_leg_fraction"],
           "max_hand_ground_force_n": report["max_hand_ground_force_n"],
           "report": str(case_out.relative_to(ROOT)),
           "dataset": str(data_out.relative_to(ROOT))}
    print(json.dumps(row), flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", choices=tuple(CASES))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_case(name) for name in (args.cases or CASES)]
    manifest = {"schema": "robot-learning-lab.cove50-native-reach-teachers.v1",
                "method": "authored_native_MJLab_teacher",
                "selected_design_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
                "nominal_body_sha256": sha(NOMINAL),
                "heavy_body_sha256": sha(HEAVY),
                "geometry_sha256": sha(GEOMETRY), "probe_sha256": sha(PROBE),
                "jacobian_reference_sha256": sha(REFERENCE),
                "collector_sha256": sha(Path(__file__)),
                "observation_warning": "MJLab ee_to_cube is simulator truth; no perception or robot deployment qualification",
                "cases": rows}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
