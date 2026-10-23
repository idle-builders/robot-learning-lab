"""Run one declared Cove-50 squat-to-reach trial split in fresh MJLab processes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts/ape-cove50-squat-reach-handoff-20261010"
EVALUATOR = Path(__file__).with_name("evaluate_squat_reach_handoff.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", choices=("selected", "candidate"), required=True)
    parser.add_argument("--split", choices=("reference", "development", "reserved"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--roll-clearance-rad", type=float, default=-.25)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    plan = json.loads((ARTIFACT / "plan.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    body = ("artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
            if args.body == "selected" else
            "artifacts/ape-cove50-hand-clearance-coupon/physics-v2/candidate-training.xml")
    reference = ("artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/"
                 f"development-{args.body}.npz")
    cases = ([{"seed": 2601, "rail_v": 12, "delay_ms": 0,
               "toy_offset_x_m": 0, "toy_offset_y_m": 0}]
             if args.split == "reference" else
             plan["development" if args.split == "development" else "reserved_if_development_passes"])
    for case in cases:
        entry = "reference-start" if args.split == "reference" else "continuous"
        label = f"{args.body}-{'reference' if entry == 'reference-start' else 'continuous-' + str(case['seed'])}-roll025"
        report = args.out_dir / f"{label}.json"
        trace = args.out_dir / f"{label}.npz"
        log = args.out_dir / f"{label}.log"
        if args.skip_existing and report.exists() and trace.exists():
            print(f"skip existing {label}", flush=True)
            continue
        command = [sys.executable, str(EVALUATOR),
                   "--body-xml", str(ROOT / body),
                   "--squat-checkpoint", str(ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt"),
                   "--reach-checkpoint", str(ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1/controller.pt"),
                   "--reach-reference", str(ROOT / reference),
                   "--geometry", str(ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json"),
                   "--entry", entry, "--seed", str(case["seed"]),
                   "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
                   "--toy-offset-x-m", str(case["toy_offset_x_m"]),
                   "--toy-offset-y-m", str(case["toy_offset_y_m"]),
                   "--roll-clearance-rad", str(args.roll_clearance_rad),
                   "--out", str(report), "--trace", str(trace)]
        print(f"run {label}", flush=True)
        with log.open("w") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = json.loads(report.read_text())
        reach = result["phases"]["reach"]
        print(json.dumps({"label": label, "contact_s": reach["continuous_finger_contact_s"],
                          "support": reach["both_feet_fraction"],
                          "self_contact_fraction": reach["self_body_contact_fraction"],
                          "rated_leg_fraction": reach["rated_leg_fraction"],
                          "max_raw_action": result["max_abs_unbounded_action"]}), flush=True)


if __name__ == "__main__":
    main()
