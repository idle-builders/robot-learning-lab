"""Run the predeclared selected-body learned-reach handoff matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010"
EVALUATOR = Path(__file__).with_name("evaluate_squat_reach_handoff.py")


def gate(report: dict) -> list[str]:
    faults = []
    expected_duration = 8.0 if report["entry"] == "reference-start" else 15.5
    if report["ended_early"] or report["duration_s"] < expected_duration:
        faults.append("ended before full route")
    if report["max_abs_unbounded_action"] > 1.0:
        faults.append("raw action clipping")
    phases = ("reach",) if report["entry"] == "reference-start" else ("preposition", "reach")
    for phase in phases:
        part = report["phases"].get(phase)
        if part is None:
            faults.append(f"missing {phase} phase")
            continue
        if part["both_feet_fraction"] < .99:
            faults.append(f"{phase}: two-foot support below 99%")
        if part["rated_leg_fraction"] > .1:
            faults.append(f"{phase}: rated leg fraction above 10%")
        if part["max_hand_floor_force_n"] >= 1:
            faults.append(f"{phase}: hand-floor force at least 1 N")
        if part["self_body_contact_fraction"] > 0:
            faults.append(f"{phase}: self-body contact")
    if report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0) < 1:
        faults.append("finger contact shorter than 1 s")
    return faults


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("reference", "reference-clearance", "reference-tuned",
                                            "reference-tuned2", "reference-tuned3",
                                            "development", "reserved"), required=True)
    args = parser.parse_args()
    plan = json.loads((ARTIFACT / "plan.json").read_text())
    amendment = json.loads((ARTIFACT / "route-amendment.json").read_text())
    if args.split == "development":
        tuned = json.loads((ARTIFACT / "reference-tuned3/matrix.json").read_text())
        if not tuned["all_pass"]:
            raise ValueError("Do not run development before the tuned reference passes")
    if args.split == "reserved":
        development = json.loads((ARTIFACT / "development/matrix.json").read_text())
        if not development["all_pass"]:
            raise ValueError("Do not consume reserved cases before development passes")
    if args.split in ("reference-clearance", "reference-tuned", "reference-tuned2",
                      "reference-tuned3"):
        key = {"reference-clearance": "reference_clearance_check",
               "reference-tuned": "reference_tuned_check",
               "reference-tuned2": "reference_tuned2_check",
               "reference-tuned3": "reference_tuned3_check"}[args.split]
        cases = [amendment[key]]
    elif args.split == "reference":
        cases = [plan["reference_start"]]
    else:
        route = amendment["continuous_route_if_tuned3_passes"]
        cases = [{**case, **route} for case in
                 plan["development" if args.split == "development" else "reserved_if_development_passes"]]
    out_dir = ARTIFACT / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        label = "reference" if args.split.startswith("reference") else f"seed-{case['seed']}"
        result_path = out_dir / f"{label}.json"
        trace_path = out_dir / f"{label}.npz"
        log_path = out_dir / f"{label}.log"
        command = [sys.executable, str(EVALUATOR),
                   "--body-xml", str(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"),
                   "--squat-checkpoint", str(ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt"),
                   "--reach-checkpoint", str(ROOT / "artifacts/ape-cove50-side-reach-20261010/reach-arm-residual-integral-v5/actor.pt"),
                   "--reach-controller", "learned-residual",
                   "--reach-reference", str(ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"),
                   "--geometry", str(ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json"),
                   "--entry", "reference-start" if args.split.startswith("reference") else "continuous",
                   "--seed", str(case["seed"]), "--rail-v", str(case["rail_v"]),
                   "--delay-ms", str(case["delay_ms"]),
                   "--toy-offset-x-m", str(case.get("toy_offset_x_m", 0)),
                   "--toy-offset-y-m", str(case.get("toy_offset_y_m", 0)),
                   "--pitch-offset-rad", str(case.get("pitch_offset_rad", .105)),
                   "--roll-clearance-rad", str(case.get("roll_clearance_rad", -.25)),
                   "--out", str(result_path), "--trace", str(trace_path)]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with log_path.open("w") as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        report = json.loads(result_path.read_text())
        faults = gate(report)
        row = {"case": label, **case, "pass": not faults, "faults": faults,
               "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
               "both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
               "rated_leg_fraction": report["phases"]["reach"]["rated_leg_fraction"],
               "self_body_contact_fraction": report["phases"]["reach"]["self_body_contact_fraction"],
               "max_abs_unbounded_action": report["max_abs_unbounded_action"],
               "result": str(result_path.relative_to(ROOT)),
               "trace": str(trace_path.relative_to(ROOT))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    matrix = {"schema": "robot-learning-lab.cove50-squat-learned-reach-matrix.v1",
              "split": args.split, "all_pass": all(row["pass"] for row in rows), "cases": rows}
    (out_dir / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({"split": args.split, "all_pass": matrix["all_pass"]}), flush=True)


if __name__ == "__main__":
    main()
