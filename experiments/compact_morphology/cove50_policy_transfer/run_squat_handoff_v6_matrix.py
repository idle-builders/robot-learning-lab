"""Replay the handoff-trained arm residual on predeclared Cove-50 cases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate

ARTIFACT = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010"
EVALUATOR = Path(__file__).with_name("evaluate_squat_reach_handoff.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("reference", "development", "reserved"), required=True)
    args = parser.parse_args()
    plan = json.loads((ARTIFACT / "v6-plan.json").read_text())
    if args.split == "development":
        reference = json.loads((ARTIFACT / "v6/reference/matrix.json").read_text())
        if not reference["all_pass"]:
            raise ValueError("Reference gate failed; preserve development seeds")
    if args.split == "reserved":
        development = json.loads((ARTIFACT / "v6/development/matrix.json").read_text())
        if not development["all_pass"]:
            raise ValueError("Development gate failed; preserve fresh reserved seeds")
    cases = ([plan["saved_start_diagnostic"]] if args.split == "reference" else
             plan["development_replay"] if args.split == "development" else
             plan["fresh_qualification"])
    out = ARTIFACT / "v6" / args.split
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        label = "reference" if args.split == "reference" else f"seed-{case['seed']}"
        result_path = out / f"{label}.json"
        trace_path = out / f"{label}.npz"
        log_path = out / f"{label}.log"
        command = [sys.executable, str(EVALUATOR),
                   "--body-xml", str(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"),
                   "--squat-checkpoint", str(ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt"),
                   "--reach-checkpoint", str(ARTIFACT / "v6/actor.pt"),
                   "--reach-controller", "learned-residual",
                   "--reach-reference", str(ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"),
                   "--geometry", str(ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json"),
                   "--entry", "reference-start" if args.split == "reference" else "continuous",
                   "--seed", str(case["seed"]), "--rail-v", str(case["rail_v"]),
                   "--delay-ms", str(case["delay_ms"]),
                   "--toy-offset-x-m", str(case["toy_offset_x_m"]),
                   "--toy-offset-y-m", str(case["toy_offset_y_m"]),
                   "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
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
    matrix = {"schema": "robot-learning-lab.cove50-squat-handoff-v6-matrix.v1",
              "split": args.split, "all_pass": all(row["pass"] for row in rows), "cases": rows}
    (out / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({"split": args.split, "all_pass": matrix["all_pass"]}), flush=True)


if __name__ == "__main__":
    main()
