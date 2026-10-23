"""Replay the saved v7 actor with an entry-error-conditioned endpoint route."""

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
OUT = ARTIFACT / "v8/conditioned-route"
EVALUATOR = Path(__file__).with_name("evaluate_squat_reach_handoff.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("reference", "development", "reserved"), required=True)
    args = parser.parse_args()
    plan = json.loads((ARTIFACT / "v8/conditioned-route-plan.json").read_text())
    diagnostics = json.loads((OUT / "diagnostic/matrix.json").read_text())
    if not diagnostics["all_pass"]:
        raise ValueError("Contrasting diagnostic gate failed; preserve other starts")
    if args.split == "development" and not json.loads((OUT / "reference/matrix.json").read_text())["all_pass"]:
        raise ValueError("Saved-start gate failed; preserve other development starts")
    if args.split == "reserved" and not json.loads((OUT / "development/matrix.json").read_text())["all_pass"]:
        raise ValueError("Development gate failed; preserve fresh qualification seeds")
    cases = ([{"seed": 2601, "rail_v": 12, "delay_ms": 0,
               "toy_offset_x_m": 0, "toy_offset_y_m": 0}] if args.split == "reference" else
             plan["development_replay"] if args.split == "development" else
             plan["fresh_qualification"])
    out = OUT / args.split
    out.mkdir(parents=True, exist_ok=True)
    diagnostic_rows = {row["seed"]: row for row in diagnostics["cases"]}
    rows = []
    for case in cases:
        label = "reference" if args.split == "reference" else f"seed-{case['seed']}"
        if args.split == "development" and case["seed"] in diagnostic_rows:
            prior = diagnostic_rows[case["seed"]]
            if any(prior[key] != case[key] for key in case):
                raise ValueError(f"Diagnostic parameters changed for {label}")
            rows.append({**prior, "case": label, "reused_from_diagnostic": True})
            print(json.dumps(rows[-1]), flush=True)
            continue
        result_path, trace_path, log_path = [out / f"{label}.{suffix}"
                                             for suffix in ("json", "npz", "log")]
        command = [sys.executable, str(EVALUATOR),
                   "--body-xml", str(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"),
                   "--squat-checkpoint", str(ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt"),
                   "--reach-checkpoint", str(ARTIFACT / "v7/actor.pt"),
                   "--reach-controller", "learned-residual",
                   "--reach-reference", str(ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"),
                   "--geometry", str(ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json"),
                   "--entry", "reference-start" if args.split == "reference" else "continuous",
                   "--seed", str(case["seed"]), "--rail-v", str(case["rail_v"]),
                   "--delay-ms", str(case["delay_ms"]),
                   "--toy-offset-x-m", str(case["toy_offset_x_m"]),
                   "--toy-offset-y-m", str(case["toy_offset_y_m"]),
                   "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
                   "--terminal-roll-offset-rad", "-.08",
                   "--terminal-roll-mode", "entry-error-x",
                   "--reach-duration-s", "9.0",
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
               "route_entry_error_x_m": report["route_entry_error_x_m"],
               "applied_terminal_roll_offset_rad": report["applied_terminal_roll_offset_rad"],
               "result": str(result_path.relative_to(ROOT)),
               "trace": str(trace_path.relative_to(ROOT))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    matrix = {"schema": "robot-learning-lab.cove50-squat-handoff-v8-conditioned-matrix.v1",
              "split": args.split, "all_pass": all(row["pass"] for row in rows),
              "cases": rows}
    (out / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({"split": args.split, "all_pass": matrix["all_pass"]}), flush=True)


if __name__ == "__main__":
    main()
