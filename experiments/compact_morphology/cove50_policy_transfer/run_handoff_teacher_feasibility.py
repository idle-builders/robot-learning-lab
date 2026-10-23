"""Replay frozen PI on two native scenes missed by learned v6."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate

ARTIFACT = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010"
PLAN = ARTIFACT / "v7/teacher-feasibility-plan.json"
EVALUATOR = Path(__file__).with_name("evaluate_squat_reach_handoff.py")


def main() -> None:
    plan = json.loads(PLAN.read_text())
    out = ARTIFACT / "v7/teacher-feasibility"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in plan["cases"]:
        label = f"seed-{case['seed']}"
        result_path, trace_path, log_path = [out / f"{label}.{suffix}"
                                             for suffix in ("json", "npz", "log")]
        command = [sys.executable, str(EVALUATOR),
                   "--body-xml", str(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"),
                   "--squat-checkpoint", str(ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt"),
                   "--reach-checkpoint", str(ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1/controller.pt"),
                   "--reach-controller", "feedback",
                   "--reach-reference", str(ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"),
                   "--geometry", str(ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json"),
                   "--entry", "continuous", "--seed", str(case["seed"]),
                   "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
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
               "min_cube_to_hand_distance_m": report["phases"]["reach"]["min_cube_to_hand_distance_m"],
               "both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
               "self_body_contact_fraction": report["phases"]["reach"]["self_body_contact_fraction"],
               "result": str(result_path.relative_to(ROOT)),
               "trace": str(trace_path.relative_to(ROOT))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / "matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-handoff-v7-teacher-feasibility-matrix.v1",
        "controller": "frozen observed-goal PI", "cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
