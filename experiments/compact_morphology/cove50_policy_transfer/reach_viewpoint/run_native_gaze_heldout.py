"""Replay the frozen slow gaze on four disjoint native Cove-50 handoffs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate

OUT = ROOT / "artifacts/ape-cove50-reach-viewpoint-20261010"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
ACTOR = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010/v7/actor.pt"
EVALUATOR = Path(__file__).resolve().parents[1] / "evaluate_squat_reach_handoff.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=OUT / "gaze-native-heldout-plan.json")
    parser.add_argument("--out-dir", type=Path, default=OUT / "native-gaze/heldout")
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    args.out_dir = args.out_dir.resolve()
    plan = json.loads(args.plan.read_text())
    assert sha(BODY) == plan["selected_body_sha256"]
    assert sha(ACTOR) == plan["saved_reach_actor_sha256"]
    assert sha(EVALUATOR) == plan["gaze_evaluator_sha256"]
    dest = args.out_dir
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in plan["cases"]:
        stem = f"seed-{case['seed']}"
        result = dest / f"{stem}.json"
        trace = dest / f"{stem}.npz"
        log = dest / f"{stem}.log"
        command = [sys.executable, str(EVALUATOR),
                   "--body-xml", str(BODY),
                   "--squat-checkpoint", str(ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt"),
                   "--reach-checkpoint", str(ACTOR),
                   "--reach-controller", "learned-residual",
                   "--reach-reference", str(ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"),
                   "--geometry", str(ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json"),
                   "--entry", "continuous", "--seed", str(case["seed"]),
                   "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
                   "--toy-offset-x-m", str(case["toy_offset_x_m"]),
                   "--toy-offset-y-m", str(case["toy_offset_y_m"]),
                   "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
                   "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
                   "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(plan["gaze_head_yaw_rad"]),
                   "--gaze-start-s", str(plan["gaze_start_s"]),
                   "--gaze-ramp-s", str(plan["gaze_ramp_s"]),
                   "--out", str(result), "--trace", str(trace)]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if result.exists():
            prior = json.loads(result.read_text())
            assert prior["seed"] == case["seed"] and prior["rail_v"] == case["rail_v"]
            assert prior["delay_ms"] == case["delay_ms"] and prior["gaze_head_yaw_rad"] == plan["gaze_head_yaw_rad"]
            assert prior["gaze_start_s"] == plan["gaze_start_s"] and prior["gaze_ramp_s"] == plan["gaze_ramp_s"]
            assert prior["body_sha256"] == plan["selected_body_sha256"]
            assert prior["reach_checkpoint_sha256"] == plan["saved_reach_actor_sha256"]
            assert prior["evaluator_sha256"] == plan["gaze_evaluator_sha256"]
            assert prior["toy_offset_b_m"] == [case["toy_offset_x_m"], case["toy_offset_y_m"]]
            completed_code = 0
        else:
            with log.open("w") as stream:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, check=False)
            completed_code = completed.returncode
        if completed_code != 0 or not result.exists():
            rows.append({"case": stem, **case, "pass": False,
                         "faults": [f"evaluator exit {completed_code}"],
                         "log": str(log.relative_to(ROOT))})
            print(json.dumps(rows[-1]), flush=True)
            continue
        report = json.loads(result.read_text())
        entry = next(row for row in report["timeseries"] if row["time_s"] == 7.5)
        yaw_error = round(abs(entry["head_yaw_rad"] - plan["gaze_head_yaw_rad"]), 5)
        faults = gate(report)
        if yaw_error > .05:
            faults.append("head yaw more than 0.05 rad from target at reach entry")
        rows.append({"case": stem, **case, "pass": not faults, "faults": faults,
                     "entry_head_yaw_rad": entry["head_yaw_rad"],
                     "entry_yaw_error_rad": yaw_error,
                     "entry_head_yaw_effort_nm": entry["head_yaw_effort_nm"],
                     "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
                     "reach_both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
                     "reach_rated_leg_fraction": report["phases"]["reach"]["rated_leg_fraction"],
                     "max_abs_unbounded_action": report["max_abs_unbounded_action"],
                     "result": str(result.relative_to(ROOT)),
                     "trace": str(trace.relative_to(ROOT)),
                     "log": str(log.relative_to(ROOT))})
        print(json.dumps(rows[-1]), flush=True)
    matrix = {"schema": "robot-learning-lab.cove50-native-gaze-heldout-matrix.v1",
              "plan": str(args.plan.relative_to(ROOT)),
              "native_pass_count": sum(row["pass"] for row in rows),
              "native_sample_count": len(rows), "cases": rows}
    (dest / "matrix.json").write_text(json.dumps(matrix, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"native_pass_count": matrix["native_pass_count"],
                      "native_sample_count": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
