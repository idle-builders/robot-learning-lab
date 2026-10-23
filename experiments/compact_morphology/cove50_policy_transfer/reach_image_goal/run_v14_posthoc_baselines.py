"""Diagnose fresh v14 contact misses with saved v9 and feedback baselines.

These replays are opened post hoc. They never alter the frozen fresh score.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate

OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V14 = OUT / "v14"
DEST = V14 / "posthoc-baselines"
EVALUATOR = Path(__file__).with_name("evaluate_calibrated_goal.py")
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
SQUAT = ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt"
REFERENCE = ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"
GEOMETRY = ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json"
CALIBRATION = OUT / "rgb-calibration-model.npz"
ACTORS = {"v9_locked": OUT / "v9/actor.pt",
          "feedback_locked": ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1/controller.pt"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads((V14 / "plan.json").read_text())
    fresh = json.loads((V14 / "fresh-reserved/matrix.json").read_text())
    if fresh["case_count"] != 20 or fresh["plan_sha256"] != sha(V14 / "plan.json"):
        raise ValueError("Complete frozen fresh score required before post hoc diagnosis")
    cases = {case["seed"]: case for case in plan["fresh_reserved_cases"]}
    misses = [row for row in fresh["cases"] if not row["pass"]]
    DEST.mkdir(parents=True, exist_ok=True)
    diagnostic_plan = {
        "schema": "robot-learning-lab.cove50-v14-posthoc-baselines-plan.v1",
        "role": "opened_failure_diagnostic_only",
        "fresh_matrix_sha256": sha(V14 / "fresh-reserved/matrix.json"),
        "evaluator_sha256": sha(EVALUATOR),
        "actors_sha256": {mode: sha(path) for mode, path in ACTORS.items()},
        "cases": [cases[row["seed"]] for row in misses],
        "modes": list(ACTORS),
    }
    (DEST / "plan.json").write_text(json.dumps(diagnostic_plan, indent=2) + "\n")
    rows = []
    for row in misses:
        case = cases[row["seed"]]
        for mode, actor in ACTORS.items():
            stem = f"seed-{case['seed']}-{mode}"
            report = DEST / f"{stem}.json"
            trace = DEST / f"{stem}.npz"
            log = DEST / f"{stem}.log"
            command = [sys.executable, str(EVALUATOR),
                       "--body-xml", str(BODY), "--calibration-model", str(CALIBRATION),
                       "--squat-checkpoint", str(SQUAT), "--reach-checkpoint", str(actor),
                       "--reach-controller", "feedback" if mode == "feedback_locked" else "learned-residual",
                       "--reach-reference", str(REFERENCE),
                       "--geometry", str(GEOMETRY), "--entry", "continuous", "--seed", str(case["seed"]),
                       "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
                       "--toy-offset-x-m", str(case["toy_offset_x_m"]),
                       "--toy-offset-y-m", str(case["toy_offset_y_m"]),
                       "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
                       "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
                       "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
                       "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
                       "--max-goal-age-s", "9", "--out", str(report), "--trace", str(trace)]
            command.append("--lock-goal-after-entry")
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                [str(ROOT), str(ROOT / "experiments/compact_morphology/mjlab_ape/src"),
                 "/workspace/microduck_rl/src"]
                + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
            with log.open("w") as stream:
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, check=False).returncode
            if code or not report.exists():
                result = {"seed": case["seed"], "mode": mode, "pass": False,
                          "faults": [f"evaluator exit {code}"], "log": str(log.relative_to(ROOT))}
            else:
                data = json.loads(report.read_text())
                faults = gate(data)
                result = {"seed": case["seed"], "mode": mode, "pass": not faults,
                          "contact_s": data["phases"]["reach"]["continuous_finger_contact_s"],
                          "faults": faults, "result": str(report.relative_to(ROOT)),
                          "trace": str(trace.relative_to(ROOT)), "log": str(log.relative_to(ROOT))}
            rows.append(result)
            (DEST / "matrix.json").write_text(json.dumps({
                "schema": "robot-learning-lab.cove50-v14-posthoc-baselines-native.v1",
                "role": "opened_failure_diagnostic_only",
                "plan_sha256": sha(DEST / "plan.json"),
                "case_count": len(rows), "pass_count": sum(r["pass"] for r in rows), "cases": rows,
            }, indent=2) + "\n")
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
