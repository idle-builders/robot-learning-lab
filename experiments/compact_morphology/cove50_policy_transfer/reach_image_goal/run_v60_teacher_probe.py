"""Check whether the saved V10 arm can rescue one opened off-route V57 miss."""

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
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v16_opened_matrix as v16,
    run_v60_route_probe as v60,
)

OUT = v60.OUT / "teacher-probe"
PLAN = OUT / "plan.json"
CASE_ID = "positive-10-pitch-high"
TEACHER = v60.BASE / "v10/actor.pt"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    candidate = json.loads(v60.PLAN.read_text())
    case = next(row for row in candidate["cases"] if row["id"] == CASE_ID)
    failed = v60.DEST / f"seed-{case['seed']}-{CASE_ID}.json"
    failure = json.loads(failed.read_text())
    if failure["phases"]["reach"]["continuous_finger_contact_s"] >= 1:
        raise ValueError("Teacher probe requires the exact failed V60 off-route entry")
    plan = {
        "schema": "robot-learning-lab.cove50-v60-v10-teacher-probe-plan.v1",
        "role": "opened_off_route_saved_actor_diagnostic_not_trained_controller",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "case": case,
        "v60_plan_sha256": sha(v60.PLAN),
        "v60_failure_report_sha256": sha(failed),
        "evaluator_sha256": sha(v60.EVALUATOR),
        "teacher_actor_sha256": sha(TEACHER),
        "runner_sha256": sha(Path(__file__)),
        "gate": "The saved V10 arm must pass the same full native support, motor, >=1 s contact, hand-floor, self-contact, clipping and <=10 mm RGB gates at positive-10 pitch-high. This is a source-separate teacher diagnostic; no retuning or route qualification follows from a single saved-actor replay.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V60 teacher plan differs from source")
    PLAN.write_bytes(encoded)
    result = OUT / f"seed-{case['seed']}-{CASE_ID}-v10.json"
    trace = result.with_suffix(".npz")
    log = result.with_suffix(".log")
    sources = v16.SOURCES
    command = [sys.executable, str(v60.EVALUATOR),
               "--body-xml", str(sources["body"]),
               "--calibration-model", str(sources["calibration"]),
               "--squat-checkpoint", str(sources["squat"]),
               "--reach-checkpoint", str(TEACHER),
               "--alternate-reach-checkpoint", str(sources["alternate_actor"]),
               "--specialist-reach-checkpoint", str(sources["specialist_actor"]),
               "--reach-controller", "learned-residual",
               "--reach-reference", str(sources["reference"]),
               "--geometry", str(sources["geometry"]),
               "--entry", "continuous", "--seed", str(case["seed"]),
               "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
               "--toy-offset-x-m", str(case["toy_offset_x_m"]),
               "--toy-offset-y-m", str(case["toy_offset_y_m"]),
               "--pitch-offset-rad", str(case["pitch_offset_rad"]),
               "--roll-clearance-rad", str(case["roll_clearance_rad"]),
               "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
               "--reach-duration-s", "9", "--gaze-head-yaw-rad", str(case["gaze_head_yaw_rad"]),
               "--gaze-start-s", "0", "--gaze-ramp-s", "7", "--capture-stride-steps", "5",
               "--max-goal-age-s", "9", "--out", str(result), "--trace", str(trace)]
    if not result.exists() or not trace.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), str(ROOT / "experiments/compact_morphology/mjlab_ape/src"),
             "/root/microduck_rl/src", "/workspace/microduck_rl/src"]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        with log.open("w") as stream:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, check=False).returncode
        if code:
            raise RuntimeError(f"V60 teacher evaluator failed: {log}")
    report = json.loads(result.read_text())
    if (report["reach_checkpoint_sha256"] != plan["teacher_actor_sha256"] or
            report["evaluator_sha256"] != plan["evaluator_sha256"] or
            not report["off_authored_route_probe"]):
        raise ValueError("Saved teacher or off-route source identity mismatch")
    faults = gate(report)
    entry = next((event for event in report["image_goal_capture_events"]
                  if event.get("time_s") == report["reach_start_s"]), None)
    error = None if entry is None else entry.get("xy_error_truth_mm_scoring_only")
    if error is None or error > 10:
        faults.append("RGB goal error above 10 mm or unavailable")
    if report.get("image_goal_missing_at_s") is not None:
        faults.append("image goal missing during reach")
    verdict = {"schema": "robot-learning-lab.cove50-v60-v10-teacher-probe.v1",
               "role": plan["role"], "body_role": plan["body_role"],
               "plan_sha256": sha(PLAN), "id": CASE_ID, "seed": case["seed"],
               "source_v57_contact_s": failure["phases"]["reach"]["continuous_finger_contact_s"],
               "teacher_contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
               "rgb_error_mm": error, "pass": not faults, "faults": faults,
               "result": str(result.relative_to(ROOT)),
               "trace": str(trace.relative_to(ROOT)), "log": str(log.relative_to(ROOT))}
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict), flush=True)


if __name__ == "__main__":
    main()
