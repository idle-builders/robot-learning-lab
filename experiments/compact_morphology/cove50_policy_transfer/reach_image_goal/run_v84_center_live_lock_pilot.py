"""Pilot an observed live-image lock on V83's older authored-route miss."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v83_thin_route_pilot as v83,
)


OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v84"
PLAN = OUT / "center-live-lock-pilot-plan.json"
DEST = OUT / "native-pilot"
EVALUATOR = Path(__file__).with_name("evaluate_center_live_lock_v84.py")
PRIOR = OUT.parent / "v83/prior75-retention-plan.json"
FAILED = OUT.parent / "v83/prior75-pilot-matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    failure = json.loads(FAILED.read_text())
    if (not failure["complete"] or failure["pass_count"] != 9 or
            not any(row["seed"] == 10534 and not row["pass"] for row in failure["cases"])):
        raise ValueError("Need V83's complete 10534 older-start failure")
    parent = json.loads(PRIOR.read_text())
    case = next(row for row in parent["cases"] if row["seed"] == 10534)
    sources = {**parent["source_sha256"], "evaluator": sha(EVALUATOR)}
    plan = {
        "schema": "robot-learning-lab.cove50-v84-center-live-lock-pilot-plan.v1",
        "role": "post_pilot_observable_lock_development_not_retention",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "runner_sha256": sha(Path(__file__)),
        "v83_prior75_plan_sha256": sha(PRIOR),
        "v83_prior75_pilot_matrix_sha256": sha(FAILED),
        "v83_evaluator_sha256": sha(v83.EVALUATOR),
        "case_count": 1, "cases": [case],
        "gate": "On exact older 10534, retain full native contact, support, effort, floor, self-contact, clipping and RGB gates while the new rule activates from a visible >=40-pixel positive-X/negative-Y camera region on the unchanged authored route and locks the image goal at reach entry. The old V83 correction must stay off. A passing single-seed pilot is only an opened diagnostic; 63-route and 75-start retention plus new varied scoring would still be required.",
        "design": "Copy the source-frozen V83 evaluator without changing body, actor, motor, toy or gaze. After its normal observed selector chooses v16_live, lock subsequent image-goal updates only if the commanded route is authored, the observed camera has world X >=0 and Y <= -0.08 m, and the entry RGB image has at least 40 orange pixels. No seed, contact outcome or simulator toy truth chooses the lock.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V84 pilot differs from current source")
    PLAN.write_bytes(encoded)
    return plan


def main() -> None:
    plan = freeze()
    case = plan["cases"][0]
    DEST.mkdir(parents=True, exist_ok=True)
    result, trace, log = (DEST / f"seed-{case['seed']}-center-live-lock.{suffix}"
                          for suffix in ("json", "npz", "log"))
    sources = v83.v16.SOURCES
    command = [sys.executable, str(EVALUATOR),
               "--body-xml", str(sources["body"]),
               "--calibration-model", str(sources["calibration"]),
               "--squat-checkpoint", str(sources["squat"]),
               "--reach-checkpoint", str(v83.v57.ACTOR),
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
            raise RuntimeError(f"V84 native pilot failed: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if any(report.get(key) != value for key, value in {
            "evaluator_sha256": hashes["evaluator"],
            "selector_source_sha256": hashes["selector"],
            "template_source_sha256": hashes["template"],
            "body_sha256": hashes["body"],
            "selected_pointer_sha256": hashes["pointer"],
            "reach_checkpoint_sha256": hashes["actor"],
            "adaptive_variant": "v84_observed_center_live_lock_candidate",
            "pitch_offset_rad": case["pitch_offset_rad"],
            "roll_clearance_rad": case["roll_clearance_rad"],
    }.items()):
        raise ValueError(f"V84 source or route identity mismatch: {result}")
    faults = v83.gate(report)
    entry = next((event for event in report["image_goal_capture_events"]
                  if event["time_s"] == report["reach_start_s"]), None)
    if not entry or not entry.get("used_for_control") or not entry.get("template_applied"):
        faults.append("image template unavailable at reach entry")
    if entry and (entry.get("xy_error_truth_mm_scoring_only") is None or
                  entry["xy_error_truth_mm_scoring_only"] > 10):
        faults.append("RGB goal error above 10 mm")
    if report["image_goal_missing_at_s"] is not None:
        faults.append("image goal missing during reach")
    if (not report["automatic_center_live_lock_applied"] or
            report["selection_mode"] != "v16_locked" or
            not report["image_goal_lock_after_entry"]):
        faults.append("observed center live-lock rule did not activate")
    if report["automatic_thin_route_bias_applied"]:
        faults.append("V83 off-author thin-route rule activated at center")
    contact = report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0)
    row = {
        "id": case["id"], "seed": case["seed"],
        "v41_contact_s": case["v73_contact_s"],
        "v83_contact_s": next(item["contact_s"] for item in
                              json.loads(FAILED.read_text())["cases"] if item["seed"] == case["seed"]),
        "contact_s": contact,
        "entry_orange_pixels": None if entry is None else entry["orange_pixels"],
        "entry_goal_error_mm_scoring_only": (None if entry is None else
                                             entry.get("xy_error_truth_mm_scoring_only")),
        "automatic_center_live_lock_applied": report["automatic_center_live_lock_applied"],
        "pass": not faults, "faults": faults,
        "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
    }
    matrix = {
        "schema": "robot-learning-lab.cove50-v84-center-live-lock-pilot.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "complete": True,
        "case_count": 1, "pass_count": int(row["pass"]),
        "pilot_gate_met": row["pass"] and contact >= 1.2,
        "cases": [row],
    }
    (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({"pass": row["pass"], "contact_s": contact,
                      "lock_used": row["automatic_center_live_lock_applied"],
                      "faults": faults}, indent=2))


if __name__ == "__main__":
    main()
