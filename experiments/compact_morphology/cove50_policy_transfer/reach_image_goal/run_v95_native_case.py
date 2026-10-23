"""Fresh native V95 two-region observed reach rule with full V84 gates."""

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


EVALUATOR = Path(__file__).with_name("evaluate_rear_two_region_v95.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(case: dict, plan: dict, dest: Path) -> dict:
    ident, seed = case["id"], case["seed"]
    result, trace, log = (dest / f"seed-{seed}-{ident}.{suffix}"
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
               "--entry", "continuous", "--seed", str(seed),
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
            raise RuntimeError(f"V95 native replay failed for {ident}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if any(report.get(key) != value for key, value in {
            "evaluator_sha256": hashes["evaluator"],
            "selector_source_sha256": hashes["selector"],
            "template_source_sha256": hashes["template"],
            "body_sha256": hashes["body"],
            "selected_pointer_sha256": hashes["pointer"],
            "reach_checkpoint_sha256": hashes["actor"],
            "adaptive_variant": "v95_observed_rear_two_region",
            "pitch_offset_rad": case["pitch_offset_rad"],
            "roll_clearance_rad": case["roll_clearance_rad"],
    }.items()):
        raise ValueError(f"V95 source or route identity mismatch: {result}")
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
    if report["reacquisition_probe_orange_pixels"] == 0 and not (
            report["reacquisition_trigger_s"] is not None and
            report["reacquisition_head_yaw_rad"] in (-1.3, -1.5) and
            entry and entry.get("orange_pixels", 0) >= 10):
        faults.append("physical camera reacquisition did not observe toy")
    expected_lock = case.get("expected_center_live_lock")
    if expected_lock is not None and report["automatic_center_live_lock_applied"] != expected_lock:
        faults.append("center live-lock exposure differs from frozen expectation")
    if (report["automatic_center_live_lock_applied"] and
            (report["selection_mode"] != "v16_locked" or
             not report["image_goal_lock_after_entry"] or
             report["off_authored_route_probe"])):
        faults.append("center live-lock branch identity mismatch")
    if case.get("expected_thin_route_bias") is not None and (
            report["automatic_thin_route_bias_applied"] != case["expected_thin_route_bias"]):
        faults.append("thin-route rule exposure differs from frozen expectation")
    if case.get("expected_rear_contact_bias") is not None and (
            report["automatic_rear_contact_bias_applied"] != case["expected_rear_contact_bias"]):
        faults.append("rear-contact rule exposure differs from frozen expectation")
    if case.get("expected_alternate_gaze_specialist") is not None and (
            report["automatic_alternate_gaze_specialist_applied"] !=
            case["expected_alternate_gaze_specialist"]):
        faults.append("alternate-gaze specialist exposure differs from frozen expectation")
    if case.get("expected_selection_mode") is not None and (
            report["selection_mode"] != case["expected_selection_mode"]):
        faults.append("selection mode differs from frozen expectation")
    if case["route_variant"] == "center" and (
            report["off_authored_route_probe"] or report["automatic_thin_route_bias_applied"] or
            case["pitch_offset_rad"] != .105 or case["roll_clearance_rad"] != -.25):
        faults.append("authored center route activated off-author controller")
    contact = report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0)
    return {
        "id": ident, "seed": seed, "route_variant": case["route_variant"],
        "baseline_contact_s": case.get("baseline_contact_s", case.get("v73_contact_s")),
        "selection_mode": report["selection_mode"],
        "expected_center_live_lock": expected_lock,
        "automatic_center_live_lock_applied": report["automatic_center_live_lock_applied"],
        "expected_thin_route_bias": case.get("expected_thin_route_bias"),
        "automatic_thin_route_bias_applied": report["automatic_thin_route_bias_applied"],
        "automatic_thin_route_region": report["automatic_thin_route_region"],
        "expected_rear_contact_bias": case.get("expected_rear_contact_bias"),
        "automatic_rear_contact_bias_applied": report["automatic_rear_contact_bias_applied"],
        "expected_alternate_gaze_specialist": case.get("expected_alternate_gaze_specialist"),
        "automatic_alternate_gaze_specialist_applied": report["automatic_alternate_gaze_specialist_applied"],
        "entry_orange_pixels": None if entry is None else entry["orange_pixels"],
        "entry_goal_error_mm_scoring_only": (None if entry is None else
                                             entry.get("xy_error_truth_mm_scoring_only")),
        "contact_s": contact, "margin_1p2_s_met": contact >= 1.2,
        "pass": not faults, "faults": faults,
        "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
    }
