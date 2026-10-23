"""Run only frozen opened-development V95/V165 capture parity pairs on CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as native,
)


ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "artifacts/ape-low-level-training-20261015/reach-aligned-v165"
METHOD = RUN / "parity-method.json"
PLAN = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v158/new-seed-contact-plan.json"
DEST = RUN / "cuda-parity"
MATRIX = DEST / "matrix.json"
WRAPPER = Path(__file__).with_name("evaluate_aligned_reach_v165_cli.py")
EVALUATOR = Path(__file__).with_name("evaluate_aligned_reach_v165.py")
VERIFY = Path(__file__).with_name("verify_aligned_reach_v165.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def entry_event(report: dict) -> dict:
    events = [event for event in report["image_goal_capture_events"]
              if event["time_s"] == report["reach_start_s"] and event["used_for_control"]]
    if len(events) != 1:
        raise ValueError("Expected exactly one used RGB goal at reach entry")
    return events[0]


def run_pair(case: dict, method: dict, plan: dict) -> dict:
    seed = case["seed"]
    sources = plan["native_source_sha256"]
    original = native.EVALUATOR
    try:
        native.EVALUATOR = original
        v95 = native.run_case({**case, "id": f"v95-parity-{seed}"},
                              {"source_sha256": sources}, DEST)
        native.EVALUATOR = WRAPPER
        v165 = native.run_case({**case, "id": f"v165-parity-{seed}"},
                               {"source_sha256": {**sources,
                                                  "evaluator": method["v165_evaluator_sha256"]}},
                              DEST)
    finally:
        native.EVALUATOR = original

    old_report = json.loads((ROOT / v95["result"]).read_text())
    new_report = json.loads((ROOT / v165["result"]).read_text())
    expected_report_hashes = {
        "body_sha256": sources["body"],
        "selected_pointer_sha256": sources["pointer"],
        "squat_checkpoint_sha256": sources["squat"],
        "reach_checkpoint_sha256": sources["actor"],
        "alternate_reach_checkpoint_sha256": sources["alternate_actor"],
        "specialist_reach_checkpoint_sha256": sources["specialist_actor"],
        "reach_reference_sha256": sources["reference"],
        "geometry_sha256": sources["geometry"],
        "calibration_model_sha256": sources["calibration"],
        "selector_source_sha256": sources["selector"],
        "template_source_sha256": sources["template"],
    }
    for report in (old_report, new_report):
        if any(report.get(key) != value for key, value in expected_report_hashes.items()):
            raise ValueError(f"V95/V165 auxiliary source changed at seed {seed}")
    aligned_trace = (ROOT / v165["trace"]).with_name(
        Path(v165["trace"]).stem + "-aligned.npz")
    command = [sys.executable, str(VERIFY), "--report", str(ROOT / v165["result"]),
               "--trace", str(ROOT / v165["trace"]),
               "--aligned-trace", str(aligned_trace)]
    check = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    alignment = json.loads(check.stdout)
    write(DEST / f"seed-{seed}-v165-verification.json", alignment)
    with np.load(ROOT / v95["trace"], allow_pickle=False) as before, \
            np.load(ROOT / v165["trace"], allow_pickle=False) as after:
        if before["qpos"].shape != after["qpos"].shape or \
                before["action"].shape != after["action"].shape:
            raise ValueError("V95/V165 full trajectory dimensions differ")
        qpos_gap = float(np.max(np.abs(before["qpos"] - after["qpos"])))
        action_gap = float(np.max(np.abs(before["action"] - after["action"])))
        same_phase = bool(np.array_equal(before["phase"], after["phase"]))
    old_entry, new_entry = entry_event(old_report), entry_event(new_report)
    rgb_goal_gap_mm = float(1000 * np.linalg.norm(
        np.asarray(old_entry["estimated_goal_world_xyz_m"][:2]) -
        np.asarray(new_entry["estimated_goal_world_xyz_m"][:2])))
    contact_gap_s = abs(v95["contact_s"] - v165["contact_s"])
    matched_route = old_report["selection_mode"] == new_report["selection_mode"]
    matched_capture = bool(
        old_entry["template_applied"] == new_entry["template_applied"] and
        old_report["image_goal_used_update_count"] ==
        new_report["image_goal_used_update_count"] and
        old_report["image_goal_lock_after_entry"] ==
        new_report["image_goal_lock_after_entry"] and
        old_report["image_goal_missing_at_s"] ==
        new_report["image_goal_missing_at_s"])
    faults = []
    gate = method["parity_gate"]
    if qpos_gap > gate["max_qpos_abs_gap"]:
        faults.append("full_native_qpos_gap")
    if action_gap > gate["max_applied_action_abs_gap"]:
        faults.append("applied_action_gap")
    if rgb_goal_gap_mm > gate["max_entry_rgb_goal_gap_mm"]:
        faults.append("entry_rgb_goal_gap")
    if contact_gap_s > gate["max_contact_duration_gap_s"]:
        faults.append("contact_duration_gap")
    if not same_phase or not matched_route or not matched_capture:
        faults.append("phase_route_or_capture_change")
    if not v95["pass"] or not v165["pass"] or not alignment["full_contact_margin_pass"]:
        faults.append("full_native_gate_or_margin_failure")
    result = {
        "seed": seed,
        "stratum": case["stratum"],
        "v95_report": v95["result"],
        "v165_report": v165["result"],
        "v95_trace": v95["trace"],
        "v165_trace": v165["trace"],
        "aligned_trace": str(aligned_trace.relative_to(ROOT)),
        "v95_report_sha256": sha(ROOT / v95["result"]),
        "v165_report_sha256": sha(ROOT / v165["result"]),
        "v95_trace_sha256": sha(ROOT / v95["trace"]),
        "v165_trace_sha256": sha(ROOT / v165["trace"]),
        "aligned_trace_sha256": sha(aligned_trace),
        "qpos_max_abs_gap": qpos_gap,
        "action_max_abs_gap": action_gap,
        "entry_rgb_goal_gap_mm": rgb_goal_gap_mm,
        "contact_duration_gap_s": contact_gap_s,
        "selected_route": new_report["selection_mode"],
        "same_phase": same_phase,
        "matched_route": matched_route,
        "matched_capture": matched_capture,
        "v95_full_gate_pass": v95["pass"],
        "v165_full_gate_pass": v165["pass"],
        "v165_source_alignment_pass": alignment["source_alignment_pass"],
        "v165_margin_pass": alignment["full_contact_margin_pass"],
        "v95_contact_s": v95["contact_s"],
        "v165_contact_s": v165["contact_s"],
        "entry_orange_pixel_delta": (new_entry["orange_pixels"] -
                                      old_entry["orange_pixels"]),
        "entry_template_disagreement_delta": (
            new_entry.get("template_pixels_disagree", 0) -
            old_entry.get("template_pixels_disagree", 0)),
        "faults": faults,
        "pass": not faults,
    }
    write(DEST / f"seed-{seed}-parity.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    method = json.loads(METHOD.read_text())
    plan = json.loads(PLAN.read_text())
    expected = method["source_sha256"]
    if (sha(PLAN) != expected["v158_plan"] or
            sha(Path(__file__)) != expected["runner"] or
            sha(WRAPPER) != expected["cli_wrapper"] or
            sha(EVALUATOR) != method["v165_evaluator_sha256"] or
            sha(VERIFY) != expected["verifier"] or
            sha(native.EVALUATOR) != plan["native_source_sha256"]["evaluator"] or
            sha(ROOT / "robot/ape-current-design.json") != expected["pointer"]):
        raise ValueError("Frozen V165 parity source changed")
    source_files = native.v83.v16.SOURCES
    for key in ("body", "squat", "alternate_actor", "specialist_actor",
                "reference", "geometry", "calibration", "camera_freeze",
                "virtual_rgb_observer", "calibrated_rgb_observer", "pointer"):
        if sha(Path(source_files[key])) != plan["native_source_sha256"][key]:
            raise ValueError(f"V158 source closure changed: {key}")
    if sha(Path(native.v83.v57.ACTOR)) != plan["native_source_sha256"]["actor"]:
        raise ValueError("V158 primary reach actor changed")
    chosen = {case["seed"]: case for case in plan["cases"]
              if case["seed"] in method["seeds"] and case["split"] == "development"}
    if set(chosen) != set(method["seeds"]) or args.seeds != method["seeds"]:
        raise ValueError("V165 parity can use only its three opened development seeds")
    if DEST.exists():
        raise FileExistsError("V165 parity destination exists; inspect originals before any new attempt")
    DEST.mkdir(parents=True)
    rows = {}
    for seed in args.seeds:
        rows[seed] = run_pair(chosen[seed], method, plan)
        write(MATRIX, {"schema": "robot-learning-lab.reach-v165-parity-matrix.v1",
                       "method_sha256": sha(METHOD),
                       "planned_seeds": method["seeds"],
                       "completed_count": len(rows),
                       "cases": [rows[s] for s in method["seeds"] if s in rows]})
        print(json.dumps({"seed": seed, "pass": rows[seed]["pass"],
                          "faults": rows[seed]["faults"]}), flush=True)
        if not rows[seed]["pass"]:
            raise ValueError(f"V165 instrumentation parity failed at seed {seed}")


if __name__ == "__main__":
    main()
