"""Retain every V41 older opened native pass under the V59 combination."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    run_v41_opened_screen as screen,
    run_v41_opened_gate as opened,
    run_v57_contact_actor_screen as v57,
    run_v59_combined_screen as v59,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v59"
PLAN = OUT / "older-retention-plan.json"
DEST = OUT / "older-native-retention"
EVALUATOR = Path(__file__).with_name("evaluate_combined_v59.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v58.py")
ACTOR = v57.ACTOR


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases() -> list[dict]:
    frozen = json.loads(opened.PLAN.read_text())["cases"]
    if len(frozen) != 75 or len({case["seed"] for case in frozen}) != 75:
        raise ValueError("Expected 75 unique frozen older V41 starts")
    return [{**case, "id": f"older-{case['seed']}", "cohort": "older",
             "development_role": "v41_opened_native_retention",
             "pitch_offset_rad": .105, "roll_clearance_rad": -.25}
            for case in frozen]


def freeze() -> dict:
    predecessor = v59.DEST / "matrix.json"
    current = json.loads(predecessor.read_text())
    earlier = opened.DEST / "matrix.json"
    prior = json.loads(earlier.read_text())
    if (not current["complete"] or not current["screen_gate_met"] or
            current["pass_count"] != 63 or not prior["complete"] or
            prior["case_count"] != 75 or prior["pass_count"] != 75):
        raise ValueError("Older retention requires complete V41 and passing V59 native gates")
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE))
    sig_keys = ("seed", "rail_v", "delay_ms", "toy_offset_x_m",
                "toy_offset_y_m", "gaze_head_yaw_rad", "pitch_offset_rad",
                "roll_clearance_rad")
    input_cases = {case["id"]: case for case in json.loads(v59.PLAN.read_text())["cases"]}
    reuses = {tuple(input_cases[row["id"]][key] for key in sig_keys): row
              for row in current["cases"]}
    prior_by_seed = {row["seed"]: row for row in prior["cases"]}
    paired_cases = cases()
    for case in paired_cases:
        matched = prior_by_seed[case["seed"]]
        case["matched_prior_contact_s"] = matched["contact_s"]
        case["matched_prior_pass"] = matched["pass"]
        case["baseline_v41_contact_s"] = matched["contact_s"]
        reuse = reuses.get(tuple(case[key] for key in sig_keys))
        case["reuse_v59_result"] = None if reuse is None else reuse["result"]
        case["reuse_v59_trace"] = None if reuse is None else reuse["trace"]
        case["reuse_v59_log"] = None if reuse is None else reuse["log"]
    if sum(case["reuse_v59_result"] is not None for case in paired_cases) != 6:
        raise ValueError("Expected six exact older/V59 native report matches")
    plan = {
        "schema": "robot-learning-lab.cove50-v59-older-retention-plan.v1",
        "role": "opened_older_native_retention_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "v41_older_plan_sha256": sha(opened.PLAN),
        "v41_older_matrix_sha256": sha(earlier),
        "v59_combined_plan_sha256": sha(v59.PLAN),
        "v59_combined_matrix_sha256": sha(predecessor),
        "runner_sha256": sha(Path(__file__)),
        "case_count": 75,
        "cases": paired_cases,
        "gate": "All 75 exact V41 older opened passing starts must remain full-gate native passes under the V59 automatic combined controller, including support, rated motor effort, >=1 s uninterrupted fingertip contact, hand-floor, self-contact, clipping and <=10 mm scored RGB error. Exact V59 full native traces may be reused only when seed, voltage, delay, toy offsets, gaze and authored route are identical. Record every reused identity. This is older retention, not fresh scoring or selected-body qualification.",
        "design": "Replay the complete 75-start V41 opened retention plan with the V59 frozen trained arm and observed camera rules. Preserve body, motors, toy, gaze and route for each older start. Six exact starts are already in the source-matched 63-start V59 screen and can reuse those full native reports and traces. All other starts run natively; any prior pass regression blocks advance toward the sealed 109xx score.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V59 older-retention plan differs from source")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    seed = case["seed"]
    ident = case["id"]
    result, trace, log = (DEST / f"seed-{seed}-{ident}.{suffix}"
                          for suffix in ("json", "npz", "log"))
    reused = case["reuse_v59_result"] is not None
    if reused:
        result = ROOT / case["reuse_v59_result"]
        trace = ROOT / case["reuse_v59_trace"]
        log = ROOT / case["reuse_v59_log"]
        if not all(path.is_file() for path in (result, trace, log)):
            raise ValueError(f"Expected exact reused V59 evidence for {ident}")
    sources = v16.SOURCES
    command = [sys.executable, str(EVALUATOR),
               "--body-xml", str(sources["body"]),
               "--calibration-model", str(sources["calibration"]),
               "--squat-checkpoint", str(sources["squat"]),
               "--reach-checkpoint", str(ACTOR),
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
            raise RuntimeError(f"V59 older retention evaluator failed for {ident}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if any(report.get(key) != value for key, value in {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v59_trained_arm_observed_camera_and_edge_candidate",
    }.items()):
        raise ValueError(f"V59 older retention source identity mismatch: {result}")
    faults = gate(report)
    entry_time = report["reach_start_s"]
    entry = next((e for e in report["image_goal_capture_events"]
                  if e.get("time_s") == entry_time), None)
    if not entry or not entry.get("used_for_control") or not entry.get("template_applied"):
        faults.append("image template unavailable at reach entry")
    if entry:
        expected_negative = bool(screen.expected_template(entry) and
                                24 <= entry["orange_pixels"] < 64 and
                                entry["camera_world_xyz_m"][0] < -.05 and
                                entry["camera_world_xyz_m"][1] < 0)
        correction = entry.get("template_correction_xy_mm") or [0, 0]
        expected_positive = bool(screen.expected_template(entry) and not expected_negative and
                                 24 <= entry["orange_pixels"] < 64 and
                                 entry["camera_world_xyz_m"][0] >= 0 and
                                 entry["camera_world_xyz_m"][1] < -.08 and
                                 (correction[0] ** 2 + correction[1] ** 2) >= 36)
        correction_magnitude = (correction[0] ** 2 + correction[1] ** 2) ** .5
        expected_partial = expected_negative and correction_magnitude >= 9.0
        if entry.get("negative_camera_sparse_abstention") != expected_negative:
            faults.append("observable sparse-quadrant rule mismatch")
        if entry.get("negative_camera_sparse_partial_correction") != expected_partial:
            faults.append("observable negative sparse partial-correction rule mismatch")
        if entry.get("positive_camera_large_sparse_abstention") != expected_positive:
            faults.append("observable large sparse-correction rule mismatch")
        if entry.get("template_used_for_control") != (
                screen.expected_template(entry) and (not expected_negative or expected_partial)
                and not expected_positive):
            faults.append("observable template rule mismatch")
        camera = entry["camera_world_xyz_m"]
        expected_edge = bool(
            report["selection_mode"] == "v21_scoped_y_locked" and
            24 <= entry["orange_pixels"] <= 50 and
            -.055 <= camera[0] <= -.040 and
            -.095 <= camera[1] <= -.085)
        if report.get("edge_goal_bias_applied") != expected_edge:
            faults.append("observable edge goal-bias rule mismatch")
        expected_bias = (-.004 if expected_edge else -.002 if report["selection_mode"] in
                         ("v16_bias_y_locked", "v21_scoped_y_locked") else 0.0)
        if report.get("goal_bias_body_xy_m") != [0.0, expected_bias]:
            faults.append("goal bias differs from observed entry rule")
    entry_error = None if entry is None else entry.get("xy_error_truth_mm_scoring_only")
    if entry_error is None or entry_error > 10:
        faults.append("RGB goal error above 10 mm or unavailable")
    if report.get("image_goal_missing_at_s") is not None:
        faults.append("image goal missing during reach")
    if report["reacquisition_probe_orange_pixels"] == 0 and not (
            report["reacquisition_trigger_s"] is not None and
            report["reacquisition_head_yaw_rad"] in (-1.3, -1.5) and
            entry and entry.get("orange_pixels", 0) >= 10):
        faults.append("physical camera reacquisition did not observe toy")
    return {"id": ident, "cohort": case["cohort"],
            "seed": seed, "development_role": case["development_role"],
            "reused_v59_native_report": reused,
            "matched_prior_contact_s": case["matched_prior_contact_s"],
            "matched_prior_pass": case["matched_prior_pass"],
            "baseline_v41_contact_s": case["baseline_v41_contact_s"],
            "selection_mode": report.get("selection_mode"), "pass": not faults,
            "contact_s": report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0),
            "reacquisition_trigger_s": report["reacquisition_trigger_s"],
            "reacquisition_probe_orange_pixels": report["reacquisition_probe_orange_pixels"],
            "reacquisition_head_yaw_rad": report["reacquisition_head_yaw_rad"],
            "negative_camera_sparse_abstention": report["negative_camera_sparse_abstention_entry"],
            "negative_camera_sparse_partial_correction": report["negative_camera_sparse_partial_correction_entry"],
            "positive_camera_large_sparse_abstention": report["positive_camera_large_sparse_abstention_entry"],
            "edge_goal_bias_applied": report["edge_goal_bias_applied"],
            "reach_start_s": entry_time,
            "entry_orange_pixels": None if entry is None else entry.get("orange_pixels"),
            "entry_goal_error_mm_scoring_only": entry_error,
            "faults": faults,
            "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
            "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case["id"] for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[c["id"]] for c in plan["cases"] if c["id"] in rows]
            negative_uses = sum(r["negative_camera_sparse_partial_correction"] for r in ordered)
            positive_uses = sum(r["positive_camera_large_sparse_abstention"] for r in ordered)
            edge_uses = sum(r["edge_goal_bias_applied"] for r in ordered)
            physical_scans = sum(r["reacquisition_probe_orange_pixels"] == 0 and
                                 r["reacquisition_trigger_s"] is not None for r in ordered)
            matrix = {"schema": "robot-learning-lab.cove50-v59-older-retention.v1",
                      "role": "opened_older_native_retention_not_fresh_score",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "negative_partial_correction_count": negative_uses,
                      "positive_sparse_abstention_count": positive_uses,
                      "edge_bias_use_count": edge_uses,
                      "physical_scan_count": physical_scans,
                      "reuse_count": sum(r["reused_v59_native_report"] for r in ordered),
                      "retention_gate_met": len(ordered) == 75 and all(r["pass"] for r in ordered),
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "seed", "selection_mode", "pass", "contact_s", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
