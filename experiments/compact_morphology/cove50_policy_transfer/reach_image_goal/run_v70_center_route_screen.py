"""Screen the observed center-route correction with the roll-aware rule."""

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
    run_v57_contact_actor_screen as v57,
    run_v60_route_probe as v60,
    run_v61_ybias_probe as v61,
    run_v61_roll_ybias_probe as v61_roll,
    run_v61_edge_ybias_probe as v61_edge,
    run_v62_route_rule_screen as v62,
    run_v64_positive_roll_ybias_probe as v64,
    run_v65_route_rule_screen as v65,
    run_v69_center_ybias_probe as v69,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v70"
PLAN = OUT / "center-and-roll-route-rule-plan.json"
DEST = OUT / "opened-center-and-roll-route-rule"
EVALUATOR = Path(__file__).with_name("evaluate_route_rule_v70.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v58.py")
ACTOR = v57.ACTOR
PILOT_IDS = ("positive-10-roll-high", "positive-10-pitch-high", "positive-10-center",
             "large-20-roll-high", "large-20-pitch-high",
             "edge-12-pitch-low", "edge-12-roll-low", "edge-12-roll-high")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases() -> list[dict]:
    frozen = json.loads(v60.PLAN.read_text())["cases"]
    if len(frozen) != 35 or len({case["id"] for case in frozen}) != 35:
        raise ValueError("Expected 35 exact frozen V60 route starts")
    return frozen


def freeze() -> dict:
    predecessor = v60.DEST / "matrix.json"
    prior = json.loads(predecessor.read_text())
    if not prior["complete"] or prior["case_count"] != 35 or prior["pass_count"] >= 35:
        raise ValueError("V70 requires a complete V60 route screen with observed failures")
    v62_path = v62.DEST / "matrix.json"
    v62_screen = json.loads(v62_path.read_text())
    by_v62 = {row["id"]: row for row in v62_screen["cases"]}
    if (not v62_screen["complete"] or v62_screen["case_count"] != 35 or
            v62_screen["screen_gate_met"] or by_v62["positive-10-roll-high"]["pass"] or
            by_v62["positive-10-roll-high"]["contact_s"] >= 1):
        raise ValueError("V70 requires the complete failed V62 route screen")
    center_v62 = by_v62["positive-10-center"]
    if center_v62["pass"] or center_v62["contact_s"] >= 1:
        raise ValueError("V70 requires the V62 center-route contact regression")
    pilot_path = v65.OUT / "staged-pilot-matrix.json"
    pilot = json.loads(pilot_path.read_text())
    by_pilot = {row["id"]: row for row in pilot["cases"]}
    if (not pilot["complete"] or pilot["case_count"] != 8 or pilot["pilot_gate_met"] or
            by_pilot["positive-10-center"]["pass"] or
            not by_pilot["positive-10-roll-high"]["pass"]):
        raise ValueError("V70 requires the complete failed V65 pilot and recovered roll case")
    center_path = v69.DEST / "matrix.json"
    center_teacher = json.loads(center_path.read_text())
    center_minus4 = next((row for row in center_teacher["cases"]
                          if row["diagnostic_y_bias_m"] == -.004), None)
    if (not center_teacher["complete"] or center_teacher["case_count"] != 7 or
            not center_minus4 or not center_minus4["pass"] or
            center_minus4["contact_s"] < 1.2):
        raise ValueError("V70 requires a full-gate -4 mm center-route teacher")
    positive_path = v64.DEST / "matrix.json"
    positive_teacher = json.loads(positive_path.read_text())
    minus6 = next((row for row in positive_teacher["cases"]
                   if row["diagnostic_y_bias_m"] == -.006), None)
    if (not positive_teacher["complete"] or not minus6 or not minus6["pass"] or
            minus6["contact_s"] < 1.2):
        raise ValueError("V70 requires a full-gate -6 mm positive-roll teacher")
    teacher_path = v61.DEST / "matrix.json"
    teacher = json.loads(teacher_path.read_text())
    minus4 = next((row for row in teacher["cases"]
                   if row["diagnostic_y_bias_m"] == -.004), None)
    if not teacher["complete"] or not teacher["teacher_found"] or not minus4 or not minus4["pass"]:
        raise ValueError("V70 requires a passing frozen -4 mm V61 prescribed teacher")
    roll_path = v61_roll.DEST / "matrix.json"
    edge_path = v61_edge.DEST / "matrix.json"
    roll_teacher = json.loads(roll_path.read_text())
    edge_teacher = json.loads(edge_path.read_text())
    roll_minus4 = next((row for row in roll_teacher["cases"]
                        if row["diagnostic_y_bias_m"] == -.004), None)
    edge_minus4 = [row for row in edge_teacher["cases"]
                   if row["diagnostic_y_bias_m"] == -.004]
    if (not roll_teacher["complete"] or not roll_minus4 or not roll_minus4["pass"] or
            roll_minus4["contact_s"] < 1.2 or not edge_teacher["complete"] or
            len(edge_minus4) != 3 or not all(row["pass"] and row["contact_s"] >= 1.2
                                                 for row in edge_minus4)):
        raise ValueError("V70 requires passing -4 mm roll and all three edge-route teachers")
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE))
    varied = cases()
    by_id = {row["id"]: row for row in prior["cases"]}
    for case in varied:
        case["baseline_v60_contact_s"] = by_id[case["id"]]["contact_s"]
        case["baseline_v60_pass"] = by_id[case["id"]]["pass"]
        case["baseline_v62_contact_s"] = by_v62[case["id"]]["contact_s"]
        case["baseline_v62_pass"] = by_v62[case["id"]]["pass"]
    plan = {
        "schema": "robot-learning-lab.cove50-v70-center-and-roll-route-image-goal-plan.v1",
        "role": "opened_observed_route_rule_development_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "v60_route_plan_sha256": sha(v60.PLAN),
        "v60_route_matrix_sha256": sha(predecessor),
        "v62_route_plan_sha256": sha(v62.PLAN),
        "v62_route_matrix_sha256": sha(v62_path),
        "v65_pilot_plan_sha256": sha(v65.PLAN),
        "v65_failed_pilot_matrix_sha256": sha(pilot_path),
        "v69_prescribed_center_plan_sha256": sha(v69.PLAN),
        "v69_prescribed_center_matrix_sha256": sha(center_path),
        "v64_positive_roll_plan_sha256": sha(v64.PLAN),
        "v64_positive_roll_matrix_sha256": sha(positive_path),
        "v61_prescribed_plan_sha256": sha(v61.PLAN),
        "v61_prescribed_matrix_sha256": sha(teacher_path),
        "v61_roll_plan_sha256": sha(v61_roll.PLAN),
        "v61_roll_matrix_sha256": sha(roll_path),
        "v61_edge_plan_sha256": sha(v61_edge.PLAN),
        "v61_edge_matrix_sha256": sha(edge_path),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(varied),
        "cases": varied,
        "pilot_ids": PILOT_IDS,
        "pilot_gate": "The same eight predeclared native pilot visits used by V65, including all six V60 route failures and the center route, must pass the full native gate before the full 35-route matrix.",
        "gate": "All 35 exact V60 opened anchor-by-route native replays must pass support, rated effort, >=1 s continuous fingertip contact, hand floor, self-contact, clipping and <=10 mm scored RGB gates. Require >=2 front-route corrections including >=2 stronger roll corrections, >=1 observable front-center correction, >=3 edge-route corrections, >=5 negative-camera corrections, >=5 positive sparse abstentions, >=5 earlier edge bias uses and >=5 physical zero-pixel scans. Report sub-1.2 s contacts as thin. This tuned-on-V60 replay is development, not fresh score or new-body qualification.",
        "design": "Retain V65's trained arm, camera selector, -6 mm stronger raised-roll and -4 mm raised-pitch/sparse-edge corrections. On the authored center route only, an observed front view with negative camera X, at least 48 orange pixels and RGB template correction Y at most -4 mm gets the V69 passing -4 mm body-Y goal offset. This is an entry-view rule; seed, simulator toy truth and scored contact do not select it. Recheck all 35 opened route visits, then different-anchor and branch-view holdouts and fresh scoring.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V70 center-and-roll plan differs from source")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    seed = case["seed"]
    ident = case["id"]
    result, trace, log = (DEST / f"seed-{seed}-{ident}.{suffix}"
                          for suffix in ("json", "npz", "log"))
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
            raise RuntimeError(f"V70 center-and-roll route replay failed for {ident}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if any(report.get(key) != value for key, value in {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v70_center_and_roll_aware_observed_image_goal_candidate",
    }.items()):
        raise ValueError(f"V70 source identity mismatch: {result}")
    if (report["pitch_offset_rad"] != case["pitch_offset_rad"] or
            report["roll_clearance_rad"] != case["roll_clearance_rad"] or
            report["off_authored_route_probe"] != (case["route_variant"] != "center")):
        raise ValueError(f"V70 route identity mismatch: {result}")
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
        pitch_delta = case["pitch_offset_rad"] - .105
        roll_delta = case["roll_clearance_rad"] + .25
        expected_front = bool(report["selection_mode"] == "v16_locked" and
                              camera[1] >= .08 and entry["orange_pixels"] >= 48 and
                              (pitch_delta >= .005 or roll_delta >= .005))
        expected_front_center = bool(
            report["selection_mode"] == "v16_locked" and
            camera[0] < 0 and camera[1] >= .08 and
            entry["orange_pixels"] >= 48 and correction[1] <= -4 and
            abs(pitch_delta) < .005 and abs(roll_delta) < .005)
        expected_edge_route = bool(report["selection_mode"] == "v21_scoped_y_locked" and
                                   camera[0] <= -.06 and camera[1] <= -.08 and
                                   entry["orange_pixels"] <= 26 and
                                   (pitch_delta <= -.005 or abs(roll_delta) >= .005))
        if report.get("automatic_front_route_bias_applied") != expected_front:
            faults.append("observable front-route goal-bias rule mismatch")
        if report.get("automatic_front_center_bias_applied") != expected_front_center:
            faults.append("observable front-center goal-bias rule mismatch")
        expected_front_bias = (-.006 if roll_delta >= .005 else -.004) if expected_front else 0.0
        if report.get("automatic_front_route_bias_y_m") != expected_front_bias:
            faults.append("observable front-route bias magnitude mismatch")
        if report.get("automatic_edge_route_bias_applied") != expected_edge_route:
            faults.append("observable edge-route goal-bias rule mismatch")
        expected_bias = (-.004 if expected_edge else -.002 if report["selection_mode"] in
                         ("v16_bias_y_locked", "v21_scoped_y_locked") else 0.0)
        expected_bias += expected_front_bias
        expected_bias += -.004 if expected_front_center else 0.0
        expected_bias += -.004 if expected_edge_route else 0.0
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
    return {"id": ident, "anchor_id": case["anchor_id"],
            "route_variant": case["route_variant"],
            "pitch_offset_rad": case["pitch_offset_rad"],
            "roll_clearance_rad": case["roll_clearance_rad"],
            "off_authored_route_probe": report["off_authored_route_probe"],
            "cohort": case["cohort"],
            "seed": seed, "development_role": case["development_role"],
            "matched_prior_contact_s": case["matched_prior_contact_s"],
            "matched_prior_pass": case["matched_prior_pass"],
            "baseline_v60_contact_s": case.get("baseline_v60_contact_s"),
            "baseline_v60_pass": case.get("baseline_v60_pass"),
            "baseline_v62_contact_s": case.get("baseline_v62_contact_s"),
            "baseline_v62_pass": case.get("baseline_v62_pass"),
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
            "automatic_front_route_bias_applied": report["automatic_front_route_bias_applied"],
            "automatic_front_route_bias_y_m": report["automatic_front_route_bias_y_m"],
            "automatic_front_center_bias_applied": report["automatic_front_center_bias_applied"],
            "automatic_edge_route_bias_applied": report["automatic_edge_route_bias_applied"],
            "reach_start_s": entry_time,
            "entry_orange_pixels": None if entry is None else entry.get("orange_pixels"),
            "entry_goal_error_mm_scoring_only": entry_error,
            "faults": faults,
            "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
            "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    DEST.mkdir(parents=True, exist_ok=True)
    selected = ([case for case in plan["cases"] if case["id"] in PILOT_IDS]
                if args.pilot else plan["cases"])
    if args.pilot and len(selected) != len(PILOT_IDS):
        raise ValueError("Pilot route identities missing")
    matrix_path = OUT / "staged-pilot-matrix.json" if args.pilot else DEST / "matrix.json"
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Recheck every V62 failure first, then measure all prior passing routes.
        execution_order = sorted(selected, key=lambda case: case["baseline_v62_pass"])
        futures = {pool.submit(run_case, case, plan): case["id"] for case in execution_order}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[c["id"]] for c in selected if c["id"] in rows]
            negative_uses = sum(r["negative_camera_sparse_partial_correction"] for r in ordered)
            positive_uses = sum(r["positive_camera_large_sparse_abstention"] for r in ordered)
            edge_uses = sum(r["edge_goal_bias_applied"] for r in ordered)
            front_route_uses = sum(r["automatic_front_route_bias_applied"] for r in ordered)
            front_center_uses = sum(r["automatic_front_center_bias_applied"] for r in ordered)
            edge_route_uses = sum(r["automatic_edge_route_bias_applied"] for r in ordered)
            physical_scans = sum(r["reacquisition_probe_orange_pixels"] == 0 and
                                 r["reacquisition_trigger_s"] is not None for r in ordered)
            roll_front_uses = sum(r["automatic_front_route_bias_y_m"] == -.006 for r in ordered)
            matrix = {"schema": ("robot-learning-lab.cove50-v70-center-and-roll-route-pilot.v1"
                                  if args.pilot else
                                  "robot-learning-lab.cove50-v70-center-and-roll-route-image-goal-screen.v1"),
                      "role": "opened_observed_route_rule_development_not_fresh_score",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == len(selected),
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "negative_partial_correction_count": negative_uses,
                      "positive_sparse_abstention_count": positive_uses,
                      "edge_bias_use_count": edge_uses,
                      "front_route_bias_use_count": front_route_uses,
                      "front_center_bias_use_count": front_center_uses,
                      "roll_front_bias_use_count": roll_front_uses,
                      "edge_route_bias_use_count": edge_route_uses,
                      "physical_scan_count": physical_scans,
                      "pilot_gate_met": args.pilot and len(ordered) == len(selected) and
                                        all(r["pass"] for r in ordered),
                      "screen_gate_met": (not args.pilot and len(ordered) == len(plan["cases"]) and
                                          all(r["pass"] for r in ordered) and negative_uses >= 5 and
                                          positive_uses >= 5 and edge_uses >= 5 and physical_scans >= 5 and
                                          front_route_uses >= 2 and edge_route_uses >= 3 and
                                          roll_front_uses >= 2 and front_center_uses >= 1),
                      "cases": ordered}
            matrix_path.write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "seed", "selection_mode", "pass", "contact_s", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
