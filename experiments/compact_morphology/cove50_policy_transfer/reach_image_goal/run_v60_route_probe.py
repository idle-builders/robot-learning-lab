"""Source-frozen nearby route probe before touching the sealed 109xx score."""

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
    run_v59_combined_screen as v59,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v60"
PLAN = OUT / "route-variation-probe-plan.json"
DEST = OUT / "opened-route-variation"
EVALUATOR = Path(__file__).with_name("evaluate_route_probe_v60.py")
SELECTOR = Path(__file__).with_name("selective_template_rgb_goal_v58.py")
ACTOR = v57.ACTOR
ANCHORS = ("large-02", "large-20", "large-30", "edge-01", "edge-12",
           "positive-03", "positive-10")
ROUTES = (
    ("center", .105, -.25),
    ("pitch-low", .095, -.25),
    ("pitch-high", .115, -.25),
    ("roll-low", .105, -.26),
    ("roll-high", .105, -.24),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cases() -> list[dict]:
    frozen = json.loads(v59.PLAN.read_text())["cases"]
    anchors = [case for case in frozen if case["id"] in ANCHORS]
    if len(anchors) != len(ANCHORS) or len({case["id"] for case in anchors}) != len(ANCHORS):
        raise ValueError("Route-probe anchor identity missing")
    return [{**case, "anchor_id": case["id"],
             "id": f"{case['id']}-{label}", "route_variant": label,
             "pitch_offset_rad": pitch, "roll_clearance_rad": roll}
            for case in anchors for label, pitch, roll in ROUTES]


def freeze() -> dict:
    predecessor = v59.DEST / "matrix.json"
    prior = json.loads(predecessor.read_text())
    if not prior["complete"] or not prior["screen_gate_met"] or prior["pass_count"] != 63:
        raise ValueError("Route probe requires all 63 combined native starts passing")
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(SELECTOR), template=sha(screen.core.TEMPLATE))
    varied = cases()
    by_id = {row["id"]: row for row in prior["cases"]}
    for case in varied:
        case["matched_prior_contact_s"] = by_id[case["anchor_id"]]["contact_s"]
        case["matched_prior_pass"] = by_id[case["anchor_id"]]["pass"]
    plan = {
        "schema": "robot-learning-lab.cove50-v60-route-variation-probe-plan.v1",
        "role": "opened_off_authored_route_diagnostic_not_actor_qualification",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "v59_combined_plan_sha256": sha(v59.PLAN),
        "v59_combined_matrix_sha256": sha(predecessor),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(varied),
        "cases": varied,
        "gate": "All 35 opened anchor-by-route native replays must pass the full support, rated effort, >=1 s continuous fingertip contact, hand floor, self-contact, clipping and <=10 mm scored RGB gates. Each route axis is varied independently by 0.01 rad on both sides of the saved route, on seven branch-diverse anchors. Require >=5 observed negative-camera partial corrections, >=5 positive sparse abstentions, >=5 edge bias uses and >=5 actual physical zero-pixel scans. Report sub-1.2 s contacts as thin. This is a source-separate generalization probe; it does not change the checkpoint's declared route contract or open sealed 109xx seeds.",
        "design": "Apply the exact V59 automatic controller and saved V57 arm to the original authored route and four nearby routes. Anchor visits include a known camera miss, physical edge scan, new edge contact, positive sparse branch and two thin 1.02 s contact visits. Hold body, motor, toy, gaze and RGB rules fixed within each anchor. Route identity is an offline experiment factor, not a runtime choice or evidence that the arm was trained on these routes.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V60 route-probe plan differs from source")
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
            raise RuntimeError(f"V60 route probe failed for {ident}: {log}")
    report = json.loads(result.read_text())
    hashes = plan["source_sha256"]
    if any(report.get(key) != value for key, value in {
        "evaluator_sha256": hashes["evaluator"],
        "selector_source_sha256": hashes["selector"],
        "template_source_sha256": hashes["template"],
        "body_sha256": hashes["body"],
        "selected_pointer_sha256": hashes["pointer"],
        "reach_checkpoint_sha256": hashes["actor"],
        "adaptive_variant": "v60_bounded_off_route_transfer_probe_not_qualified",
    }.items()):
        raise ValueError(f"V60 source identity mismatch: {result}")
    if (report["pitch_offset_rad"] != case["pitch_offset_rad"] or
            report["roll_clearance_rad"] != case["roll_clearance_rad"] or
            report["off_authored_route_probe"] != (case["route_variant"] != "center")):
        raise ValueError(f"V60 off-route experiment identity mismatch: {result}")
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
    return {"id": ident, "anchor_id": case["anchor_id"],
            "route_variant": case["route_variant"],
            "pitch_offset_rad": case["pitch_offset_rad"],
            "roll_clearance_rad": case["roll_clearance_rad"],
            "off_authored_route_probe": report["off_authored_route_probe"],
            "cohort": case["cohort"],
            "seed": seed, "development_role": case["development_role"],
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
            matrix = {"schema": "robot-learning-lab.cove50-v60-route-variation-probe.v1",
                      "role": "opened_off_authored_route_diagnostic_not_actor_qualification",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "negative_partial_correction_count": negative_uses,
                      "positive_sparse_abstention_count": positive_uses,
                      "edge_bias_use_count": edge_uses,
                      "physical_scan_count": physical_scans,
                      "screen_gate_met": (len(ordered) == len(plan["cases"]) and
                                          all(r["pass"] for r in ordered) and negative_uses >= 5 and
                                          positive_uses >= 5 and edge_uses >= 5 and physical_scans >= 5),
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "seed", "selection_mode", "pass", "contact_s", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
