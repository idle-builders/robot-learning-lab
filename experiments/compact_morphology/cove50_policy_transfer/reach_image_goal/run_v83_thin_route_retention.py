"""Retain the V83 image-conditioned route rule on V70 and V73 opened starts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v70_center_route_screen as v70,
    run_v73_route_holdout as v73,
    run_v83_sibling_repeatability as sibling,
    run_v83_thin_route_pilot as v83,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v83"
PLAN = OUT / "opened-retention-plan.json"
DEST = OUT / "native-retention"
SUPERSEDED = OUT / "opened-retention-invalid-center-validator-plan.json"
EXPOSURE = ("positive-01-pitch-low", "positive-01-roll-high",
            "positive-07-pitch-high", "positive-07-roll-low",
            "edge-11-pitch-high", "edge-11-roll-low")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed_hit(report: dict, case: dict) -> bool:
    entry = next(event for event in report["image_goal_capture_events"]
                 if event["time_s"] == report["reach_start_s"])
    camera_x, camera_y = entry["camera_world_xyz_m"][:2]
    selected_y = report["selection_features"]["selected_image_error_xy_m"][1]
    pitch_delta = case["pitch_offset_rad"] - report["actor_authored_route_pitch_offset_rad"]
    roll_delta = case["roll_clearance_rad"] - report["actor_authored_route_roll_clearance_rad"]
    if (abs(pitch_delta) < .005 and abs(roll_delta) < .005) or any((
            report["automatic_front_route_bias_applied"],
            report["automatic_front_center_bias_applied"],
            report["automatic_edge_route_bias_applied"])):
        return False
    return ((report["selection_mode"] == "v16_locked" and selected_y >= .018 and
             camera_x >= 0 and camera_y <= -.08) or
            (report["selection_mode"] == "v21_scoped_y_locked" and
             selected_y >= .0075 and camera_x <= -.06 and camera_y >= .05))


def run_retention_case(case: dict, plan: dict) -> dict:
    """Use the frozen pilot validator, allowing the three unchanged V70 centers."""
    if case["route_variant"] != "center":
        return v83.run_case(case, plan)
    try:
        return v83.run_case(case, plan)
    except ValueError as exc:
        result = DEST / f"seed-{case['seed']}-{case['id']}.json"
        if str(exc) != f"V83 route identity mismatch: {result}":
            raise
    # The pilot purposely requires an off-author route. Its source/hash checks
    # have passed above; only that pilot-specific route guard rejected center.
    report = json.loads(result.read_text())
    if (case["expected_thin_route_bias"] or
            report["pitch_offset_rad"] != case["pitch_offset_rad"] or
            report["roll_clearance_rad"] != case["roll_clearance_rad"] or
            report["off_authored_route_probe"] or
            case["pitch_offset_rad"] != .105 or
            case["roll_clearance_rad"] != -.25):
        raise ValueError(f"Center-route identity differs from frozen V70 case: {result}")
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
    if report["automatic_thin_route_bias_applied"] or report["automatic_thin_route_region"]:
        faults.append("center route activated off-author thin-route rule")
    expected_bias = (-.004 if report["edge_goal_bias_applied"] else
                     -.002 if report["selection_mode"] in
                     ("v16_bias_y_locked", "v21_scoped_y_locked") else 0.0)
    expected_bias += report["automatic_front_route_bias_y_m"]
    expected_bias += -.004 if report["automatic_front_center_bias_applied"] else 0.0
    expected_bias += -.004 if report["automatic_edge_route_bias_applied"] else 0.0
    if report["goal_bias_body_xy_m"] != [0.0, expected_bias]:
        faults.append("goal bias differs from observed entry rule")
    trace = DEST / f"seed-{case['seed']}-{case['id']}.npz"
    log = DEST / f"seed-{case['seed']}-{case['id']}.log"
    row = {
        "id": case["id"], "seed": case["seed"],
        "route_variant": case["route_variant"],
        "v73_contact_s": case["v73_contact_s"],
        "selection_mode": report["selection_mode"],
        "expected_thin_route_bias": False,
        "automatic_thin_route_bias_applied": report["automatic_thin_route_bias_applied"],
        "automatic_thin_route_region": report["automatic_thin_route_region"],
        "entry_orange_pixels": None if entry is None else entry["orange_pixels"],
        "entry_goal_error_mm_scoring_only": (None if entry is None else
                                             entry.get("xy_error_truth_mm_scoring_only")),
        "contact_s": report["phases"].get("reach", {}).get("continuous_finger_contact_s", 0),
        "pass": not faults, "faults": faults,
        "result": str(result.relative_to(ROOT)), "trace": str(trace.relative_to(ROOT)),
        "log": str(log.relative_to(ROOT)),
    }
    row["margin_1p2_s_met"] = row["contact_s"] >= 1.2
    return row


def freeze() -> dict:
    pilot = json.loads((v83.DEST / "matrix.json").read_text())
    extra = json.loads((sibling.DEST / "matrix.json").read_text())
    if (not pilot["complete"] or not pilot["pilot_gate_met"] or
            not extra["complete"] or not extra["gate_met"]):
        raise ValueError("Need passing V83 pilot, sibling routes and exact repeats")
    sources = json.loads(v83.PLAN.read_text())["source_sha256"]
    if (sources["evaluator"] != sha(v83.EVALUATOR) or
            sources["runner"] != sha(Path(v83.__file__))):
        raise ValueError("V83 policy source changed after pilot freeze")
    cases = []
    hits = []
    for label, runner, count, passes in (("V70", v70, 35, 35),
                                         ("V73", v73, 28, 27)):
        matrix = json.loads((runner.DEST / "matrix.json").read_text())
        if (not matrix["complete"] or matrix["case_count"] != count or
                matrix["pass_count"] != passes):
            raise ValueError(f"Expected complete {label} opened source matrix")
        by_id = {row["id"]: row for row in matrix["cases"]}
        for original in json.loads(runner.PLAN.read_text())["cases"]:
            previous = by_id[original["id"]]
            report = json.loads((ROOT / previous["result"]).read_text())
            expected = observed_hit(report, original)
            if expected:
                hits.append(original["id"])
            cases.append({**original, "v73_contact_s": previous["contact_s"],
                          "baseline_source": label,
                          "expected_thin_route_bias": expected})
    if len(cases) != 63 or len({case["id"] for case in cases}) != 63 or set(hits) != set(EXPOSURE):
        raise ValueError("Opened retention exposure differs from six audited V73 routes")
    plan = {
        "schema": "robot-learning-lab.cove50-v83-opened-retention-plan.v1",
        "role": "opened_observable_rule_retention_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "runner_sha256": sha(Path(__file__)),
        "pilot_plan_sha256": sha(v83.PLAN),
        "pilot_matrix_sha256": sha(v83.DEST / "matrix.json"),
        "sibling_plan_sha256": sha(sibling.PLAN),
        "sibling_matrix_sha256": sha(sibling.DEST / "matrix.json"),
        "v70_plan_sha256": sha(v70.PLAN),
        "v70_matrix_sha256": sha(v70.DEST / "matrix.json"),
        "v73_plan_sha256": sha(v73.PLAN),
        "v73_matrix_sha256": sha(v73.DEST / "matrix.json"),
        "exposure_ids": EXPOSURE,
        "case_count": len(cases), "cases": cases,
        "exposure_gate": "The six V73 visits whose prior observed entry features select the V83 offset must all pass the full native gate, activate the rule and hold at least 1.2 s fingertip contact in new processes before wider retention starts.",
        "full_gate": "All 35 formerly passing V70 tuned routes and all 28 V73 different-anchor routes must pass the full native contact, support, rated-effort, hand-floor, self-contact, clipping and RGB gates. Exactly the six audited V73 visits may activate the added V83 rule, and all six must hold at least 1.2 s. This is opened retention on the rejected body, not a disjoint score or selected-body qualification.",
        "design": "Replay the frozen V83 evaluator and V57 actor on every V70 and V73 case without changing any body, camera, motor or route parameter. Derive expected V83 activation from each saved V70/V73 entry image, camera pose, selected image-goal error and commanded route; confirm that only six V73 siblings meet the rule. Stage the six activations first, then replay all 63 opened starts. The first full-run validator was pilot-only and rejected an unchanged center route; this runner accepts those three frozen V70 center routes without altering evaluator, case or gate.",
        "superseded_invalid_harness_plan_sha256": (sha(SUPERSEDED) if SUPERSEDED.exists() else None),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V83 opened retention differs from current source")
    PLAN.write_bytes(encoded)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--stage", choices=("exposure", "full"), default="exposure")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    if args.stage == "full":
        earlier = json.loads((OUT / "exposure-stage-matrix.json").read_text())
        if not earlier["complete"] or not earlier["exposure_gate_met"]:
            raise ValueError("Six-route exposure stage must pass before full retention")
    selected = ([case for case in plan["cases"] if case["id"] in EXPOSURE]
                if args.stage == "exposure" else plan["cases"])
    DEST.mkdir(parents=True, exist_ok=True)
    v83.DEST = DEST
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_retention_case, case, plan): case["id"] for case in selected}
        for future in as_completed(futures):
            row = future.result()
            original = next(case for case in selected if case["id"] == row["id"])
            row["baseline_contact_s"] = row.pop("v73_contact_s")
            row["baseline_source"] = original["baseline_source"]
            rows[row["id"]] = row
            ordered = [rows[case["id"]] for case in selected if case["id"] in rows]
            exposure = [item for item in ordered if item["expected_thin_route_bias"]]
            exposure_gate = (len(exposure) == len(EXPOSURE) and
                             all(item["pass"] and item["automatic_thin_route_bias_applied"] and
                                 item["margin_1p2_s_met"] for item in exposure))
            matrix = {
                "schema": f"robot-learning-lab.cove50-v83-opened-{args.stage}-retention.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "stage": args.stage, "plan_sha256": sha(PLAN),
                "complete": len(ordered) == len(selected),
                "case_count": len(ordered), "pass_count": sum(item["pass"] for item in ordered),
                "thin_rule_use_count": sum(item["automatic_thin_route_bias_applied"]
                                           for item in ordered),
                "exposure_gate_met": exposure_gate,
                "screen_gate_met": (args.stage == "full" and len(ordered) == len(selected) and
                                    all(item["pass"] for item in ordered) and exposure_gate and
                                    sum(item["automatic_thin_route_bias_applied"]
                                        for item in ordered) == len(EXPOSURE)),
                "cases": ordered,
            }
            path = (OUT / "exposure-stage-matrix.json" if args.stage == "exposure" else
                    DEST / "matrix.json")
            path.write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "automatic_thin_route_region", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
