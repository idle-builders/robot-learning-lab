"""Score frozen V87 on new seeds, thin routes and center-lock views."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v73_route_holdout as v73,
    run_v83_fresh_neighbor_challenge as earlier_design,
    run_v87_native_case as native,
    run_v84_prior75_pilot as pilot,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v87"
PLAN = OUT / "fresh-neighbor-challenge-plan.json"
DEST = OUT / "native-fresh-neighbors"

# New seeds only. Off-author routes cover all fourteen V73 anchors; the
# authored-center set perturbs the camera view that exposed the V83 regression.
CENTER_VARIATIONS = (
    (13001, 10, 0, -1.3, -.00050, -.00050),
    (13002, 12, 10, -1.3, .00050, .00050),
    (13003, 11, 5, -1.5, -.00050, .00050),
    (13004, 10, 10, -1.5, .00050, -.00050),
    (13005, 12, 0, -1.3, -.00025, .00025),
    (13006, 11, 10, -1.5, .00025, -.00025),
    (13007, 10, 5, -1.3, 0.0, .00050),
    (13008, 12, 5, -1.5, 0.0, -.00050),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    route_path = OUT / "native-routes63-retention/matrix.json"
    older_path = OUT / "native-prior75-retention/matrix.json"
    controls_path = OUT / "native-boundary-controls/matrix.json"
    variance_path = OUT / "native-boundary-variance/matrix.json"
    routes = json.loads(route_path.read_text())
    older = json.loads(older_path.read_text())
    controls = json.loads(controls_path.read_text())
    variance = json.loads(variance_path.read_text())
    if (not routes["complete"] or not routes["screen_gate_met"] or
            routes["case_count"] != 63 or routes["pass_count"] != 63 or
            routes["thin_rule_use_count"] != 6 or routes["lock_use_count"] != 0):
        raise ValueError("Need complete passing V87 63-route opened retention")
    if (not older["complete"] or not older["screen_gate_met"] or
            older["case_count"] != 75 or older["pass_count"] != 75 or
            older["thin_rule_use_count"] != 0):
        raise ValueError("Need complete passing V87 75-start older retention")
    if (not controls["complete"] or not controls["controls_gate_met"] or
            controls["case_count"] != 4 or controls["pass_count"] != 4 or
            not variance["complete"] or not variance["variance_gate_met"] or
            variance["case_count"] != 16 or variance["pass_count"] != 16):
        raise ValueError("Need passing V87 near-boundary controls and varied neighborhoods")
    ten_plan = json.loads(pilot.PLAN.read_text())
    retention_plan = json.loads((OUT / "routes63-retention-plan.json").read_text())
    sources = retention_plan["source_sha256"]
    if (sources["evaluator"] != sha(native.EVALUATOR) or
            retention_plan["native_case_runner_sha256"] != sha(Path(native.__file__))):
        raise ValueError("Frozen V87 policy source changed")
    originals = {case["id"]: case for case in json.loads(v73.PLAN.read_text())["cases"]}
    baseline = {row["id"]: row for row in
                json.loads((v73.DEST / "matrix.json").read_text())["cases"]}
    cases = []
    for base_id, seed, rail_v, delay_ms, gaze, shift_x, shift_y, expected in (
            earlier_design.VARIATIONS):
        original = originals[base_id]
        pitch_jitter, roll_jitter = earlier_design.ROUTE_JITTER.get(seed, (0.0, 0.0))
        cases.append({**original,
                      "id": f"newseed-{seed}-{base_id}", "seed": seed,
                      "rail_v": rail_v, "delay_ms": delay_ms,
                      "gaze_head_yaw_rad": gaze,
                      "toy_offset_x_m": round(original["toy_offset_x_m"] + shift_x, 6),
                      "toy_offset_y_m": round(original["toy_offset_y_m"] + shift_y, 6),
                      "pitch_offset_rad": round(original["pitch_offset_rad"] + pitch_jitter, 6),
                      "roll_clearance_rad": round(original["roll_clearance_rad"] + roll_jitter, 6),
                      "baseline_contact_s": baseline[base_id]["contact_s"],
                      "base_route_id": base_id,
                      "expected_thin_route_bias": expected,
                      "expected_center_live_lock": False})
    base = next(case for case in ten_plan["cases"] if case["seed"] == 10534)
    for seed, rail_v, delay_ms, gaze, shift_x, shift_y in CENTER_VARIATIONS:
        cases.append({**base,
                      "id": f"newseed-{seed}-center-lock", "seed": seed,
                      "rail_v": rail_v, "delay_ms": delay_ms,
                      "gaze_head_yaw_rad": gaze,
                      "toy_offset_x_m": round(base["toy_offset_x_m"] + shift_x, 6),
                      "toy_offset_y_m": round(base["toy_offset_y_m"] + shift_y, 6),
                      "baseline_contact_s": base["baseline_contact_s"],
                      "expected_center_live_lock": True,
                      "expected_thin_route_bias": False})
    historical_seeds = {row["seed"] for row in routes["cases"] + older["cases"]}
    anchors = {case["base_route_id"].rsplit("-", 2)[0] for case in cases[:26]}
    if (len(cases) != 34 or len({case["id"] for case in cases}) != 34 or
            len({case["seed"] for case in cases}) != 34 or
            any(case["seed"] in historical_seeds for case in cases) or
            sum(case["expected_thin_route_bias"] for case in cases) != 12 or
            len(anchors) != 14 or
            [case["seed"] for case in cases[26:]] != [item[0] for item in CENTER_VARIATIONS]):
        raise ValueError("Fresh challenge seeds, anchors or exposures differ")
    plan = {
        "schema": "robot-learning-lab.cove50-v87-fresh-neighbor-challenge-plan.v1",
        "role": "new_seed_varied_route_and_center_challenge_not_new_body_qualification",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "runner_sha256": sha(Path(__file__)),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "route_retention_plan_sha256": sha(OUT / "routes63-retention-plan.json"),
        "route_retention_matrix_sha256": sha(route_path),
        "older_retention_plan_sha256": sha(OUT / "prior75-retention-plan.json"),
        "older_retention_matrix_sha256": sha(older_path),
        "boundary_controls_plan_sha256": sha(OUT / "boundary-controls-plan.json"),
        "boundary_controls_matrix_sha256": sha(controls_path),
        "boundary_variance_plan_sha256": sha(OUT / "boundary-variance-plan.json"),
        "boundary_variance_matrix_sha256": sha(variance_path),
        "v73_plan_sha256": sha(v73.PLAN),
        "v73_matrix_sha256": sha(v73.DEST / "matrix.json"),
        "variation_source_sha256": sha(Path(earlier_design.__file__)),
        "center_anchor_plan_sha256": sha(pilot.PLAN),
        "case_count": len(cases), "thin_target_count": 12,
        "route_control_count": 14, "center_lock_target_count": 8,
        "cases": cases,
        "gate": ("All 34 new-seed native starts must pass contact, support, rated effort, "
                 "floor, self-contact, clipping and scored RGB. Twelve thin-route targets "
                 "must activate the observed route rule and hold at least 1.2 s continuous "
                 "fingertip contact; fourteen route controls must keep both new rules off. "
                 "Eight authored-center targets must activate the observed live lock and "
                 "hold at least 1.2 s, with the thin-route rule off. A missing image or "
                 "branch is a failure. This is varied scoring around opened anchors on a "
                 "rejected body, not a new-anchor or selected-body qualification."),
        "design": ("Freeze V87 only after 63-route and 75-start retention and the "
                   "four neighboring older controls plus sixteen new-seed boundary "
                   "variations all pass. The off-author "
                   "26 cases are the unused V83 predeclared new-seed challenge over all "
                   "fourteen V73 anchors, with rail 10-12 V, delay 0-10 ms, two gaze "
                   "angles, up to 1 mm toy shifts and .003 rad route jitter. Eight further "
                   "unused seeds perturb the one older authored camera region that caused "
                   "V83 regression, spanning the same rail, delay and gaze range and "
                   "up to .5 mm toy shifts. All expectations are frozen before V87 scoring."),
    }
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V87 fresh neighbor plan differs from current source")
    PLAN.write_bytes(encoded)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(plan["cases"])
        pending = {}
        for _ in range(args.workers):
            case = next(iterator, None)
            if case is not None:
                pending[pool.submit(native.run_case, case, plan, DEST)] = case
        while pending:
            future = next(as_completed(tuple(pending)))
            case = pending.pop(future)
            row = future.result()
            row["base_route_id"] = case.get("base_route_id")
            rows[row["id"]] = row
            ordered = [rows[item["id"]] for item in plan["cases"] if item["id"] in rows]
            thin = [item for item in ordered if item["expected_thin_route_bias"]]
            controls = [item for item in ordered if not item["expected_thin_route_bias"]
                        and item["route_variant"] != "center"]
            center = [item for item in ordered if item["route_variant"] == "center"]
            complete = len(ordered) == len(plan["cases"])
            score_gate = (complete and all(item["pass"] for item in ordered) and
                          len(thin) == 12 and len(controls) == 14 and len(center) == 8 and
                          all(item["automatic_thin_route_bias_applied"] and
                              item["margin_1p2_s_met"] and
                              not item["automatic_center_live_lock_applied"] for item in thin) and
                          all(not item["automatic_thin_route_bias_applied"] and
                              not item["automatic_center_live_lock_applied"] for item in controls) and
                          all(item["automatic_center_live_lock_applied"] and
                              item["margin_1p2_s_met"] and
                              not item["automatic_thin_route_bias_applied"] for item in center))
            matrix = {
                "schema": "robot-learning-lab.cove50-v87-fresh-neighbor-challenge.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN), "complete": complete,
                "case_count": len(ordered), "pass_count": sum(item["pass"] for item in ordered),
                "thin_rule_use_count": sum(item["automatic_thin_route_bias_applied"]
                                           for item in ordered),
                "center_lock_use_count": sum(item["automatic_center_live_lock_applied"]
                                             for item in ordered),
                "margin_1p2_s_count": sum(item["margin_1p2_s_met"] for item in ordered),
                "score_gate_met": score_gate, "cases": ordered,
            }
            temp = DEST / "matrix.json.tmp"
            temp.write_text(json.dumps(matrix, indent=2) + "\n")
            os.replace(temp, DEST / "matrix.json")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "automatic_center_live_lock_applied",
                               "automatic_thin_route_bias_applied", "faults")}), flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(native.run_case, next_case, plan, DEST)] = next_case


if __name__ == "__main__":
    main()
