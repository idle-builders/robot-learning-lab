"""Opened V85 neighborhood screen around two recovered older route-boundary misses."""

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
    run_v85_native_case as native,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v85"
PLAN = OUT / "boundary-variance-plan.json"
DEST = OUT / "native-boundary-variance"

# Eight distinct motor, gaze and toy perturbations per opened failure anchor.
VARIATIONS = (
    (10, 0, -1.3, -.001, -.0005),
    (12, 10, -1.5, .001, .0005),
    (10, 10, -1.5, -.0005, .001),
    (12, 0, -1.3, .0005, -.001),
    (11, 5, -1.3, 0, -.0005),
    (11, 0, -1.5, -.0005, 0),
    (12, 5, -1.3, .001, -.0005),
    (10, 5, -1.5, .0005, .001),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    diagnostic_path = OUT / "native-boundary-diagnostic/matrix.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    route_plan_path = OUT / "routes63-retention-plan.json"
    route_plan = json.loads(route_plan_path.read_text())
    source_path = BASE / "v84/prior75-retention-plan.json"
    source = json.loads(source_path.read_text())
    if (not diagnostic["complete"] or not diagnostic["diagnostic_gate_met"] or
            diagnostic["case_count"] != 4 or diagnostic["pass_count"] != 4 or
            route_plan["source_sha256"]["evaluator"] != sha(native.EVALUATOR) or
            route_plan["native_case_runner_sha256"] != sha(Path(native.__file__))):
        raise ValueError("Need passing source-matched V85 boundary diagnostic")
    anchors = {case["seed"]: case for case in source["cases"] if case["seed"] in (10505, 10507)}
    if len(anchors) != 2:
        raise ValueError("Missing original older boundary anchors")
    cases = []
    for anchor_seed, start_seed in ((10505, 14001), (10507, 14009)):
        anchor = anchors[anchor_seed]
        for index, (rail, delay, gaze, shift_x, shift_y) in enumerate(VARIATIONS):
            seed = start_seed + index
            cases.append({**anchor, "id": f"variance-{seed}-from-{anchor_seed}",
                          "seed": seed, "base_seed": anchor_seed,
                          "rail_v": rail, "delay_ms": delay,
                          "gaze_head_yaw_rad": gaze,
                          "toy_offset_x_m": round(anchor["toy_offset_x_m"] + shift_x, 6),
                          "toy_offset_y_m": round(anchor["toy_offset_y_m"] + shift_y, 6),
                          "expected_center_live_lock": False,
                          "expected_thin_route_bias": False})
    if (len(cases) != 16 or len({case["seed"] for case in cases}) != 16 or
            any(case["seed"] in {item["seed"] for item in source["cases"]} for case in cases)):
        raise ValueError("Variance seeds overlap older retention")
    plan = {
        "schema": "robot-learning-lab.cove50-v85-boundary-variance-plan.v1",
        "role": "opened_boundary_neighborhood_not_disjoint_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": route_plan["source_sha256"],
        "source_v84_older_plan_sha256": sha(source_path),
        "v85_boundary_matrix_sha256": sha(diagnostic_path),
        "v85_route_plan_sha256": sha(route_plan_path),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "cases": cases,
        "gate": ("Sixteen predeclared new-seed neighborhood visits, eight around each "
                 "opened failure, must all pass native RGB, support, rated effort, "
                 "contact, floor, self-contact and clipping gates and hold at least "
                 "1.2 s uninterrupted fingertip contact. At least eight visits must "
                 "actually choose the height-assisted route; thin and center-lock "
                 "rules must remain off. This is opened development, not the frozen "
                 "34-case disjoint score or new-body qualification."),
        "variation": ("Rail 10-12 V, delay 0-10 ms, gaze -1.3/-1.5 rad and toy "
                      "X/Y perturbations up to 1 mm around both 10505 and 10507. "
                      "The controller sees only the camera and robot state, never "
                      "the anchor label or scored toy truth."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != plan:
            raise ValueError("Frozen V85 boundary variance plan changed")
    else:
        PLAN.write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four native evaluators")
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
            row["base_seed"] = case["base_seed"]
            rows[row["id"]] = row
            ordered = [rows[item["id"]] for item in plan["cases"] if item["id"] in rows]
            height = [item for item in ordered if item["selection_mode"] == "v16_v10_height_locked"]
            complete = len(ordered) == len(plan["cases"])
            matrix = {
                "schema": "robot-learning-lab.cove50-v85-boundary-variance.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN), "complete": complete,
                "case_count": len(ordered), "pass_count": sum(item["pass"] for item in ordered),
                "height_route_use_count": len(height),
                "minimum_contact_s": min(item["contact_s"] for item in ordered),
                "variance_gate_met": (complete and all(item["pass"] and
                                       item["margin_1p2_s_met"] and
                                       not item["automatic_center_live_lock_applied"] and
                                       not item["automatic_thin_route_bias_applied"]
                                       for item in ordered) and len(height) >= 8),
                "cases": ordered,
            }
            temp = DEST / "matrix.json.tmp"
            temp.write_text(json.dumps(matrix, indent=2) + "\n")
            os.replace(temp, DEST / "matrix.json")
            print(json.dumps({"id": row["id"], "pass": row["pass"],
                              "selection_mode": row["selection_mode"],
                              "contact_s": row["contact_s"],
                              "faults": row["faults"]}), flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(native.run_case, next_case, plan, DEST)] = next_case


if __name__ == "__main__":
    main()
