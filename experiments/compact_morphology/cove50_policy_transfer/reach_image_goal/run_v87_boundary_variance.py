"""Replay the exact frozen V85 neighborhood under V87's scoped timing."""

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
    run_v87_native_case as native,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v87"
PLAN = OUT / "boundary-variance-plan.json"
DEST = OUT / "native-boundary-variance"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    v85_plan_path = BASE / "v85/boundary-variance-plan.json"
    v85_matrix_path = BASE / "v85/native-boundary-variance/matrix.json"
    pilot_plan_path = OUT / "scoped-approach-probe-plan.json"
    pilot_matrix_path = OUT / "native-scoped-approach-probe/matrix.json"
    v85_plan = json.loads(v85_plan_path.read_text())
    v85_matrix = json.loads(v85_matrix_path.read_text())
    pilot_plan = json.loads(pilot_plan_path.read_text())
    pilot_matrix = json.loads(pilot_matrix_path.read_text())
    if (not v85_matrix["complete"] or v85_matrix["case_count"] != 16 or
            v85_matrix["plan_sha256"] != sha(v85_plan_path) or
            not pilot_matrix["complete"] or not pilot_matrix["diagnostic_gate_met"] or
            pilot_matrix["plan_sha256"] != sha(pilot_plan_path) or
            pilot_plan["source_sha256"]["evaluator"] != sha(native.EVALUATOR) or
            pilot_plan["native_case_runner_sha256"] != sha(Path(native.__file__))):
        raise ValueError("Need complete V85 neighborhood and passing source-matched V87 pilot")
    v85_by_id = {row["id"]: row for row in v85_matrix["cases"]}
    if set(v85_by_id) != {case["id"] for case in v85_plan["cases"]}:
        raise ValueError("V85 neighborhood cases differ from its frozen plan")
    cases = [{**case, "baseline_contact_s": v85_by_id[case["id"]]["contact_s"]}
             for case in v85_plan["cases"]]
    plan = {
        "schema": "robot-learning-lab.cove50-v87-boundary-variance-plan.v1",
        "role": "opened_paired_boundary_neighborhood_not_disjoint_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": pilot_plan["source_sha256"],
        "source_v85_plan_sha256": sha(v85_plan_path),
        "source_v85_matrix_sha256": sha(v85_matrix_path),
        "source_v87_pilot_plan_sha256": sha(pilot_plan_path),
        "source_v87_pilot_matrix_sha256": sha(pilot_matrix_path),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "cases": cases,
        "gate": ("All sixteen exact V85 opened new-seed neighborhood visits must "
                 "pass fresh native RGB, support, rated effort, contact, floor, "
                 "self-contact and clipping gates and hold at least 1.2 s "
                 "uninterrupted fingertip contact. At least eight must select "
                 "the height-assisted route; thin and center-lock rules must stay "
                 "off. This is paired development, not the 34-case disjoint score."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != plan:
            raise ValueError("Frozen V87 boundary variance plan changed")
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
            height = [item for item in ordered if item["selection_mode"] ==
                      "v16_v10_height_locked"]
            complete = len(ordered) == len(plan["cases"])
            matrix = {
                "schema": "robot-learning-lab.cove50-v87-boundary-variance.v1",
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
