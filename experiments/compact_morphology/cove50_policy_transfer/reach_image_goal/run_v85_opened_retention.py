"""Source-frozen V85 route and older-start retention after the opened boundary diagnostic."""

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
SOURCE = BASE / "v84"
EXPECTED = {"routes63": 63, "prior75": 75}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(stage: str) -> tuple[dict, Path, Path]:
    diagnostic_path = OUT / "native-boundary-diagnostic/matrix.json"
    diagnostic_plan_path = OUT / "boundary-diagnostic-plan.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    diagnostic_plan = json.loads(diagnostic_plan_path.read_text())
    if (not diagnostic["complete"] or not diagnostic["diagnostic_gate_met"] or
            diagnostic["case_count"] != 4 or diagnostic["pass_count"] != 4 or
            diagnostic["plan_sha256"] != sha(diagnostic_plan_path) or
            diagnostic_plan["source_sha256"]["evaluator"] != sha(native.EVALUATOR) or
            diagnostic_plan["native_case_runner_sha256"] != sha(Path(native.__file__))):
        raise ValueError("Need passing source-matched V85 four-start diagnostic")
    if stage == "prior75":
        prior_route = json.loads((OUT / "native-routes63-retention/matrix.json").read_text())
        if (not prior_route["complete"] or not prior_route["screen_gate_met"] or
                prior_route["case_count"] != 63 or prior_route["pass_count"] != 63):
            raise ValueError("Need passing V85 63-route retention before older starts")
    source_plan_path = SOURCE / ("routes63-retention-plan.json" if stage == "routes63"
                                 else "prior75-retention-plan.json")
    source = json.loads(source_plan_path.read_text())
    if len(source["cases"]) != EXPECTED[stage]:
        raise ValueError("V84 source case count changed")
    hashes = {**source["source_sha256"], "evaluator": sha(native.EVALUATOR)}
    if hashes["pointer"] != sha(ROOT / "robot/ape-current-design.json"):
        raise ValueError("Mechanical design pointer changed")
    cases = []
    for source_case in source["cases"]:
        case = dict(source_case)
        if stage == "prior75" and case["seed"] in (10505, 10507):
            case["expected_selection_mode"] = "v16_v10_height_locked"
        cases.append(case)
    plan = {
        "schema": f"robot-learning-lab.cove50-v85-{stage}-retention-plan.v1",
        "role": f"opened_{stage}_native_retention_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": hashes,
        "source_v84_plan_sha256": sha(source_plan_path),
        "v85_boundary_plan_sha256": sha(diagnostic_plan_path),
        "v85_boundary_matrix_sha256": sha(diagnostic_path),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "cases": cases,
        "gate": ("All 63 earlier V70/V73 routes pass native RGB, contact, support, motor, "
                 "floor and clipping gates; exactly six audited thin-rule visits hold "
                 "at least 1.2 s contact; center lock stays off."
                 if stage == "routes63" else
                 "All 75 formerly passing V41 starts pass full native gates. No thin "
                 "route activates; ten pilot center-lock exposures match the frozen "
                 "plan; 10534 and the 10505/10507 boundary rescues hold at least "
                 "1.2 s, with the latter two selecting the height route."),
        "change": ("From V84, only extend the height-assisted V10 route boundary "
                   "from X < -12 mm to X < -11.5 mm when observed entry Y < -7 mm."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / f"{stage}-retention-plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("Frozen V85 plan/source identity changed")
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan, plan_path, OUT / f"native-{stage}-retention"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=EXPECTED)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan, path, dest = freeze(args.stage)
    if args.freeze_only:
        print(f"frozen {path} sha256={sha(path)}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = iter(plan["cases"])
        pending = {}
        for _ in range(args.workers):
            case = next(iterator, None)
            if case is not None:
                pending[pool.submit(native.run_case, case, plan, dest)] = case
        while pending:
            future = next(as_completed(tuple(pending)))
            case = pending.pop(future)
            row = future.result()
            if (case.get("expected_selection_mode") is not None and
                    row["selection_mode"] != case["expected_selection_mode"]):
                row["faults"].append("boundary route differs from frozen expectation")
                row["pass"] = False
            rows[row["id"]] = row
            ordered = [rows[item["id"]] for item in plan["cases"] if item["id"] in rows]
            thin = [item for item in ordered if item["automatic_thin_route_bias_applied"]]
            locks = [item for item in ordered if item["automatic_center_live_lock_applied"]]
            complete = len(ordered) == len(plan["cases"])
            if args.stage == "routes63":
                rule_gate = (len(thin) == 6 and
                             all(item["expected_thin_route_bias"] and
                                 item["margin_1p2_s_met"] for item in thin) and not locks)
            else:
                pilot = [item for item in ordered if item["expected_center_live_lock"] is not None]
                boundary = [item for item in ordered if item["seed"] in (10505, 10507)]
                rescue = [item for item in ordered if item["seed"] == 10534]
                rule_gate = (not thin and len(pilot) == 10 and
                             all(item["automatic_center_live_lock_applied"] ==
                                 item["expected_center_live_lock"] for item in pilot) and
                             len(boundary) == 2 and
                             all(item["margin_1p2_s_met"] and
                                 item["selection_mode"] == "v16_v10_height_locked"
                                 for item in boundary) and
                             len(rescue) == 1 and rescue[0]["margin_1p2_s_met"])
            matrix = {
                "schema": f"robot-learning-lab.cove50-v85-{args.stage}-retention.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(path), "complete": complete,
                "case_count": len(ordered),
                "pass_count": sum(item["pass"] for item in ordered),
                "thin_rule_use_count": len(thin), "lock_use_count": len(locks),
                "minimum_contact_s": min(item["contact_s"] for item in ordered),
                "margin_1p2_s_count": sum(item["margin_1p2_s_met"] for item in ordered),
                "rule_gate_met": rule_gate,
                "screen_gate_met": complete and all(item["pass"] for item in ordered) and rule_gate,
                "cases": ordered,
            }
            temp = dest / "matrix.json.tmp"
            temp.write_text(json.dumps(matrix, indent=2) + "\n")
            os.replace(temp, dest / "matrix.json")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "selection_mode",
                               "automatic_center_live_lock_applied",
                               "automatic_thin_route_bias_applied", "faults")}), flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(native.run_case, next_case, plan, dest)] = next_case


if __name__ == "__main__":
    main()
