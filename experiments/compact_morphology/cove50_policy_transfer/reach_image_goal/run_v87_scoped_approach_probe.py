"""Opened scoped-timing probe after V86 plain-route contact regression."""

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
PLAN = OUT / "scoped-approach-probe-plan.json"
DEST = OUT / "native-scoped-approach-probe"
SEEDS = (14002, 14004, 14001, 14003, 10505)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    route_plan_path = BASE / "v85/routes63-retention-plan.json"
    route_plan = json.loads(route_plan_path.read_text())
    variance_plan_path = BASE / "v85/boundary-variance-plan.json"
    variance_plan = json.loads(variance_plan_path.read_text())
    diagnostic_plan_path = BASE / "v85/boundary-diagnostic-plan.json"
    diagnostic_plan = json.loads(diagnostic_plan_path.read_text())
    v86_regression_path = next((BASE / "v86/native-boundary-variance").glob(
        "seed-14002-*.json"))
    v86_regression = json.loads(v86_regression_path.read_text())
    sources = route_plan["source_sha256"].copy()
    sources["evaluator"] = sha(native.EVALUATOR)
    sources["runner"] = sha(Path(native.__file__))
    by_seed = {case["seed"]: case for case in
               variance_plan["cases"] + diagnostic_plan["cases"]}
    if (set(SEEDS) - by_seed.keys() or
            route_plan["source_sha256"]["body"] != sources["body"] or
            v86_regression["adaptive_variant"] != "v86_five_second_approach_probe" or
            v86_regression["phases"]["reach"]["continuous_finger_contact_s"] >= 1):
        raise ValueError("Missing source-matched V85 opened cases")
    cases = []
    source_reports = {}
    for seed in SEEDS:
        case = by_seed[seed].copy()
        source_dir = BASE / "v85" / ("native-boundary-variance" if seed != 10505
                                     else "native-boundary-diagnostic")
        report_path = next(source_dir.glob(f"seed-{seed}-*.json"))
        report = json.loads(report_path.read_text())
        if (report["evaluator_sha256"] != route_plan["source_sha256"]["evaluator"] or
                report["body_sha256"] != sources["body"]):
            raise ValueError(f"V85 report identity mismatch for {seed}")
        source_reports[str(seed)] = {"path": str(report_path.relative_to(ROOT)),
                                     "sha256": sha(report_path)}
        case["id"] = f"scoped-{seed}"
        case["baseline_contact_s"] = report["phases"]["reach"]["continuous_finger_contact_s"]
        case["expected_selection_mode"] = ("v16_v10_locked" if seed == 14002
                                           else "v16_v10_height_locked")
        case["expected_center_live_lock"] = False
        case["expected_thin_route_bias"] = False
        cases.append(case)
    plan = {
        "schema": "robot-learning-lab.cove50-v87-scoped-approach-probe-plan.v1",
        "role": "opened_scoped_timing_diagnostic_not_disjoint_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sources,
        "source_v85_route_plan_sha256": sha(route_plan_path),
        "source_v85_variance_plan_sha256": sha(variance_plan_path),
        "source_v85_diagnostic_plan_sha256": sha(diagnostic_plan_path),
        "source_v85_reports": source_reports,
        "source_v86_14002_regression_report": {
            "path": str(v86_regression_path.relative_to(ROOT)),
            "sha256": sha(v86_regression_path),
        },
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "cases": cases,
        "single_change": ("Use a five-second authored reach blend and shin-clearance pulse "
                          "only on the observed height-assisted route; preserve four "
                          "seconds on every other route. Actor weights, observed "
                          "router, RGB goal, body, motors and scoring stay fixed."),
        "gate": ("The opened V86 plain-route regression 14002, the V85 height-route "
                 "miss 14004 and three passing controls must all pass fresh full "
                 "native gates with at least "
                 "1.2 s uninterrupted contact. This diagnostic does not replace "
                 "63-route, 75-older, 16-varied or 34-disjoint gates."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != plan:
            raise ValueError("Frozen V87 scoped approach probe plan changed")
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
        futures = {pool.submit(native.run_case, case, plan, DEST): case
                   for case in plan["cases"]}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            if row["selection_mode"] != case["expected_selection_mode"]:
                row["faults"].append("observed route differs from frozen expectation")
                row["pass"] = False
            rows[row["id"]] = row
            ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
            complete = len(ordered) == len(plan["cases"])
            matrix = {
                "schema": "robot-learning-lab.cove50-v87-scoped-approach-probe.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN), "complete": complete,
                "case_count": len(ordered), "pass_count": sum(x["pass"] for x in ordered),
                "minimum_contact_s": min(x["contact_s"] for x in ordered),
                "diagnostic_gate_met": (complete and all(x["pass"] and
                                        x["margin_1p2_s_met"] for x in ordered)),
                "cases": ordered,
            }
            temp = DEST / "matrix.json.tmp"
            temp.write_text(json.dumps(matrix, indent=2) + "\n")
            os.replace(temp, DEST / "matrix.json")
            print(json.dumps({"seed": row["seed"], "pass": row["pass"],
                              "contact_s": row["contact_s"],
                              "selection_mode": row["selection_mode"],
                              "faults": row["faults"]}), flush=True)


if __name__ == "__main__":
    main()
