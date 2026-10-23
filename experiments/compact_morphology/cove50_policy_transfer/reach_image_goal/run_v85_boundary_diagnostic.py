"""Open the V84 negative-Y height-route failures with one frozen V85 change."""

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
    run_v85_native_case as native,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v84"
OUT = BASE.parent / "v85"
PLAN = OUT / "boundary-diagnostic-plan.json"
DEST = OUT / "native-boundary-diagnostic"
SEEDS = (10505, 10507, 10501, 10504)
RESCUES = {10505, 10507}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    source_plan_path = BASE / "prior75-retention-plan.json"
    source_matrix_path = BASE / "native-prior75-retention/matrix.json"
    source = json.loads(source_plan_path.read_text())
    opened = json.loads(source_matrix_path.read_text())
    by_seed = {case["seed"]: case for case in source["cases"]}
    observed = {row["seed"]: row for row in opened["cases"]}
    if not set(SEEDS) <= observed.keys() or any(observed[seed]["pass"] for seed in RESCUES):
        raise ValueError("Need both V84 failures and both passing native controls")
    if any(not observed[seed]["pass"] for seed in set(SEEDS) - RESCUES):
        raise ValueError("V84 controls did not pass")
    cases = []
    for seed in SEEDS:
        case = {**by_seed[seed], "id": f"boundary-{seed}",
                "v84_contact_s": observed[seed]["contact_s"],
                "v84_selection_mode": observed[seed]["selection_mode"],
                "expected_center_live_lock": False,
                "expected_thin_route_bias": False,
                "expected_selection_mode": "v16_v10_height_locked"}
        cases.append(case)
    hashes = {**source["source_sha256"], "evaluator": sha(native.EVALUATOR)}
    plan = {
        "schema": "robot-learning-lab.cove50-v85-boundary-diagnostic-plan.v1",
        "role": "post_v84_older_failure_opened_route_diagnostic_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_plan_sha256": sha(source_plan_path),
        "source_matrix_sha256": sha(source_matrix_path),
        "source_sha256": hashes,
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases), "cases": cases,
        "gate": ("Fresh native processes on two V84 failed negative-Y center starts and "
                 "two passing controls. All four must pass full support, rated effort, "
                 "RGB and contact gates; only the two failed starts may switch from "
                 "plain V10 to height-assisted V10, all four must select height, and "
                 "rescued contact must reach at least 1.2 s. This opened diagnostic "
                 "cannot promote the controller or qualify a selected body."),
        "change": ("Extend the height-assisted V10 route boundary from X < -12 mm "
                   "to X < -11.5 mm only when observed entry Y < -7 mm. Keep actor, "
                   "body, camera, motor, timing, thin-route and center-lock rules fixed."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        prior = json.loads(PLAN.read_text())
        if prior != plan:
            raise ValueError("Frozen V85 plan/source identity changed")
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
        futures = {pool.submit(native.run_case, case, plan, DEST): case for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            case = futures[future]
            if row["selection_mode"] != case["expected_selection_mode"]:
                row["faults"].append("height-route selection differs from frozen expectation")
                row["pass"] = False
            rows[row["id"]] = row
            ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
            complete = len(ordered) == len(plan["cases"])
            rescued = [row for row in ordered if row["seed"] in RESCUES]
            gate = (complete and all(row["pass"] for row in ordered) and
                    len(rescued) == 2 and all(row["contact_s"] >= 1.2 for row in rescued) and
                    not any(row["automatic_center_live_lock_applied"] or
                            row["automatic_thin_route_bias_applied"] for row in ordered))
            matrix = {
                "schema": "robot-learning-lab.cove50-v85-boundary-diagnostic.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN), "complete": complete,
                "case_count": len(ordered),
                "pass_count": sum(row["pass"] for row in ordered),
                "minimum_contact_s": min(row["contact_s"] for row in ordered),
                "diagnostic_gate_met": gate, "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({"seed": row["seed"], "pass": row["pass"],
                              "selection_mode": row["selection_mode"],
                              "contact_s": row["contact_s"],
                              "faults": row["faults"]}), flush=True)


if __name__ == "__main__":
    main()
