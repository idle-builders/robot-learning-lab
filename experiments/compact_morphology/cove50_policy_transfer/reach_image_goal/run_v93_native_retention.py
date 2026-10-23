"""Source-frozen native retention ladder for the observed rear-contact correction."""

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
    run_v93_native_case as native,
    run_v88_native_recheck as v88,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v93"
SOURCES = {
    "variance16": ("boundary-variance-plan.json", 16),
    "controls4": ("boundary-controls-plan.json", 4),
    "routes63": ("routes63-retention-plan.json", 63),
    "prior75": ("prior75-retention-plan.json", 75),
}
PREREQUISITES = {
    "variance16": (),
    "controls4": ("variance16",),
    "routes63": ("controls4",),
    "prior75": ("routes63",),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(stage: str) -> tuple[dict, Path, Path]:
    pilot_plan_path = OUT / "observed-bias-pilot-plan.json"
    pilot_matrix_path = OUT / "native-observed-bias-pilot/matrix.json"
    pilot = json.loads(pilot_matrix_path.read_text())
    pilot_plan = json.loads(pilot_plan_path.read_text())
    if (not pilot["complete"] or not pilot["development_gate_met"] or
            pilot["plan_sha256"] != sha(pilot_plan_path)):
        raise ValueError("Need passing source-matched V93 automatic pilot")
    evidence = {"pilot": sha(pilot_matrix_path)}
    for prerequisite in PREREQUISITES[stage]:
        earlier_path = OUT / f"native-{prerequisite}-retention/matrix.json"
        earlier = json.loads(earlier_path.read_text())
        if not earlier["complete"] or not earlier["screen_gate_met"]:
            raise ValueError(f"Need passing V93 {prerequisite} retention before {stage}")
        evidence[prerequisite] = sha(earlier_path)
    source_name, count = SOURCES[stage]
    source_path = BASE / "v87" / source_name
    source = json.loads(source_path.read_text())
    if len(source["cases"]) != count:
        raise ValueError("V87 retained case count changed")
    hashes = pilot_plan["source_sha256"]
    if (hashes["evaluator"] != sha(native.EVALUATOR) or
            hashes["pointer"] != sha(ROOT / "robot/ape-current-design.json") or
            hashes["actor"] != sha(native.v83.v57.ACTOR)):
        raise ValueError("Frozen V93 source or selected mechanical pointer changed")
    cases = [{**case,
              "expected_rear_contact_bias": bool(stage == "prior75" and case["seed"] == 10433)}
             for case in source["cases"]]
    plan = {
        "schema": f"robot-learning-lab.cove50-v93-{stage}-retention-plan.v1",
        "role": f"opened_{stage}_automatic_rear_contact_retention_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": hashes,
        "source_v87_plan_sha256": sha(source_path),
        "prerequisite_matrix_sha256": evidence,
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": count, "cases": cases,
        "change": ("V93 applies the bounded observed rear-camera −1 mm reach-goal "
                   "correction after V88 and V89 route changes. Robot, checkpoints, "
                   "RGB observer, motor model and timing remain source-frozen."),
        "gate": ("Every native visit must pass the full contact, RGB, support, motor and "
                 "collision gate; retain the original V88 route and thin-rule exposures. "
                 "The rear correction must activate only on the predeclared 10433 "
                 "older-start exposure. Stop promotion after any failed stage."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / f"{stage}-retention-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen V93 retention plan or source identity changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan, plan_path, OUT / f"native-{stage}-retention"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=SOURCES)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan, plan_path, dest = freeze(args.stage)
    if args.freeze_only:
        print(f"frozen {plan_path} sha256={sha(plan_path)}")
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
            rows[row["id"]] = row
            ordered = [rows[item["id"]] for item in plan["cases"] if item["id"] in rows]
            complete = len(ordered) == len(plan["cases"])
            expected_exposure = all(item["automatic_rear_contact_bias_applied"] ==
                                    item["expected_rear_contact_bias"] for item in ordered)
            route_gate = v88.rule_gate(args.stage, ordered)
            matrix = {
                "schema": f"robot-learning-lab.cove50-v93-{args.stage}-retention.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(plan_path), "complete": complete,
                "case_count": len(ordered), "pass_count": sum(item["pass"] for item in ordered),
                "minimum_contact_s": min(item["contact_s"] for item in ordered),
                "rear_bias_use_count": sum(item["automatic_rear_contact_bias_applied"]
                                           for item in ordered),
                "rear_bias_exposure_gate_met": expected_exposure,
                "route_rule_gate_met": route_gate,
                "screen_gate_met": (complete and expected_exposure and route_gate and
                                    all(item["pass"] for item in ordered)),
                "cases": ordered,
            }
            temp = dest / "matrix.json.tmp"
            temp.write_text(json.dumps(matrix, indent=2) + "\n")
            os.replace(temp, dest / "matrix.json")
            print(json.dumps({key: row[key] for key in
                              ("id", "pass", "contact_s", "selection_mode",
                               "automatic_rear_contact_bias_applied", "faults")}), flush=True)
            next_case = next(iterator, None)
            if next_case is not None:
                pending[pool.submit(native.run_case, next_case, plan, dest)] = next_case


if __name__ == "__main__":
    main()
