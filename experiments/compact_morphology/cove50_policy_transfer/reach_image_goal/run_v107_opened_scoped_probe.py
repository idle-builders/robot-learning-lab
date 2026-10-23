"""Pair opened V101 contact failures with a prescribed scoped-Y route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as native,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v107"
PLAN = OUT / "opened-scoped-route-plan.json"
DEST = OUT / "native-opened-scoped"
SCORE_PLAN = BASE / "v101/disjoint-reach-plan.json"
SCORE_MATRIX = BASE / "v101/native-disjoint/matrix.json"
EVALUATOR = Path(__file__).with_name("evaluate_forced_scoped_v107.py")
SEEDS = (22021, 22028, 22031)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    scored_plan = json.loads(SCORE_PLAN.read_text())
    scored = json.loads(SCORE_MATRIX.read_text())
    if (not scored["complete"] or scored["score_gate_met"] or
            scored["plan_sha256"] != sha(SCORE_PLAN) or
            tuple(row["seed"] for row in scored["cases"] if not row["pass"]) != SEEDS):
        raise ValueError("Expected complete, failed V101 score")
    source = scored_plan["source_sha256"].copy()
    if source["evaluator"] != sha(native.EVALUATOR):
        raise ValueError("V95 evaluator changed from the scored baseline")
    source["evaluator"] = sha(EVALUATOR)
    score_cases = {case["seed"]: case for case in scored_plan["cases"]}
    rows = {row["seed"]: row for row in scored["cases"]}
    cases = []
    for seed in SEEDS:
        base = {key: value for key, value in score_cases[seed].items()
                if key not in ("id", "baseline_contact_s") and
                not key.startswith("expected_")}
        cases.append({**base, "id": f"scoped-{seed}",
                      "baseline_contact_s": rows[seed]["contact_s"],
                      "baseline_selection_mode": rows[seed]["selection_mode"]})
    plan = {
        "schema": "robot-learning-lab.cove50-v107-opened-scoped-route-plan.v1",
        "role": "opened_postscore_paired_route_diagnostic_not_disjoint_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": source,
        "baseline_evaluator_sha256": sha(native.EVALUATOR),
        "baseline_plan_sha256": sha(SCORE_PLAN),
        "baseline_matrix_sha256": sha(SCORE_MATRIX),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "cases": cases, "case_count": len(cases),
        "method": ("Revisit only the three already opened V101 contact failures, "
                   "keeping their seeds and conditions while forcing the V21 "
                   "scoped-Y arm and -2 mm observed-goal Y offset at reach entry. "
                   "22028 is a same-route negative control. Use the unchanged "
                   "full native gate. This tests a route hypothesis; it does not "
                   "alter the failed V101 score or justify automatic promotion."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V107 plan or sources changed")
    PLAN.write_bytes(encoded)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    print(f"frozen {PLAN} sha256={sha(PLAN)}", flush=True)
    if args.freeze_only:
        return
    DEST.mkdir(parents=True, exist_ok=True)
    matrix_path = DEST / "matrix.json"
    existing = json.loads(matrix_path.read_text()) if matrix_path.exists() else None
    if existing and existing["plan_sha256"] != sha(PLAN):
        raise ValueError("Existing V107 matrix uses another plan")
    rows = {row["id"]: row for row in existing["cases"]} if existing else {}
    original_evaluator = native.EVALUATOR
    native.EVALUATOR = EVALUATOR
    try:
        for case in plan["cases"]:
            if case["id"] in rows:
                continue
            row = native.run_case(case, plan, DEST)
            report = json.loads((ROOT / row["result"]).read_text())
            if (not report.get("diagnostic_force_scoped") or
                    row["selection_mode"] != "v21_scoped_y_locked"):
                raise ValueError("V107 prescribed scoped route was not active")
            row.update({"baseline_selection_mode": case["baseline_selection_mode"],
                        "baseline_contact_s": case["baseline_contact_s"],
                        "contact_delta_s": round(row["contact_s"] -
                                                 case["baseline_contact_s"], 3)})
            rows[row["id"]] = row
            matrix = {
                "schema": "robot-learning-lab.cove50-v107-opened-scoped-route.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN), "complete": len(rows) == len(plan["cases"]),
                "case_count": len(rows),
                "pass_count": sum(value["pass"] for value in rows.values()),
                "cases": [rows[item["id"]] for item in plan["cases"] if item["id"] in rows],
            }
            temporary = DEST / "matrix.json.tmp"
            temporary.write_text(json.dumps(matrix, indent=2) + "\n")
            os.replace(temporary, matrix_path)
            print(json.dumps({key: row[key] for key in
                              ("seed", "baseline_selection_mode", "baseline_contact_s",
                               "selection_mode", "contact_s", "pass", "faults")}),
                  flush=True)
    finally:
        native.EVALUATOR = original_evaluator


if __name__ == "__main__":
    main()
