"""Replay opened even V21 starts with a fully attained head turn."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v16_opened_matrix as v16,
    run_v21_fresh_challenge as fresh,
)

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v29"
PLAN = OUT / "opened-gaze-plan.json"
DEST = OUT / "opened-gaze"
SOURCE_PLAN = BASE / "v21/fresh-challenge-plan.json"
SOURCE_MATRIX = BASE / "v21/fresh-reserved/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan = json.loads(PLAN.read_text())
    if (plan["runner_sha256"] != sha(Path(__file__)) or
            plan["source_plan_sha256"] != sha(SOURCE_PLAN) or
            plan["source_matrix_sha256"] != sha(SOURCE_MATRIX) or
            len(plan["cases"]) != 12 or
            {case["seed"] for case in plan["cases"]} != set(range(10602, 10625, 2)) or
            any(case["gaze_head_yaw_rad"] != -1.5 for case in plan["cases"])):
        raise ValueError("Frozen V29 opened gaze plan changed")
    sources = {key: sha(path) for key, path in v16.SOURCES.items() if key != "evaluator"}
    sources["evaluator"] = sha(fresh.EVALUATOR)
    if sources != plan["source_sha256"]:
        raise ValueError("Native source changed")
    old = json.loads(SOURCE_MATRIX.read_text())
    by_seed = {row["seed"]: row for row in old["cases"]}
    if not old["complete"] or old["case_count"] != 24:
        raise ValueError("Need complete opened V21 baseline")
    fresh.DEST = DEST
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fresh.run_case, case, plan): case for case in plan["cases"]}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            baseline = by_seed[case["seed"]]
            row["baseline_pass"] = baseline["pass"]
            row["baseline_contact_s"] = baseline["contact_s"]
            row["baseline_gaze_head_yaw_rad"] = -1.3
            row["new_gaze_head_yaw_rad"] = -1.5
            rows[case["seed"]] = row
            ordered = [rows[case["seed"]] for case in plan["cases"] if case["seed"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v29-opened-gaze.v1",
                      "role": "postscore_opened_gaze_change_not_fresh",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == 12,
                      "case_count": len(ordered), "pass_count": sum(item["pass"] for item in ordered),
                      "baseline_pass_count": sum(item["baseline_pass"] for item in ordered),
                      "rescues": [item["seed"] for item in ordered if item["pass"] and not item["baseline_pass"]],
                      "regressions": [item["seed"] for item in ordered if item["baseline_pass"] and not item["pass"]],
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({"seed": row["seed"], "pass": row["pass"],
                              "contact_s": row["contact_s"],
                              "selection_mode": row["selection_mode"],
                              "entry_goal_error_mm_scoring_only": row["entry_goal_error_mm_scoring_only"],
                              "faults": row["faults"]}), flush=True)


if __name__ == "__main__":
    main()
