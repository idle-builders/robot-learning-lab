"""Evaluate the center-and-roll controller on predeclared correction views."""

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
    run_v16_opened_matrix as v16,
    run_v41_opened_screen as screen,
    run_v57_contact_actor_screen as v57,
    run_v70_center_route_screen as v70,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v74"
MANIFEST = BASE / "v67/predeclared-branch-view-holdout.json"
PLAN = OUT / "branch-view-holdout-run-plan.json"
DEST = OUT / "opened-branch-view-holdout"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    source_matrix = v70.DEST / "matrix.json"
    source = json.loads(source_matrix.read_text())
    if (not source["complete"] or not source["screen_gate_met"] or
            source["pass_count"] != 35 or manifest["case_count"] != 12):
        raise ValueError("Predeclared branch holdout requires V70 full-gate 35/35 screen")
    tuned_anchors = {row["anchor_id"] for row in source["cases"]}
    if any(row["anchor_id"] in tuned_anchors for row in manifest["cases"]
           if row["branch_view_family"] == "different_anchor_front_view"):
        raise ValueError("A front-view holdout anchor occurred in the V70 tuning matrix")
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(v57.ACTOR), evaluator=sha(v70.EVALUATOR),
                   selector=sha(v70.SELECTOR), template=sha(screen.core.TEMPLATE),
                   v70_runner=sha(Path(v70.__file__)))
    plan = {
        "schema": "robot-learning-lab.cove50-v74-branch-view-holdout-run-plan.v1",
        "role": manifest["role"], "body_role": manifest["body_role"],
        "manifest_sha256": sha(MANIFEST),
        "v70_plan_sha256": sha(v70.PLAN), "v70_matrix_sha256": sha(source_matrix),
        "source_sha256": sources, "runner_sha256": sha(Path(__file__)),
        "case_count": manifest["case_count"], "cases": manifest["cases"],
        "gate": manifest["gate"],
        "design": manifest["design"] + " Replay the frozen V70 automatic controller with no route or image threshold changes after seeing its V70 screen.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V74 branch-view plan differs from current source")
    PLAN.write_bytes(encoded)
    return plan


def run_case(case: dict, plan: dict) -> dict:
    # Reuse the exact native gate and observable-rule validator frozen for V70.
    row = v70.run_case(case, plan)
    for key in ("baseline_v60_contact_s", "baseline_v60_pass",
                "baseline_v62_contact_s", "baseline_v62_pass"):
        row.pop(key)
    row["baseline_v59_center_contact_s"] = case["baseline_v59_center_contact_s"]
    row["branch_view_family"] = case["branch_view_family"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    DEST.mkdir(parents=True, exist_ok=True)
    v70.DEST = DEST
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case, plan): case["id"] for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
            roll_uses = sum(row["automatic_front_route_bias_y_m"] == -.006
                            for row in ordered)
            edge_uses = sum(row["automatic_edge_route_bias_applied"] for row in ordered)
            matrix = {
                "schema": "robot-learning-lab.cove50-v74-branch-view-holdout-screen.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN),
                "complete": len(ordered) == len(plan["cases"]),
                "case_count": len(ordered), "pass_count": sum(row["pass"] for row in ordered),
                "roll_front_bias_use_count": roll_uses,
                "edge_route_bias_use_count": edge_uses,
                "screen_gate_met": len(ordered) == len(plan["cases"]) and
                                   all(row["pass"] for row in ordered) and
                                   roll_uses >= 3 and edge_uses >= 3,
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "seed", "selection_mode", "pass", "contact_s", "faults")}),
                  flush=True)


if __name__ == "__main__":
    main()
