"""Repeat one thin V70 contact in fresh native processes without retuning."""

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
    run_v70_center_route_screen as v70,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v77"
PLAN = OUT / "thin-contact-repeatability-plan.json"
DEST = OUT / "native-repeats"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    source_plan = json.loads(v70.PLAN.read_text())
    source_matrix = json.loads((v70.DEST / "matrix.json").read_text())
    anchor = next(case for case in source_plan["cases"] if case["id"] == "large-20-center")
    scored = next(row for row in source_matrix["cases"] if row["id"] == anchor["id"])
    if (not source_matrix["screen_gate_met"] or source_matrix["pass_count"] != 35 or
            not scored["pass"] or not 1 <= scored["contact_s"] < 1.2 or
            scored["automatic_front_center_bias_applied"] or
            scored["automatic_front_route_bias_applied"] or
            scored["automatic_edge_route_bias_applied"]):
        raise ValueError("Expected the thin V70 large-20 center visit with no route correction")
    cases = [dict(anchor, id=f"large-20-center-repeat-{index}") for index in (1, 2, 3)]
    plan = {
        "schema": "robot-learning-lab.cove50-v77-thin-contact-repeatability-plan.v1",
        "role": "opened_exact_case_repeatability_diagnostic_not_fresh_score",
        "body_role": source_plan["body_role"],
        "v70_plan_sha256": sha(v70.PLAN),
        "v70_matrix_sha256": sha(v70.DEST / "matrix.json"),
        "source_sha256": source_plan["source_sha256"],
        "v70_runner_sha256": sha(Path(v70.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "original_v70_contact_s": scored["contact_s"],
        "case_count": len(cases), "cases": cases,
        "gate": "Each of three fresh native processes must pass the exact V70 full gate. Record the minimum contact and whether every contact exceeds the separate 1.2 s diagnostic margin. Repetition of one opened case does not establish variation robustness.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V77 repeatability plan differs from source")
    PLAN.write_bytes(encoded)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    v70.DEST = DEST
    rows = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(v70.run_case, case, plan): case["id"]
                   for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
            contacts = [row["contact_s"] for row in ordered]
            matrix = {
                "schema": "robot-learning-lab.cove50-v77-thin-contact-repeatability.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN),
                "complete": len(ordered) == plan["case_count"],
                "case_count": len(ordered), "pass_count": sum(row["pass"] for row in ordered),
                "minimum_contact_s": min(contacts),
                "maximum_contact_s": max(contacts),
                "full_gate_met": len(ordered) == plan["case_count"] and all(row["pass"] for row in ordered),
                "margin_1p2_s_met": len(ordered) == plan["case_count"] and all(row["pass"] and row["contact_s"] >= 1.2 for row in ordered),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("id", "seed", "pass", "contact_s", "faults")}), flush=True)


if __name__ == "__main__":
    main()
