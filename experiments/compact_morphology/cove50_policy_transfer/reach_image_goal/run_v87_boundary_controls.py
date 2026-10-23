"""Opened native V87 screen of earlier passing routes adjacent to the boundary."""

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
    run_v87_native_case as native,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v87"
PLAN = OUT / "boundary-controls-plan.json"
DEST = OUT / "native-boundary-controls"
SEEDS = (10526, 10531, 10527, 10528)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    source_path = BASE / "v84/prior75-retention-plan.json"
    source = json.loads(source_path.read_text())
    route_path = OUT / "routes63-retention-plan.json"
    route = json.loads(route_path.read_text())
    by_seed = {case["seed"]: case for case in source["cases"]}
    if (route["source_sha256"]["evaluator"] != sha(native.EVALUATOR) or
            route["native_case_runner_sha256"] != sha(Path(native.__file__)) or
            not set(SEEDS) <= by_seed.keys()):
        raise ValueError("Missing source-matched earlier boundary controls")
    cases = [{**by_seed[seed], "id": f"control-{seed}",
              "prior_v41_route": "v16_v10_locked" if seed in (10526, 10531)
              else "v16_v10_height_locked",
              "expected_center_live_lock": False,
              "expected_thin_route_bias": False} for seed in SEEDS]
    plan = {
        "schema": "robot-learning-lab.cove50-v87-boundary-controls-plan.v1",
        "role": "opened_prior_passes_near_route_boundary_not_disjoint_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": route["source_sha256"],
        "source_v84_older_plan_sha256": sha(source_path),
        "v87_route_plan_sha256": sha(route_path),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": 4, "cases": cases,
        "gate": ("Four formerly passing older starts adjacent to the observed height "
                 "boundary must all pass fresh native RGB, contact, support, rated "
                 "effort, floor, self-contact and clipping gates and hold at least "
                 "1.2 s contact. Two earlier plain-V10 and two height-route choices "
                 "are recorded; V87 may choose either if contact passes. No thin or "
                 "center-lock rule may activate. This opened screen does not replace "
                 "all 75 older retention visits or a new-seed score."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    if PLAN.exists():
        if json.loads(PLAN.read_text()) != plan:
            raise ValueError("Frozen V87 boundary controls plan changed")
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
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(native.run_case, case, plan, DEST): case
                   for case in plan["cases"]}
        rows = {}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            row["prior_v41_route"] = case["prior_v41_route"]
            rows[row["id"]] = row
            ordered = [rows[case["id"]] for case in plan["cases"] if case["id"] in rows]
            complete = len(ordered) == 4
            matrix = {
                "schema": "robot-learning-lab.cove50-v87-boundary-controls.v1",
                "role": plan["role"], "body_role": plan["body_role"],
                "plan_sha256": sha(PLAN), "complete": complete,
                "case_count": len(ordered),
                "pass_count": sum(row["pass"] for row in ordered),
                "minimum_contact_s": min(row["contact_s"] for row in ordered),
                "controls_gate_met": (complete and all(row["pass"] and
                                      row["margin_1p2_s_met"] and
                                      not row["automatic_thin_route_bias_applied"] and
                                      not row["automatic_center_live_lock_applied"]
                                      for row in ordered)),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({"seed": row["seed"], "pass": row["pass"],
                              "prior_route": row["prior_v41_route"],
                              "v87_route": row["selection_mode"],
                              "contact_s": row["contact_s"],
                              "faults": row["faults"]}), flush=True)


if __name__ == "__main__":
    main()
