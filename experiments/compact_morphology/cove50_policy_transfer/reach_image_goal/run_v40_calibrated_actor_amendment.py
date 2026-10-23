"""Actual calibrated-only V21 actor comparison after V38 retained template matching."""

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
    run_v40_regression_diagnostic as earlier,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v40/postscore-calibrated-actor-amendment"
PLAN = OUT / "plan.json"
EVALUATOR = Path(__file__).with_name("evaluate_adaptive_goal_v21.py")
STOP = BASE / "v40/opened-native/gate-stop.json"
PRIOR_MATRIX = BASE / "v40/postscore-regression-diagnostic/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    prior = json.loads(PRIOR_MATRIX.read_text())
    if not prior["complete"] or prior["case_count"] != 4:
        raise ValueError("First postscore ablation must be complete")
    start = next(c for c in json.loads((BASE / "v40/opened-gate-plan.json").read_text())["cases"]
                 if c["seed"] == 10546)
    visits = [{"id": f"10546-v21-{actor}", "seed": 10546,
               "evaluator": "v21", "actor": actor, "start": start}
              for actor in ("v13", "v36")]
    sources = {**{f"actor_{name}": sha(path) for name, path in earlier.ACTORS.items()},
               "evaluator_v21": sha(EVALUATOR), "body": sha(earlier.v16.SOURCES["body"]),
               "pointer": sha(earlier.v16.SOURCES["pointer"]),
               "prior_runner": sha(Path(earlier.__file__)), "runner": sha(Path(__file__))}
    plan = {"schema": "robot-learning-lab.cove50-v40-calibrated-actor-amendment-plan.v1",
            "role": "opened_postscore_diagnostic_not_promotion",
            "body_role": "rejected_historical_direct_frame_sandbox",
            "why_amended": "V38 also applied the sparse template at 10546, so the first V38/V40 contrast did not isolate calibrated-only RGB. V21 is the frozen calibrated-only evaluator. This amendment is separate from the first diagnostic and leaves its results intact.",
            "gate_stop_sha256": sha(STOP), "prior_diagnostic_matrix_sha256": sha(PRIOR_MATRIX),
            "source_sha256": sources, "visits": visits}
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.is_file() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen amendment plan changed")
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
    earlier.OUT = OUT
    earlier.EVALUATORS["v21"] = EVALUATOR
    rows = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(earlier.run, visit, plan): visit["id"]
                   for visit in plan["visits"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["id"]] = row
            ordered = [rows[visit["id"]] for visit in plan["visits"] if visit["id"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v40-calibrated-actor-amendment.v1",
                      "role": "opened_postscore_diagnostic_not_promotion",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == 2,
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "cases": ordered}
            (OUT / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
