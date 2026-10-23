"""Force saved modes with the V36 trained arm on remaining opened misses."""

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
    run_v33_forced_route_screen as core,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v37"
PLAN = OUT / "new-actor-forced-route-plan.json"
DEST = OUT / "new-actor-forced-route"
ACTOR = BASE / "v36/actor.pt"
SOURCE_PLAN = BASE / "v32/prior-failure-screen-plan.json"
SOURCE_MATRIX = BASE / "v36/actor-screen/matrix.json"
VISITS = (
    (10534, "locked", "v16_locked", 0.0, 0.0),
    (10534, "specialist", "v16_specialist_locked", 0.0, 0.0),
    (10539, "v10", "v16_v10_locked", 0.0, 0.0),
    (10539, "specialist", "v16_specialist_locked", 0.0, 0.0),
    (10513, "live", "v16_live", 0.0, 0.0),
    (10513, "v10", "v16_v10_locked", 0.0, 0.0),
    (10513, "specialist", "v16_specialist_locked", 0.0, 0.0),
    (10513, "bias_y", "v16_bias_y_locked", 0.0, 0.0),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    matrix = json.loads(SOURCE_MATRIX.read_text())
    prior = {row["seed"]: row for row in matrix["cases"]}
    if (not matrix["complete"] or matrix["case_count"] != 10 or
            any(prior[seed]["pass"] for seed in (10534, 10539, 10513))):
        raise ValueError("Expected three unresolved V36 automatic misses")
    cases = {row["seed"]: row for row in json.loads(SOURCE_PLAN.read_text())["cases"]}
    hashes = {name: sha(path) for name, path in v16.SOURCES.items()}
    hashes["actor"] = sha(ACTOR)
    plan = {
        "schema": "robot-learning-lab.cove50-v37-new-actor-forced-route-plan.v1",
        "role": "opened_new_actor_forced_route_diagnostic_not_automatic_not_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_plan_sha256": sha(SOURCE_PLAN),
        "source_matrix_sha256": sha(SOURCE_MATRIX),
        "source_actor_training_report_sha256": sha(BASE / "v36/training-report.json"),
        "source_sha256": hashes,
        "evaluator_sha256": sha(core.EVALUATOR),
        "core_runner_sha256": sha(Path(core.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(VISITS),
        "cases": [
            {"seed": seed, "route": route, "force_selection_mode": mode,
             "extra_goal_bias_body_xy_mm": [x, y], "role": "unresolved_v36_failure",
             **{key: cases[seed][key] for key in
                ("rail_v", "delay_ms", "toy_offset_x_m", "toy_offset_y_m", "gaze_head_yaw_rad")}}
            for seed, route, mode, x, y in VISITS
        ],
        "qualification": "Experimenter forces modes on already opened failures using the V36 actor and calibrated RGB. These are teacher probes, not an automatic selector, retention gate, or fresh score. A success requires observable routing and paired previous-pass tests before promotion.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists():
        if PLAN.read_bytes() != encoded:
            raise ValueError("Frozen V37 plan differs from source")
    else:
        PLAN.write_bytes(encoded)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    v16.SOURCES["actor"] = ACTOR
    core.DEST = DEST
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(core.run_case, case, plan): (case["seed"], case["route"])
                   for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[(row["seed"], row["route"])] = row
            ordered = [rows[(case["seed"], case["route"])] for case in plan["cases"]
                       if (case["seed"], case["route"]) in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v37-new-actor-forced-route.v1",
                      "role": "opened_new_actor_forced_route_diagnostic_not_automatic_not_fresh",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN),
                      "complete": len(ordered) == len(plan["cases"]),
                      "case_count": len(ordered),
                      "pass_count": sum(row["pass"] for row in ordered),
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("seed", "route", "pass", "contact_s", "faults")}), flush=True)


if __name__ == "__main__":
    main()
