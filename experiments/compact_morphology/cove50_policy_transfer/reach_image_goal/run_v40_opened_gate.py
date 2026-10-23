"""Staged 75-start native retention gate for calibrated route selection."""

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
    run_v35_wider_rgb_screen as core,
    run_v39_corner_image_screen as image,
    run_v40_route_stability_screen as route,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v40"
PLAN = OUT / "opened-gate-plan.json"
DEST = OUT / "opened-native"
STAGE_SIZE = 6


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    old_plan = BASE / "v38/observable-router-gate-plan.json"
    previous = json.loads(old_plan.read_text())
    if len(previous["cases"]) != 75:
        raise ValueError("Expected 75 frozen earlier/recent starts")
    screen_matrix = OUT / "route-stability-screen/matrix.json"
    screened = json.loads(screen_matrix.read_text())
    if not screened["complete"] or screened["pass_count"] != 9:
        raise ValueError("V40 nine-case screen must pass before full retention")
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(image.ACTOR), evaluator=sha(route.EVALUATOR),
                   selector=sha(image.SELECTOR), template=sha(core.TEMPLATE))
    plan = {"schema": "robot-learning-lab.cove50-v40-opened-gate-plan.v1",
            "role": "opened_51_plus_24_retention_not_fresh",
            "body_role": "rejected_historical_direct_frame_sandbox",
            "source_sha256": sources,
            "source_plan_and_matrix_sha256": previous["source_plan_and_matrix_sha256"],
            "v38_plan_sha256": sha(old_plan),
            "v39_gate_stop_sha256": sha(BASE / "v39/opened-native/gate-stop.json"),
            "v40_screen_plan_sha256": sha(route.PLAN),
            "v40_screen_matrix_sha256": sha(screen_matrix),
            "core_runner_sha256": sha(Path(core.__file__)),
            "image_screen_runner_sha256": sha(Path(image.__file__)),
            "route_screen_runner_sha256": sha(Path(route.__file__)),
            "runner_sha256": sha(Path(__file__)),
            "stage_size": STAGE_SIZE,
            "case_count": 75, "cases": previous["cases"],
            "gate": "All 75 older/recent native starts must pass with zero prior-pass regression. Stop at the first failed six-case stage; a new disjoint fresh score follows only on a complete 75/75 gate."}
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V40 full plan differs from source")
    PLAN.write_bytes(encoded)
    return plan


def write_matrix(plan: dict, rows: dict[int, dict], stopped: bool) -> dict:
    ordered = [rows[c["seed"]] for c in plan["cases"] if c["seed"] in rows]
    matrix = {"schema": "robot-learning-lab.cove50-v40-opened-gate.v1",
              "role": "opened_51_plus_24_retention_not_fresh",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "plan_sha256": sha(PLAN), "complete": len(ordered) == 75,
              "stopped_after_failed_stage": stopped,
              "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
              "rescues": [r["seed"] for r in ordered if r["pass"] and not r["baseline_pass"]],
              "regressions": [r["seed"] for r in ordered if not r["pass"] and r["baseline_pass"]],
              "cases": ordered}
    (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= STAGE_SIZE:
        raise ValueError("Use one to six native evaluators")
    v16.SOURCES["actor"] = image.ACTOR
    core.EVALUATOR = route.EVALUATOR
    core.SELECTOR = image.SELECTOR
    core.DEST = DEST
    core.EXPECTED_VARIANT = "v40_calibrated_route_with_observed_corner_refinement_candidate"
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    cases = plan["cases"]
    for start in range(0, len(cases), STAGE_SIZE):
        block = cases[start:start + STAGE_SIZE]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(route.run_case, case, plan): case["seed"] for case in block}
            for future in as_completed(futures):
                row = future.result()
                rows[row["seed"]] = row
                write_matrix(plan, rows, False)
                print(json.dumps({key: row[key] for key in
                                  ("seed", "selection_mode", "route_goal_source",
                                   "pass", "contact_s", "faults")}), flush=True)
        if any(not rows[c["seed"]]["pass"] for c in block):
            matrix = write_matrix(plan, rows, True)
            print(f"stopped after failed stage: {matrix['pass_count']}/{matrix['case_count']}", flush=True)
            break


if __name__ == "__main__":
    main()
