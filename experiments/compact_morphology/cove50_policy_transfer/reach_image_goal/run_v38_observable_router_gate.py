"""Native opened retention gate for V36 arm and V38 observable route changes."""

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
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v38"
PLAN = OUT / "observable-router-gate-plan.json"
DEST = OUT / "opened-native"
ACTOR = BASE / "v36/actor.pt"
EVALUATOR = Path(__file__).with_name("evaluate_selective_goal_v38.py")
COHORTS = (("v21", "opened-native"), ("v32", "opened-native"))
PILOT = {10513, 10534, 10538, 10539, 10433, 10509, 10520, 10521,
         10537, 10542, 10543, 10514, 10530, 10501, 10508,
         10608, 10612, 10620, 10621}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    rows = []
    identity = {}
    for version, directory in COHORTS:
        matrix = BASE / version / directory / "matrix.json"
        case_plan = BASE / version / "opened-gate-plan.json"
        old = json.loads(matrix.read_text())
        inputs = {c["seed"]: c for c in json.loads(case_plan.read_text())["cases"]}
        if not old["complete"] or len(old["cases"]) != old["case_count"]:
            raise ValueError(f"Incomplete {matrix}")
        identity[str(matrix.relative_to(ROOT))] = sha(matrix)
        identity[str(case_plan.relative_to(ROOT))] = sha(case_plan)
        for row in old["cases"]:
            source = inputs[row["seed"]]
            rows.append({"seed": row["seed"], "cohort": version,
                         "role": "prior_failure" if not row["pass"] else "retention",
                         "baseline_pass": row["pass"],
                         "baseline_contact_s": row["contact_s"],
                         "baseline_trace": row["trace"],
                         **{key: source[key] for key in
                            ("rail_v", "delay_ms", "toy_offset_x_m",
                             "toy_offset_y_m", "gaze_head_yaw_rad")}})
    if len(rows) != 75 or len({r["seed"] for r in rows}) != 75:
        raise ValueError("Expected 51 historical and 24 recent unique starts")
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(ACTOR), evaluator=sha(EVALUATOR),
                   selector=sha(core.SELECTOR), template=sha(core.TEMPLATE))
    plan = {"schema": "robot-learning-lab.cove50-v38-opened-gate-plan.v1",
            "role": "opened_51_plus_24_retention_not_fresh",
            "body_role": "rejected_historical_direct_frame_sandbox",
            "source_sha256": sources,
            "source_plan_and_matrix_sha256": identity,
            "core_runner_sha256": sha(Path(core.__file__)),
            "runner_sha256": sha(Path(__file__)),
            "v36_training_report_sha256": sha(BASE / "v36/training-report.json"),
            "case_count": len(rows), "pilot_seeds": sorted(PILOT), "cases": rows,
            "gate": "All 75 opened native starts must pass. Verify all preceding passes remain; disjoint fresh scoring follows only after this gate.",
            "inference": "V35 pixels and calibrated RGB, V36 actor, V38 routes from entry RGB error and camera pose; no seed or truth at inference."}
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V38 plan differs from source")
    PLAN.write_bytes(encoded)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    v16.SOURCES["actor"] = ACTOR
    core.EVALUATOR = EVALUATOR
    core.DEST = DEST
    core.EXPECTED_VARIANT = "v38_v36_actor_observable_gap_router_candidate"
    DEST.mkdir(parents=True, exist_ok=True)
    cases = [c for c in plan["cases"] if args.stage == "full" or c["seed"] in PILOT]
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(core.run_case, c, plan): c["seed"] for c in cases}
        for future in as_completed(futures):
            row = future.result()
            rows[row["seed"]] = row
            ordered = [rows[c["seed"]] for c in cases if c["seed"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v38-opened-gate.v1",
                      "role": "opened_51_plus_24_retention_not_fresh",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "stage": args.stage,
                      "complete": len(ordered) == len(cases),
                      "case_count": len(ordered),
                      "pass_count": sum(row["pass"] for row in ordered),
                      "rescues": [r["seed"] for r in ordered if r["pass"] and not r["baseline_pass"]],
                      "regressions": [r["seed"] for r in ordered if not r["pass"] and r["baseline_pass"]],
                      "cases": ordered}
            (DEST / f"{args.stage}-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("seed", "pass", "contact_s", "faults")}), flush=True)


if __name__ == "__main__":
    main()
