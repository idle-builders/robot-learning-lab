"""Full 51-old plus 24-recent native retention gate for the V39 candidate."""

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
    run_v39_corner_image_screen as screen,
)

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v39"
PLAN = OUT / "opened-gate-plan.json"
DEST = OUT / "opened-native"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    prior = BASE / "v38/observable-router-gate-plan.json"
    old = json.loads(prior.read_text())
    if old["case_count"] != 75:
        raise ValueError("Expected frozen V38 opened 75 cases")
    sources = {name: sha(path) for name, path in v16.SOURCES.items()}
    sources.update(actor=sha(screen.ACTOR), evaluator=sha(screen.EVALUATOR),
                   selector=sha(screen.SELECTOR), template=sha(core.TEMPLATE))
    plan = {"schema": "robot-learning-lab.cove50-v39-opened-gate-plan.v1",
            "role": "opened_51_plus_24_retention_not_fresh",
            "body_role": "rejected_historical_direct_frame_sandbox",
            "source_sha256": sources,
            "source_plan_and_matrix_sha256": old["source_plan_and_matrix_sha256"],
            "v38_plan_sha256": sha(prior),
            "v39_screen_plan_sha256": sha(screen.PLAN),
            "core_runner_sha256": sha(Path(core.__file__)),
            "screen_runner_sha256": sha(Path(screen.__file__)),
            "runner_sha256": sha(Path(__file__)),
            "case_count": len(old["cases"]), "cases": old["cases"],
            "gate": "All 75 opened starts must pass actual native contact, support, motor and RGB gates with zero regressions. New disjoint challenge remains separate.",
            "inference": "V36 actor, V38 observable route and V39 bounded rendered-image correction; no toy truth or seed enters inference."}
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2).encode() + b"\n"
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V39 full gate differs from source")
    PLAN.write_bytes(encoded)
    return plan


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
    v16.SOURCES["actor"] = screen.ACTOR
    core.EVALUATOR = screen.EVALUATOR
    core.SELECTOR = screen.SELECTOR
    core.DEST = DEST
    core.EXPECTED_VARIANT = "v39_v36_actor_observable_route_and_corner_image_candidate"
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(screen.run_case, case, plan): case["seed"]
                   for case in plan["cases"]}
        for future in as_completed(futures):
            row = future.result()
            rows[row["seed"]] = row
            ordered = [rows[c["seed"]] for c in plan["cases"] if c["seed"] in rows]
            matrix = {"schema": "robot-learning-lab.cove50-v39-opened-gate.v1",
                      "role": "opened_51_plus_24_retention_not_fresh",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "complete": len(ordered) == len(plan["cases"]),
                      "case_count": len(ordered), "pass_count": sum(r["pass"] for r in ordered),
                      "rescues": [r["seed"] for r in ordered if r["pass"] and not r["baseline_pass"]],
                      "regressions": [r["seed"] for r in ordered if not r["pass"] and r["baseline_pass"]],
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in
                              ("seed", "pass", "contact_s", "faults")}), flush=True)


if __name__ == "__main__":
    main()
