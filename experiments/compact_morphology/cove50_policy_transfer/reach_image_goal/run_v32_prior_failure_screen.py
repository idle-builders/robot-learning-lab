"""Screen the nine earlier V21 misses before spending on full native retention."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v32"
PLAN = OUT / "prior-failure-screen-plan.json"
RECENT = OUT / "opened-native/matrix.json"
SOURCE_PLAN = BASE / "v21/opened-gate-plan.json"
SOURCE_MATRIX = BASE / "v21/opened-native/matrix.json"
DEST = OUT / "prior-failure-screen"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six native evaluators")
    plan = json.loads(PLAN.read_text())
    recent = json.loads(RECENT.read_text())
    baseline = json.loads(SOURCE_MATRIX.read_text())
    if (plan["runner_sha256"] != sha(Path(__file__)) or
            plan["core_runner_sha256"] != sha(Path(__file__).with_name("run_v32_opened_selective.py")) or
            plan["recent_opened_plan_sha256"] != sha(OUT / "opened-gate-plan.json") or
            plan["source_plan_sha256"] != sha(SOURCE_PLAN) or
            plan["source_matrix_sha256"] != sha(SOURCE_MATRIX) or
            not recent["complete"] or not recent["advance_gate_pass"] or
            not baseline["complete"] or baseline["case_count"] != 51 or
            len(plan["cases"]) != 9 or
            {case["seed"] for case in plan["cases"]} !=
            {row["seed"] for row in baseline["cases"] if not row["pass"]}):
        raise ValueError("Need sealed plans, complete V32 recent gate, and exact earlier failures")
    from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
        run_v32_opened_selective as core,
        run_v16_opened_matrix as v16,
    )
    sources = {key: sha(path) for key, path in v16.SOURCES.items() if key != "evaluator"}
    sources.update({"evaluator": sha(core.EVALUATOR), "template": sha(core.TEMPLATE),
                    "selector": sha(core.SELECTOR)})
    if sources != plan["source_sha256"]:
        raise ValueError("V32 native source changed")
    old = {row["seed"]: row for row in baseline["cases"]}
    core.DEST = DEST
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(core.run_case, case, plan, old[case["seed"]]): case
                   for case in plan["cases"]}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            rows[case["seed"]] = row
            ordered = [rows[case["seed"]] for case in plan["cases"] if case["seed"] in rows]
            matrix = {
                "schema": "robot-learning-lab.cove50-v32-prior-failure-screen.v1",
                "role": "earlier_opened_failure_screen_not_fresh",
                "body_role": "rejected_historical_direct_frame_sandbox",
                "plan_sha256": sha(PLAN),
                "complete": len(ordered) == 9,
                "case_count": len(ordered),
                "pass_count": sum(item["pass"] for item in ordered),
                "rescues": [item["seed"] for item in ordered if item["pass"]],
                "advance_gate_pass": len(ordered) == 9 and all(item["pass"] for item in ordered),
                "cases": ordered,
            }
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({"seed": row["seed"], "pass": row["pass"],
                              "contact_s": row["contact_s"], "faults": row["faults"]}),
                  flush=True)


if __name__ == "__main__":
    main()
