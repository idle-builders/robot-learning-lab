"""Gate V31 on the 51 earlier opened native reach starts."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v31"
PLAN = OUT / "prior-gate-plan.json"
SOURCE_PLAN = BASE / "v21/opened-gate-plan.json"
SOURCE_MATRIX = BASE / "v21/opened-native/matrix.json"
PRIOR_OFFLINE = OUT / "prior-offline-report.json"
RECENT_MATRIX = OUT / "opened-native/matrix.json"
DEST = OUT / "prior-native"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError("Use one to six evaluators")
    plan = json.loads(PLAN.read_text())
    prior_offline = json.loads(PRIOR_OFFLINE.read_text())
    recent = json.loads(RECENT_MATRIX.read_text())
    if (plan["runner_sha256"] != sha(Path(__file__)) or
            plan["source_plan_sha256"] != sha(SOURCE_PLAN) or
            plan["source_matrix_sha256"] != sha(SOURCE_MATRIX) or
            plan["prior_offline_report_sha256"] != sha(PRIOR_OFFLINE) or
            plan["recent_opened_plan_sha256"] != sha(OUT / "opened-gate-plan.json") or
            not prior_offline["offline_gate_pass"] or
            not recent["complete"] or not recent["advance_gate_pass"] or
            recent["case_count"] != 24 or
            len(plan["cases"]) != 51 or
            len({case["seed"] for case in plan["cases"]}) != 51 or
            any(case["gaze_head_yaw_rad"] != -1.5 for case in plan["cases"])):
        raise ValueError("Frozen prior retention plan, offline gate, or recent native gate failed")
    from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
        run_v31_opened_template_gaze as core,
        run_v16_opened_matrix as v16,
    )
    sources = {key: sha(path) for key, path in v16.SOURCES.items() if key != "evaluator"}
    sources.update({"evaluator": sha(core.EVALUATOR), "template": sha(core.TEMPLATE)})
    if sources != plan["source_sha256"] or sha(Path(core.__file__)) != plan["core_runner_sha256"]:
        raise ValueError("Prior native source changed")
    baseline_matrix = json.loads(SOURCE_MATRIX.read_text())
    if not baseline_matrix["complete"] or baseline_matrix["case_count"] != 51:
        raise ValueError("Missing full earlier opened baseline")
    baseline = {row["seed"]: row for row in baseline_matrix["cases"]}
    if set(baseline) != {case["seed"] for case in plan["cases"]}:
        raise ValueError("Prior baseline coverage differs")
    core.DEST = DEST
    DEST.mkdir(parents=True, exist_ok=True)
    rows = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(core.run_case, case, plan, baseline[case["seed"]]): case
                   for case in plan["cases"]}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            rows[case["seed"]] = row
            ordered = [rows[case["seed"]] for case in plan["cases"] if case["seed"] in rows]
            counts = Counter(item["selection_mode"] for item in ordered)
            matrix = {"schema": "robot-learning-lab.cove50-v31-prior-native-retention.v1",
                      "role": "earlier_opened_automatic_template_gaze_height_retention_gate_not_fresh",
                      "body_role": "rejected_historical_direct_frame_sandbox",
                      "plan_sha256": sha(PLAN), "source_matrix_sha256": sha(SOURCE_MATRIX),
                      "complete": len(ordered) == 51, "case_count": len(ordered),
                      "baseline_pass_count": sum(item["baseline_pass"] for item in ordered),
                      "pass_count": sum(item["pass"] for item in ordered),
                      "rescues": [item["seed"] for item in ordered if item["pass"] and not item["baseline_pass"]],
                      "regressions": [item["seed"] for item in ordered if item["baseline_pass"] and not item["pass"]],
                      "selection_counts": dict(counts),
                      "advance_gate_pass": len(ordered) == 51 and all(item["pass"] for item in ordered),
                      "cases": ordered}
            (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
            print(json.dumps({"seed": row["seed"], "pass": row["pass"],
                              "contact_s": row["contact_s"],
                              "selection_mode": row["selection_mode"],
                              "faults": row["faults"]}), flush=True)


if __name__ == "__main__":
    main()
