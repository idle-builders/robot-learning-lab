"""Opened paired native diagnostic for the V88 specialist route boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v88_native_case as native,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v88"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    old_plan_path = BASE / "v87/prior75-retention-plan.json"
    route_plan_path = BASE / "v87/routes63-retention-plan.json"
    old_plan = json.loads(old_plan_path.read_text())
    route_plan = json.loads(route_plan_path.read_text())
    cases = [next(case for case in old_plan["cases"] if case["seed"] == seed)
             for seed in (10514, 10516, 10520, 10522)]
    cases.append(next(case for case in route_plan["cases"]
                      if case["id"] == "large-36-pitch-high"))
    hashes = {**old_plan["source_sha256"], "evaluator": sha(native.EVALUATOR)}
    if hashes["pointer"] != sha(ROOT / "robot/ape-current-design.json"):
        raise ValueError("Mechanical design pointer changed")
    plan = {
        "schema": "robot-learning-lab.cove50-v88-specialist-boundary-probe.v1",
        "role": "opened_paired_native_diagnostic_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": hashes,
        "v87_prior75_plan_sha256": sha(old_plan_path),
        "v87_routes63_plan_sha256": sha(route_plan_path),
        "native_case_runner_sha256": sha(Path(native.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(cases),
        "cases": cases,
        "hypothesis": "The 10514 no-contact regression is a specialist-route boundary error: "
                      "its observed x=1.951 mm belongs to the existing scoped-Y branch "
                      "at x>1.5 mm, while the specialist interval extended to 2.0 mm.",
        "change": "Close the specialist x interval at 1.5 mm, retaining the V87 actor, "
                  "RGB goal, height-only timing, and all other route thresholds.",
        "gate": "The failed 10514 start regains at least 1.2 s native contact and all "
                "five opened cases pass native RGB, contact, motor, support and floor checks. "
                "This diagnostic cannot qualify V88 without complete replay of retained cases.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "specialist-boundary-probe-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen V88 probe/source identity changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    dest = OUT / "native-specialist-boundary-probe"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        row = native.run_case(case, plan, dest)
        rows.append(row)
        matrix = {
            "schema": "robot-learning-lab.cove50-v88-specialist-boundary-probe.v1",
            "role": plan["role"], "body_role": plan["body_role"],
            "plan_sha256": sha(plan_path),
            "complete": len(rows) == len(cases),
            "case_count": len(rows),
            "pass_count": sum(item["pass"] for item in rows),
            "screen_gate_met": (len(rows) == len(cases) and all(item["pass"] for item in rows)
                               and rows[0]["contact_s"] >= 1.2
                               and rows[0]["selection_mode"] == "v21_scoped_y_locked"),
            "cases": rows,
        }
        (dest / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
        print(json.dumps({key: row[key] for key in
                          ("id", "pass", "contact_s", "selection_mode", "faults")}),
              flush=True)


if __name__ == "__main__":
    main()
