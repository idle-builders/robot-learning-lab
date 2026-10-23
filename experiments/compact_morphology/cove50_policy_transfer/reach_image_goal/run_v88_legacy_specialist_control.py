"""Reopen an earlier specialist success to check the V88 boundary change."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v87_native_case as prior,
    run_v88_native_case as candidate,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v88"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report_path = BASE / "v41/fresh-reserved/seed-10831.json"
    old_plan_path = BASE / "v87/prior75-retention-plan.json"
    report = json.loads(report_path.read_text())
    old_plan = json.loads(old_plan_path.read_text())
    case = {
        "id": "legacy-specialist-10831", "seed": report["seed"],
        "route_variant": "center", "baseline_contact_s": 3.6,
        "rail_v": report["rail_v"], "delay_ms": report["delay_ms"],
        "toy_offset_x_m": report["toy_offset_b_m"][0],
        "toy_offset_y_m": report["toy_offset_b_m"][1],
        "pitch_offset_rad": report["pitch_offset_rad"],
        "roll_clearance_rad": report["roll_clearance_rad"],
        "gaze_head_yaw_rad": report["gaze_head_yaw_rad"],
    }
    hashes = old_plan["source_sha256"]
    if hashes["pointer"] != sha(ROOT / "robot/ape-current-design.json"):
        raise ValueError("Mechanical design pointer changed")
    plan = {
        "schema": "robot-learning-lab.cove50-v88-legacy-specialist-control.v1",
        "role": "reopened_v41_fresh_seed_for_development_not_new_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "v41_report_sha256": sha(report_path),
        "v87_prior75_plan_sha256": sha(old_plan_path),
        "v87_evaluator_sha256": sha(prior.EVALUATOR),
        "v88_evaluator_sha256": sha(candidate.EVALUATOR),
        "v87_runner_sha256": sha(Path(prior.__file__)),
        "v88_runner_sha256": sha(Path(candidate.__file__)),
        "runner_sha256": sha(Path(__file__)),
        "case": case,
        "gate": "Compare native V87 and V88 on the same previously successful specialist start. "
                "If V87 passes, V88 must also pass with at least 1.2 s contact; route "
                "and camera features must be recorded. This is opened development evidence.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / "legacy-specialist-control-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Frozen legacy control/source identity changed")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    for label, runner in (("v87", prior), ("v88", candidate)):
        run_plan = {"source_sha256": {**hashes, "evaluator": sha(runner.EVALUATOR)}}
        dest = OUT / f"native-legacy-specialist-{label}"
        dest.mkdir(parents=True, exist_ok=True)
        row = runner.run_case(case, run_plan, dest)
        rows.append({"variant": label, **row})
        print(json.dumps({key: rows[-1][key] for key in
                          ("variant", "contact_s", "selection_mode", "pass", "faults")}),
              flush=True)
    matrix = {
        "schema": "robot-learning-lab.cove50-v88-legacy-specialist-control.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(plan_path),
        "complete": True,
        "preservation_gate_met": (not rows[0]["pass"] or
                                  rows[1]["pass"] and rows[1]["contact_s"] >= 1.2),
        "cases": rows,
    }
    (OUT / "legacy-specialist-control-matrix.json").write_text(
        json.dumps(matrix, indent=2) + "\n")


if __name__ == "__main__":
    main()
