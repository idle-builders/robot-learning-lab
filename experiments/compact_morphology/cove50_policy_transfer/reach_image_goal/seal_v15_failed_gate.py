"""Seal the stopped v15 opened gate, including its two direct target replays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
import sys
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate

BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V15 = BASE / "v15"
DEST = V15 / "opened-native"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = V15 / "opened-native-plan.json"
    plan = json.loads(plan_path.read_text())
    prior = {}
    for stage in ("development", "fresh-reserved"):
        matrix = json.loads((BASE / f"v14/{stage}/matrix.json").read_text())
        prior.update({row["seed"]: row for row in matrix["cases"]})
    rows = []
    for case in plan["cases"]:
        seed = case["seed"]
        report_path = DEST / f"seed-{seed}.json"
        trace_path = DEST / f"seed-{seed}.npz"
        log_path = DEST / f"seed-{seed}.log"
        if not report_path.exists():
            continue
        if not trace_path.exists() or not log_path.exists():
            raise ValueError(f"Incomplete retained evidence: {seed}")
        report = json.loads(report_path.read_text())
        expected_mode = {"v10_locked": "v15_v10_height_locked",
                         "v14_live": "v15_live", "v14_locked": "v15_locked"}[prior[seed]["selection_mode"]]
        if (report["evaluator_sha256"] != plan["source_sha256"]["evaluator"] or
                report["reach_checkpoint_sha256"] != plan["source_sha256"]["actor"] or
                report["selection_mode"] != expected_mode or report["seed"] != seed):
            raise ValueError(f"Source identity changed for {seed}")
        faults = gate(report)
        entry = next(event for event in report["image_goal_capture_events"] if event["time_s"] == 7.5)
        if not entry["used_for_control"] or report["image_goal_missing_at_s"] is not None:
            faults.append("invalid RGB goal")
        rows.append({
            "seed": seed, "origin": case["origin"], "selection_mode": expected_mode,
            "rail_v": case["rail_v"], "delay_ms": case["delay_ms"],
            "toy_offset_b_m": [case["toy_offset_x_m"], case["toy_offset_y_m"]],
            "pass": not faults,
            "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
            "entry_goal_error_mm_scoring_only": entry["xy_error_truth_mm_scoring_only"],
            "reach_both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
            "reach_rated_leg_fraction": report["phases"]["reach"]["rated_leg_fraction"],
            "faults": faults,
            "result": str(report_path.relative_to(ROOT)),
            "trace": str(trace_path.relative_to(ROOT)),
            "log": str(log_path.relative_to(ROOT)),
        })
    if len(rows) != 13 or sum(row["pass"] for row in rows if row["origin"] == "prior_v14_development") >= 10:
        raise ValueError("Stopped-gate evidence differs from the observed 13/30 failure")
    matrix = {
        "schema": "robot-learning-lab.cove50-v15-opened-native.v1",
        "role": "opened_development_gate_only",
        "plan_sha256": sha(plan_path),
        "seal_source_sha256": sha(Path(__file__)),
        "complete": False,
        "stopped_early": True,
        "stop_reason": "Earlier ten-case development gate failed; 10310 and 10318 were run directly to measure intended corrections. No fresh v15 starts were opened.",
        "planned_case_count": 30,
        "case_count": len(rows), "pass_count": sum(row["pass"] for row in rows),
        "cases": rows,
    }
    (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(f"sealed failed v15 opened gate: {matrix['pass_count']}/{matrix['case_count']} observed of 30 planned")


if __name__ == "__main__":
    main()
