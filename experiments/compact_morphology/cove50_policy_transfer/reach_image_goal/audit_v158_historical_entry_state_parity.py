"""Calibrate pre-reach state parity from source-valid historical CUDA pairs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v158_new_seed_development_contact as runner,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
PLAN = BASE / "v152/paired-route-training-plan.json"
MATRIX = BASE / "v152/native-paired-routes/matrix.json"
SUMMARY = BASE / "v152/partial-source-checked-paired-route-summary.json"
OUT = BASE / "v158/historical-native-entry-state-parity.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    matrix = json.loads(MATRIX.read_text())
    summary = json.loads(SUMMARY.read_text())
    if (matrix["case_count"] != 47 or matrix["complete"] or
            matrix["plan_sha256"] != sha(PLAN) or
            summary["source_checked_pair_count"] != 47 or
            summary["source_checked_native_source_count"] != 94 or
            summary["source_sha256"]["plan"] != sha(PLAN) or
            summary["source_sha256"]["matrix"] != sha(MATRIX) or
            summary["source_sha256"]["checker"] != sha(
                Path(__file__).with_name(
                    "summarize_v152_paired_route_training.py")) or
            summary["numeric_rgb_entry_failure_count"] != 0):
        raise ValueError("V152 terminal source-valid CUDA pairs required")
    for item in summary["source_files"]:
        for key in ("report", "trace"):
            source = ROOT / item[key]
            if not source.is_file() or sha(source) != item[f"{key}_sha256"]:
                raise ValueError(f"Historical native source changed: {source}")
    rows = []
    for pair in matrix["cases"]:
        automatic = json.loads((ROOT / pair["baseline_report"]).read_text())
        prescribed = json.loads((ROOT / pair["prescribed_report"]).read_text())
        gap, root_gap, faults = runner.entry_state_gap(
            automatic, prescribed, pair["baseline_trace"],
            pair["prescribed_trace"])
        rows.append({"id": pair["id"], "seed": pair["seed"],
                     "automatic_route": pair["automatic_route"],
                     "prescribed_route": pair["prescribed_route"],
                     "reach_entry_qpos_max_abs_gap": gap,
                     "reach_entry_root_gap_m": root_gap,
                     "reach_entry_gap_micro_units": gap * 1e6,
                     "reach_entry_root_gap_um": root_gap * 1e6,
                     "exceeds_v158_tolerance": bool(faults),
                     "baseline_report": pair["baseline_report"],
                     "prescribed_report": pair["prescribed_report"],
                     "baseline_trace": pair["baseline_trace"],
                     "prescribed_trace": pair["prescribed_trace"]})
    if len(rows) != 47 or len({row["seed"] for row in rows}) != 47:
        raise ValueError("Historical CUDA seed pairs changed")
    result = {
        "schema": "robot-learning-lab.cove50-v158-historical-entry-state-parity.v1",
        "role": "source_checked_historical_pair_parity_tolerance_audit_not_contact_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {"plan": sha(PLAN), "matrix": sha(MATRIX),
                          "summary": sha(SUMMARY),
                          "runner": sha(Path(runner.__file__)),
                          "audit": sha(Path(__file__))},
        "source_valid_cuda_pair_count": len(rows),
        "mixed_unit_warning": "qpos maximum compares metres and radians; it is a paired-state identity screen, not a physical error norm.",
        "reach_entry_qpos_max_abs_tolerance": runner.ENTRY_QPOS_MAX_ABS_TOLERANCE,
        "historical_max_qpos_abs_gap": max(
            row["reach_entry_qpos_max_abs_gap"] for row in rows),
        "historical_max_root_gap_m": max(
            row["reach_entry_root_gap_m"] for row in rows),
        "historical_tolerance_failure_count": sum(
            row["exceeds_v158_tolerance"] for row in rows),
        "contains_v158_contact_results": False,
        "cases": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, OUT)
    print(json.dumps({"source_valid_cuda_pair_count": len(rows),
                      "historical_max_qpos_abs_gap":
                      result["historical_max_qpos_abs_gap"],
                      "tolerance": runner.ENTRY_QPOS_MAX_ABS_TOLERANCE}))


if __name__ == "__main__":
    main()
