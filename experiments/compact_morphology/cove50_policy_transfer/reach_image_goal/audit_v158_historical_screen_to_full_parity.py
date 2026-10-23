"""Calibrate camera-screen to full-contact entry parity from saved CUDA traces."""

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
PLAN = BASE / "v150/paired-contact-training-plan.json"
MATRIX = BASE / "v150/native-paired-contact/matrix.json"
SUMMARY = BASE / "v150/source-checked-paired-summary.json"
OUT = BASE / "v158/historical-screen-to-full-entry-state-parity.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    matrix = json.loads(MATRIX.read_text())
    summary = json.loads(SUMMARY.read_text())
    if (matrix["plan_sha256"] != sha(PLAN) or not matrix["complete"] or
            matrix["case_count"] != 48 or not summary["complete"] or
            summary["source_checked_seed_count"] != 48 or
            summary["source_checked_native_visit_count"] != 144 or
            summary["plan_sha256"] != sha(PLAN) or
            summary["matrix_sha256"] != sha(MATRIX) or
            summary["screen_entry_parity_failure_count"] != 0 or
            len(summary["source_files"]) != 144):
        raise ValueError("Complete source-checked V150 CUDA training required")
    for item in summary["source_files"]:
        for key in ("report", "trace"):
            path = ROOT / item[key]
            if not path.is_file() or sha(path) != item[f"{key}_sha256"]:
                raise ValueError(f"V150 native source changed: {path}")
    rows = []
    for pair in matrix["cases"]:
        screen_path = ROOT / pair["entry_report"]
        screen_trace = screen_path.with_suffix(".npz")
        if (sha(screen_path) != pair["entry_report_sha256"] or
                sha(screen_trace) != pair["entry_trace_sha256"] or
                pair["screen_to_full_entry_faults"]):
            raise ValueError(f"Historical screen source or RGB parity changed: {pair['id']}")
        screen = json.loads(screen_path.read_text())
        automatic = json.loads((ROOT / pair["baseline_result"]).read_text())
        gap, root_gap, faults = runner.entry_state_gap(
            screen, automatic, str(screen_trace.relative_to(ROOT)),
            pair["baseline_trace"])
        rows.append({
            "id": pair["id"], "seed": pair["seed"],
            "screen_route": pair["entry_selection_mode"],
            "automatic_route": pair["selection_mode"],
            "reach_entry_qpos_max_abs_gap": gap,
            "reach_entry_root_gap_m": root_gap,
            "reach_entry_gap_micro_units": gap * 1e6,
            "reach_entry_root_gap_um": root_gap * 1e6,
            "exceeds_v158_tolerance": bool(faults),
            "screen_report": pair["entry_report"],
            "screen_trace": str(screen_trace.relative_to(ROOT)),
            "automatic_report": pair["baseline_result"],
            "automatic_trace": pair["baseline_trace"],
        })
    if len(rows) != 48 or len({row["seed"] for row in rows}) != 48:
        raise ValueError("Historical screen/full CUDA seed pairs changed")
    result = {
        "schema": "robot-learning-lab.cove50-v158-screen-full-entry-state-parity.v1",
        "role": "source_checked_historical_screen_full_parity_not_new_contact_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {
            "plan": sha(PLAN), "matrix": sha(MATRIX), "summary": sha(SUMMARY),
            "runner": sha(Path(runner.__file__)), "audit": sha(Path(__file__)),
        },
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
    print(json.dumps({
        "source_valid_cuda_pair_count": len(rows),
        "historical_max_qpos_abs_gap": result["historical_max_qpos_abs_gap"],
        "tolerance": runner.ENTRY_QPOS_MAX_ABS_TOLERANCE,
    }))


if __name__ == "__main__":
    main()
