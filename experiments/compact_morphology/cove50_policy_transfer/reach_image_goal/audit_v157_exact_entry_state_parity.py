"""Check physical start identity for the frozen V157 camera repeats.

This is a posthoc diagnostic of saved CUDA traces, not another camera visit,
contact outcome, or change to the frozen V157 repeatability check.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v158_new_seed_development_contact as parity,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V156_PLAN = BASE / "v156/independent-seed-entry-family-plan.json"
V156_MATRIX = BASE / "v156/native-entry-only/matrix.json"
V156_SUMMARY = BASE / "v156/source-checked-entry-family-summary.json"
V157_PLAN = BASE / "v157/exact-rgb-repeatability-plan.json"
V157_MATRIX = BASE / "v157/native-entry-only/matrix.json"
V157_SUMMARY = BASE / "v157/source-checked-rgb-repeatability-summary.json"
V157_PROJECTION = BASE / "v157/source-checked-rgb-repeatability-inspector-matrix.json"
OUT = BASE / "v157/exact-repeat-physical-entry-parity.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_files(summary: dict) -> None:
    for source in summary["source_files"]:
        for key in ("report", "trace", "image"):
            if key in source:
                path = ROOT / source[key]
                if not path.is_file() or sha(path) != source[f"{key}_sha256"]:
                    raise ValueError(f"Saved CUDA source changed: {path}")


def main() -> None:
    original_matrix = json.loads(V156_MATRIX.read_text())
    original_summary = json.loads(V156_SUMMARY.read_text())
    repeat_plan = json.loads(V157_PLAN.read_text())
    repeat_matrix = json.loads(V157_MATRIX.read_text())
    repeat_summary = json.loads(V157_SUMMARY.read_text())
    projection = json.loads(V157_PROJECTION.read_text())
    if (original_matrix["plan_sha256"] != sha(V156_PLAN) or
            original_matrix["case_count"] != 168 or
            not original_matrix["complete"] or
            original_summary["source_sha256"]["matrix"] != sha(V156_MATRIX) or
            original_summary["source_checked_entry_count"] != 168 or
            not original_summary["complete"] or
            repeat_plan["source_sha256"]["v156_plan"] != sha(V156_PLAN) or
            repeat_matrix["plan_sha256"] != sha(V157_PLAN) or
            repeat_matrix["case_count"] != 24 or
            not repeat_matrix["complete"] or
            repeat_summary["source_sha256"]["matrix"] != sha(V157_MATRIX) or
            repeat_summary["source_sha256"]["projection"] != sha(V157_PROJECTION) or
            repeat_summary["source_checked_repeat_count"] != 24 or
            not repeat_summary["complete"] or
            projection["case_count"] != 24 or
            not projection["complete"] or
            original_matrix["contains_contact_outcomes"] or
            repeat_matrix["contains_contact_outcomes"]):
        raise ValueError("Complete source-checked V156 and V157 camera matrices required")
    check_files(original_summary)
    check_files(repeat_summary)
    original = {row["id"]: row for row in original_matrix["cases"]}
    repeated = {row["id"]: row for row in repeat_matrix["cases"]}
    observed = {row["id"]: row for row in projection["cases"]}
    rows = []
    for case in repeat_plan["cases"]:
        first = original[case["source_v156_entry_id"]]
        second = repeated[case["id"]]
        rgb = observed[case["id"]]
        if (case["seed"] != first["seed"] or
                case["seed"] != second["seed"] or
                rgb["source_v156_entry_id"] != first["id"]):
            raise ValueError(f"Exact repeat membership changed: {case['id']}")
        first_report = json.loads((ROOT / first["result"]).read_text())
        second_report = json.loads((ROOT / second["result"]).read_text())
        first_trace = str((ROOT / first["result"]).with_suffix(".npz").relative_to(ROOT))
        second_trace = str((ROOT / second["result"]).with_suffix(".npz").relative_to(ROOT))
        gap, root_gap, faults = parity.entry_state_gap(
            first_report, second_report, first_trace, second_trace)
        rows.append({
            "id": case["id"], "seed": case["seed"],
            "stratum": case["stratum"],
            "source_v156_entry_id": first["id"],
            "reach_entry_qpos_max_abs_gap": gap,
            "reach_entry_root_gap_m": root_gap,
            "reach_entry_gap_micro_units": gap * 1e6,
            "reach_entry_root_gap_um": root_gap * 1e6,
            "exceeds_v158_identity_tolerance": bool(faults),
            "rgb_parity_faults": rgb["parity_faults"],
            "rgb_goal_shift_mm": rgb["numeric_goal_shift_mm"],
            "orange_mask_differing_pixels": rgb["orange_mask_differing_pixels"],
            "route_changed": rgb["route_changed"],
            "first_report": first["result"], "first_trace": first_trace,
            "repeat_report": second["result"], "repeat_trace": second_trace,
        })
    if len(rows) != 24 or len({row["seed"] for row in rows}) != 24:
        raise ValueError("Expected one source-checked exact repeat per seed family")
    result = {
        "schema": "robot-learning-lab.cove50-v157-exact-repeat-physical-entry-parity.v1",
        "role": "posthoc_cuda_physical_entry_diagnostic_not_contact_or_policy_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {
            "v156_plan": sha(V156_PLAN), "v156_matrix": sha(V156_MATRIX),
            "v156_summary": sha(V156_SUMMARY), "v157_plan": sha(V157_PLAN),
            "v157_matrix": sha(V157_MATRIX), "v157_summary": sha(V157_SUMMARY),
            "v157_projection": sha(V157_PROJECTION),
            "parity_runner": sha(Path(parity.__file__)),
            "audit": sha(Path(__file__)),
        },
        "source_checked_exact_repeat_count": len(rows),
        "reach_entry_qpos_max_abs_tolerance": parity.ENTRY_QPOS_MAX_ABS_TOLERANCE,
        "max_qpos_abs_gap": max(row["reach_entry_qpos_max_abs_gap"] for row in rows),
        "max_root_gap_m": max(row["reach_entry_root_gap_m"] for row in rows),
        "physical_identity_fault_count": sum(
            row["exceeds_v158_identity_tolerance"] for row in rows),
        "rgb_fault_with_physical_identity_fault_count": sum(
            bool(row["rgb_parity_faults"]) and row["exceeds_v158_identity_tolerance"]
            for row in rows),
        "mixed_unit_warning": "Qpos maximum mixes metres and radians; it checks saved state identity, not physical distance.",
        "interpretation": "Exact repeated RGB observations must be read with their saved physical entry gap. This diagnostic does not replace V157's frozen RGB audit or add independent seeds or fingertip contact outcomes.",
        "cases": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, OUT)
    print(json.dumps({key: result[key] for key in (
        "source_checked_exact_repeat_count", "max_qpos_abs_gap",
        "physical_identity_fault_count", "rgb_fault_with_physical_identity_fault_count")}))


if __name__ == "__main__":
    main()
