"""Describe source-checked near-matched V149 entries across distinct seeds.

This posthoc entry-only diagnostic limits what a one-seed camera map can show.
It is not a causal seed experiment, contact result, or selector score.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    v118_observable_selector as observations,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v147_new_seed_pairs import (
    verify_trace,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
PLAN = BASE / "v149/new-seed-route-exposure-plan.json"
MATRIX = BASE / "v149/entry-exposure-inspector-matrix.json"
NATIVE_MATRIX = BASE / "v149/native-entry-only/matrix.json"
OUT = BASE / "v155/near-matched-independent-seed-entry-contrast.json"
MAX_TOY_XY_GAP_MM = .15
EXACT_FIELDS = ("cohort", "rail_v", "delay_ms", "gaze_head_yaw_rad",
                "pitch_offset_rad", "roll_clearance_rad")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    native = json.loads(NATIVE_MATRIX.read_text())
    if (not native["complete"] or native["case_count"] != 96 or
            matrix["source_matrix_sha256"] != sha(NATIVE_MATRIX) or
            matrix["case_count"] != 96 or
            matrix["case_count"] != len(matrix["cases"]) or
            matrix["plan_sha256"] != sha(PLAN) or
            len(plan["cases"]) != 96 or
            native["contains_contact_outcomes"]):
        raise ValueError("Complete source-bound V149 entry screen required")
    entries = {row["seed"]: row for row in matrix["cases"]}
    if len(entries) != 96:
        raise ValueError("V149 seeds are not unique")
    for case in plan["cases"]:
        row = entries[case["seed"]]
        report_path = ROOT / row["result"]
        trace_path = report_path.with_suffix(".npz")
        if (row["id"] != case["id"] or not row["source_checked"] or
                not row["entry_available"] or
                sha(report_path) != row["result_sha256"] or
                sha(trace_path) != row["trace_sha256"]):
            raise ValueError(f"V149 entry source changed: {case['id']}")
        report = json.loads(report_path.read_text())
        verify_trace(str(trace_path.relative_to(ROOT)), report)
        if (report["seed"] != case["seed"] or
                observations.observed_features(report) != row["observed_features"]):
            raise ValueError(f"V149 controlled RGB differs: {case['id']}")
    pairs = []
    for left, right in itertools.combinations(plan["cases"], 2):
        if any(left[key] != right[key] for key in EXACT_FIELDS):
            continue
        toy_gap = 1000 * math.hypot(
            left["toy_offset_x_m"] - right["toy_offset_x_m"],
            left["toy_offset_y_m"] - right["toy_offset_y_m"])
        if toy_gap > MAX_TOY_XY_GAP_MM:
            continue
        a, b = entries[left["seed"]], entries[right["seed"]]
        ao, bo = a["observed_features"], b["observed_features"]
        camera_gap = math.hypot(ao["camera_x_mm"] - bo["camera_x_mm"],
                                ao["camera_y_mm"] - bo["camera_y_mm"])
        pairs.append({
            "seeds": [left["seed"], right["seed"]],
            "toy_xy_gap_mm": round(toy_gap, 6),
            "camera_xy_gap_mm": round(camera_gap, 6),
            "routes": [a["selection_mode"], b["selection_mode"]],
            "routes_differ": a["selection_mode"] != b["selection_mode"],
            "camera_xy_mm": [[ao["camera_x_mm"], ao["camera_y_mm"]],
                             [bo["camera_x_mm"], bo["camera_y_mm"]]],
            "reports": [a["result"], b["result"]],
        })
    pairs.sort(key=lambda row: (-row["camera_xy_gap_mm"], row["seeds"]))
    result = {
        "schema": "robot-learning-lab.cove50-v155-cross-seed-entry-contrast.v1",
        "role": "posthoc_descriptive_entry_only_not_causal_contact_or_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {"v149_plan": sha(PLAN), "v149_matrix": sha(MATRIX),
                          "v149_native_matrix": sha(NATIVE_MATRIX),
                          "audit_source": sha(Path(__file__))},
        "source_checked_entry_count": len(entries),
        "exact_match_fields": list(EXACT_FIELDS),
        "maximum_toy_xy_gap_mm": MAX_TOY_XY_GAP_MM,
        "near_matched_pair_count": len(pairs),
        "different_route_pair_count": sum(row["routes_differ"] for row in pairs),
        "median_camera_xy_gap_mm": round(statistics.median(
            row["camera_xy_gap_mm"] for row in pairs), 6),
        "pairs": pairs,
        "interpretation": (
            "Distinct simulator seeds with close but unequal toy XY settings "
            "can have different controlled camera entries and automatic "
            "routes. This descriptive contrast does not isolate seed from toy "
            "effects. A one-seed entry map cannot establish exposure on new "
            "seeds; screen actual RGB on independent seed families before "
            "selecting source-matched contact visits."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "near_matched_pair_count", "different_route_pair_count",
        "median_camera_xy_gap_mm")}))


if __name__ == "__main__":
    main()
