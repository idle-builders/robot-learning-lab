"""Audit unused V149 camera entries before planning more route contact data.

No V152 prescribed contact outcome or sealed score entry is read here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
ENTRY_PLAN = BASE / "v149/new-seed-route-exposure-plan.json"
ENTRY_MATRIX = BASE / "v149/entry-exposure-inspector-matrix.json"
ENTRY_SUMMARY = BASE / "v149/entry-exposure-summary.json"
V150_PLAN = BASE / "v150/paired-contact-training-plan.json"
V150_MATRIX = BASE / "v150/native-paired-contact/matrix.json"
V150_SUMMARY = BASE / "v150/source-checked-paired-summary.json"
V152_PLAN = BASE / "v152/paired-route-training-plan.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
POINTER = ROOT / "robot/ape-current-design.json"
OUT = BASE / "v154/remaining-entry-route-coverage.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    entry_plan = json.loads(ENTRY_PLAN.read_text())
    entry_matrix = json.loads(ENTRY_MATRIX.read_text())
    entry_summary = json.loads(ENTRY_SUMMARY.read_text())
    v150_plan = json.loads(V150_PLAN.read_text())
    v150_matrix = json.loads(V150_MATRIX.read_text())
    v150_summary = json.loads(V150_SUMMARY.read_text())
    v152_plan = json.loads(V152_PLAN.read_text())
    reserve = json.loads(RESERVE.read_text())
    pointer = json.loads(POINTER.read_text())
    if (entry_plan["case_count"] != 96 or
            entry_matrix["plan_sha256"] != sha(ENTRY_PLAN) or
            entry_matrix["case_count"] != 96 or not entry_summary["complete"] or
            entry_matrix["source_matrix_sha256"] != entry_summary["matrix_sha256"] or
            entry_summary["inspector_matrix_sha256"] != sha(ENTRY_MATRIX) or
            entry_summary["source_checked_entry_count"] != 96 or
            v150_plan["case_count"] != 48 or
            v150_plan["source_sha256"]["v149_entry_matrix"] != sha(ENTRY_MATRIX) or
            v150_matrix["plan_sha256"] != sha(V150_PLAN) or
            v150_matrix["case_count"] != 48 or not v150_matrix["complete"] or
            v150_summary["matrix_sha256"] != sha(V150_MATRIX) or
            not v150_summary["complete"] or
            v152_plan["case_count"] != 48 or
            v152_plan["source_sha256"]["v150_plan"] != sha(V150_PLAN) or
            reserve["status"] != "sealed_no_contact_visits" or
            entry_plan["source_sha256"]["v147_unopened_score_reserve"] != sha(RESERVE) or
            pointer["selected_design_id"] is not None):
        raise ValueError("Complete V149/V150 source and sealed reserve required")
    entry_cases = {case["seed"]: case for case in entry_plan["cases"]}
    entry_rows = {row["seed"]: row for row in entry_matrix["cases"]}
    selected = {case["seed"] for case in v150_plan["cases"]}
    if (len(entry_cases) != 96 or len(entry_rows) != 96 or len(selected) != 48 or
            {case["seed"] for case in v152_plan["cases"]} != selected):
        raise ValueError("Entry pool and V150/V152 membership differ")
    remaining = []
    counts = defaultdict(Counter)
    for seed in sorted(entry_cases.keys() - selected):
        case = entry_cases[seed]
        row = entry_rows[seed]
        if (not row["source_checked"] or not row["entry_available"] or
                row["observed_features"] is None or
                row["selection_mode"] != row["observed_features"]["selection_mode"] or
                row["cohort"] != case["cohort"]):
            raise ValueError(f"Unusable source-checked entry: {seed}")
        counts[case["cohort"]][row["selection_mode"]] += 1
        remaining.append({
            "id": f"unused-entry-{seed}", "seed_audit_only": seed,
            "cohort_audit_only": case["cohort"],
            "automatic_route": row["selection_mode"],
            "rgb_goal_source": row["observed_features"]["route_goal_source"],
            "entry_error_x_mm": row["observed_features"]["entry_error_x_mm"],
            "entry_error_y_mm": row["observed_features"]["entry_error_y_mm"],
            "orange_pixels": row["observed_features"]["orange_pixels"],
            "entry_report": row["result"],
            "entry_report_sha256": row["result_sha256"],
            "entry_trace_sha256": row["trace_sha256"],
        })
    cohorts = {key: dict(sorted(value.items())) for key, value in sorted(counts.items())}
    if (len(remaining) != 48 or
            {key: sum(value.values()) for key, value in cohorts.items()} != {
                "opened_pitch_low_neighborhood": 24,
                "opened_roll_high_neighborhood": 24,
            } or
            any("v16_live" in value for value in cohorts.values())):
        raise ValueError("V149 unused pool coverage changed")
    result = {
        "schema": "robot-learning-lab.cove50-v154-unused-entry-route-coverage.v1",
        "role": "source_checked_entry_coverage_not_contact_or_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {
            "entry_plan": sha(ENTRY_PLAN), "entry_matrix": sha(ENTRY_MATRIX),
            "entry_summary": sha(ENTRY_SUMMARY), "v150_plan": sha(V150_PLAN),
            "v150_matrix": sha(V150_MATRIX), "v150_summary": sha(V150_SUMMARY),
            "v152_plan": sha(V152_PLAN), "reserve": sha(RESERVE),
            "pointer": sha(POINTER), "audit_source": sha(Path(__file__)),
        },
        "v152_prescribed_outcomes_used": False,
        "case_count": len(remaining), "planned_count": len(remaining),
        "complete": True, "source_checked_entry_count": len(remaining),
        "contains_contact_outcomes": False,
        "cohort_route_counts": cohorts,
        "live_route_entry_count": sum(value.get("v16_live", 0) for value in counts.values()),
        "cases": remaining,
        "interpretation": (
            "All 48 V149 camera-only starts left out of V150 and V152 have "
            "source-checked RGB entry. Their route mix contains no v16_live "
            "entry, so replaying this pool alone cannot add a held example "
            "for the rare live-route baseline failure 24591. It may add "
            "locked-route development contrast, but has no contact result "
            "and cannot repair the frozen V153 gate. Collect new independently "
            "varied live-route entries before a new route selector fit."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, OUT)
    print(json.dumps({"case_count": result["case_count"],
                      "cohort_route_counts": cohorts,
                      "live_route_entry_count": result["live_route_entry_count"]}))


if __name__ == "__main__":
    main()
