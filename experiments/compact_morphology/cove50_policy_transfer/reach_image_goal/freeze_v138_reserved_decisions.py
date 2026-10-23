"""Freeze V135 choices on reserved RGB entries before any contact visit."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    v118_observable_selector as observations,
    v135_planar_selector as selector,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
RESERVE = BASE / "v128/disjoint-contact-score-plan.json"
MODEL = BASE / "v135/planar-selector.json"
ENTRY_PLAN = BASE / "v137/reserved-entry-only-plan.json"
ENTRY_MATRIX = BASE / "v137/native-entry-only/matrix.json"
RETENTION = BASE / "v136/native-extra-shifts/matrix.json"
OUT = BASE / "v138/frozen-reserved-decisions.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    reserve = json.loads(RESERVE.read_text())
    model = selector.load(MODEL)
    entry = json.loads(ENTRY_MATRIX.read_text())
    retention = json.loads(RETENTION.read_text())
    if (reserve["status"] != "sealed_no_contact_visits" or
            model["source_sha256"]["sealed_reserve"] != sha(RESERVE) or
            model["source_sha256"]["inference"] != sha(Path(selector.__file__)) or
            not retention["complete"] or retention["candidate_pass_count"] != 24 or
            retention["candidate_margin_count"] != 24 or
            not entry["complete"] or entry["case_count"] != 24 or
            entry["plan_sha256"] != sha(ENTRY_PLAN) or
            entry["contains_contact_outcomes"] or
            {row["seed"] for row in entry["cases"]} !=
            {case["seed"] for case in reserve["cases"]}):
        raise ValueError("Complete entry-only reserve and older retention required")
    entries = {row["id"]: row for row in entry["cases"]}
    cases = []
    selected = Counter()
    for source_case in reserve["cases"]:
        row = entries[source_case["id"]]
        report_path = ROOT / row["result"]
        if (sha(report_path) != row["result_sha256"] or
                sha(report_path.with_suffix(".npz")) != row["trace_sha256"]):
            raise ValueError(f"Reserved entry source changed: {row['id']}")
        report = json.loads(report_path.read_text())
        if (report["seed"] != source_case["seed"] or
                not report["preflight_only"] or
                report["reach_duration_s"] != .02 or
                report["body_sha256"] != reserve["source_sha256"]["body"] or
                report["selected_pointer_sha256"] != reserve["source_sha256"]["pointer"]):
            raise ValueError(f"Reserved entry is not the frozen preflight: {row['id']}")
        features = (observations.observed_features(report)
                    if row["entry_available"] else None)
        if features != row["observed_features"]:
            raise ValueError(f"Reserved observed features changed: {row['id']}")
        decision = (selector.predict(model, features) if features is not None
                    else {"shift_body_y_m": 0.0,
                          "abstention_reasons": ["no_used_controlled_RGB_entry"]})
        if decision != row["frozen_model_decision_preview_no_contact"]:
            raise ValueError(f"Reserved candidate decision changed: {row['id']}")
        if decision["shift_body_y_m"] < 0:
            selected[(source_case["cohort"], source_case["route_variant"])] += 1
        cases.append({**source_case,
                      "entry_available": row["entry_available"],
                      "entry_report": row["result"],
                      "entry_report_sha256": row["result_sha256"],
                      "entry_trace_sha256": row["trace_sha256"],
                      "entry_observed_features": features,
                      "frozen_decision": decision})
    near = {route: selected[("heldout_near_failure", route)]
            for route in ("roll-low", "roll-high")}
    plan = {
        "schema": "robot-learning-lab.cove50-v138-reserved-frozen-decisions.v1",
        "role": "independent_contact_unseen_frozen_candidate_choices",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {"reserve": sha(RESERVE), "candidate": sha(MODEL),
                          "inference": sha(Path(selector.__file__)),
                          "entry_plan": sha(ENTRY_PLAN),
                          "entry_matrix": sha(ENTRY_MATRIX),
                          "older_retention": sha(RETENTION),
                          "freezer": sha(Path(__file__))},
        "case_count": 24,
        "entry_available_count": sum(case["entry_available"] for case in cases),
        "selected_shift_count": sum(case["frozen_decision"]["shift_body_y_m"] < 0
                                    for case in cases),
        "selected_near_failure_by_route": near,
        "correction_exposure_gate_met": all(near.values()),
        "status": "frozen_before_any_full_native_contact_visit",
        "method": ("All 24 choices are made by the frozen V135 observed-RGB "
                   "rule from source-checked first-entry camera reports. "
                   "No V128 full-native contact result exists at freeze time. "
                   "The entry-only process is never a contact score. "
                   "Roll-low and roll-high near-failure correction exposure "
                   "is required before starting the full score."),
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2) + "\n"
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Previously frozen V138 decisions changed")
    OUT.write_text(encoded)
    print(f"V138 frozen choices sha256={sha(OUT)} selected "
          f"{plan['selected_shift_count']}/24 near {near} "
          f"exposure={plan['correction_exposure_gate_met']}")


if __name__ == "__main__":
    main()
