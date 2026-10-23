"""Source-check saved online selector visits and explain entry-feature drift."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v120_online_smoke as runner,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v147_new_seed_pairs import (
    verify_trace,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v120"
PLAN = BASE / "online-opened-smoke-plan.json"
MATRIX = BASE / "native-opened-smoke/matrix.json"
OUT = BASE / "online-integration-summary.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = runner.freeze()
    matrix = json.loads(MATRIX.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 4 or
            matrix["planned_count"] != 4 or
            matrix["plan_sha256"] != sha(PLAN) or
            len(matrix["cases"]) != 4):
        raise ValueError("Complete source-matched V120 opened matrix required")
    rows = {row["id"]: row for row in matrix["cases"]}
    if set(rows) != {case["id"] for case in plan["cases"]}:
        raise ValueError("V120 rows differ from frozen plan")
    details = []
    for case in plan["cases"]:
        saved = rows[case["id"]]
        replay = runner.run_case(case, plan)
        if replay != saved:
            raise ValueError(f"Saved gate/decision differs from runner: {case['id']}")
        report_path, trace_path = ROOT / saved["result"], ROOT / saved["trace"]
        report = json.loads(report_path.read_text())
        verify_trace(saved["trace"], report)
        expected = case["entry_observed_features"]
        actual = report["online_selector_features"]
        if expected.keys() != actual.keys():
            raise ValueError(f"Online observed feature keys changed: {case['id']}")
        drift = {}
        for key, first in expected.items():
            second = actual[key]
            if isinstance(first, (int, float)) and not isinstance(first, bool):
                drift[key] = round(float(second) - float(first), 6)
            elif first != second:
                raise ValueError(f"Non-numeric online feature changed: {case['id']} {key}")
        if (saved["entry_feature_exact_match"] != (expected == actual) or
                saved["online_decision"] != case["frozen_expected_decision"] or
                not saved["gate_parity"]):
            raise ValueError(f"Online entry, decision or gate record changed: {case['id']}")
        details.append({
            "seed": case["seed"], "entry_feature_exact_match": expected == actual,
            "feature_drift": {key: value for key, value in drift.items() if value},
            "decision_matches_frozen": True,
            "gate_matches_paired_reference": True,
            "full_gate_pass": saved["pass"], "failure_reasons": saved["faults"],
            "contact_s": saved["contact_s"],
            "report": saved["result"], "report_sha256": sha(report_path),
            "trace": saved["trace"], "trace_sha256": sha(trace_path),
        })
    numeric = sorted({key for row in details for key in row["feature_drift"]})
    summary = {
        "schema": "robot-learning-lab.cove50-v120-online-integration-summary.v1",
        "role": "source_checked_opened_online_integration_not_independent_score",
        "body_role": plan["body_role"],
        "source_sha256": {"plan": sha(PLAN), "matrix": sha(MATRIX),
                          "runner": sha(Path(runner.__file__)),
                          "summary_source": sha(Path(__file__))},
        "source_checked_visit_count": 4,
        "exact_entry_feature_match_count": sum(
            row["entry_feature_exact_match"] for row in details),
        "frozen_decision_match_count": 4,
        "paired_gate_match_count": 4,
        "max_abs_feature_drift_by_key": {
            key: max(abs(row["feature_drift"].get(key, 0)) for row in details)
            for key in numeric},
        "qualification": ("Opened integration parity on the rejected historical "
                          "Direct Frame. Small rounded observed-entry drift did "
                          "not alter a decision or gate. One known contact miss "
                          "remains; no new-body qualification follows."),
        "cases": details,
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    print("V120 source-checked 4/4 opened online visits; "
          f"exact entries {summary['exact_entry_feature_match_count']}/4; "
          f"sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
