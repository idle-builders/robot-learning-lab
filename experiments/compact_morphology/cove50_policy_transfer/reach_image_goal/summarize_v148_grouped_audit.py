"""Project the frozen grouped training audit into a source-checked inspector table."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
METHOD = BASE / "v148/signed-rule-method-freeze.json"
DATASET = BASE / "v148/signed-observed-training.json"
AUDIT = BASE / "v148/signed-rule-grouped-audit.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
OUT = BASE / "v148/grouped-audit-inspector-matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    method = json.loads(METHOD.read_text())
    dataset = json.loads(DATASET.read_text())
    audit = json.loads(AUDIT.read_text())
    reserve = json.loads(RESERVE.read_text())
    if (method["source_sha256"]["v147_score_reserve"] != sha(RESERVE)
            or reserve["status"] != "sealed_no_contact_visits"
            or dataset["source_sha256"]["v147_score_reserve"] != sha(RESERVE)
            or audit["source_sha256"]["dataset"] != sha(DATASET)
            or audit["source_sha256"]["reserve"] != sha(RESERVE)
            or dataset["pair_count"] != len(dataset["rows"])
            or dataset["pair_count"] != 182
            or dataset["distinct_seed_count"] != 142
            or audit["crossfit_gate_met"] is not False):
        raise ValueError("V148 source, reserve or failed audit changed")

    held = audit["crossfit"]
    full = audit["full_fit_training_summary"]
    if (held["selected_negative_count"] != 73 or
            held["selected_positive_count"] != 0 or
            held["rescued_v147_seed_count"] != 0 or
            held["opened_score_misses_resolved_audit_only"] or
            held["margin_losses"] != 1 or
            held["selected_harm_or_unknown_seeds_audit_only"] != [19057] or
            held["full_gate_regressions"] != 0 or
            full["selected_negative_count"] != 80 or
            full["selected_positive_count"] != 0 or
            full["rescued_v147_seed_count"] != 1 or
            full["opened_score_misses_resolved_audit_only"]):
        raise ValueError("V148 failure interpretation differs from saved audit")
    cases = [
        {
            "id": "whole_seed_held_audit",
            "evaluation": "Whole seed held training audit",
            "gate_met": audit["crossfit_gate_met"],
            "selected_negative": held["selected_negative_count"],
            "selected_positive": held["selected_positive_count"],
            "rescued_seed_groups": held["rescued_seed_count"],
            "held_v147_rescues": held["rescued_v147_seed_count"],
            "opened_misses_repaired": len(held["opened_score_misses_resolved_audit_only"]),
            "full_gate_regressions": held["full_gate_regressions"],
            "margin_losses": held["margin_losses"],
            "failure_reason": (
                "No held V147 positive or negative rescue; neither exact opened "
                "V139 miss repaired; one selected margin loss on seed 19057. "
                "The signed recovery gate fails."
            ),
        },
        {
            "id": "full_fit_training_diagnostic",
            "evaluation": "Full fit on all opened training pairs",
            "gate_met": False,
            "selected_negative": full["selected_negative_count"],
            "selected_positive": full["selected_positive_count"],
            "rescued_seed_groups": full["rescued_seed_count"],
            "held_v147_rescues": full["rescued_v147_seed_count"],
            "opened_misses_repaired": len(full["opened_score_misses_resolved_audit_only"]),
            "full_gate_regressions": full["full_gate_regressions"],
            "margin_losses": full["margin_losses"],
            "failure_reason": (
                "In-sample diagnostic only: no positive action, neither exact "
                "opened V139 miss repaired, and no held-out validation."
            ),
        },
    ]
    out = {
        "schema": "robot-learning-lab.cove50-v148-grouped-audit-inspector.v1",
        "role": "failed_opened_training_audit_not_independent_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "method_sha256": sha(METHOD),
        "dataset_sha256": sha(DATASET),
        "source_audit_sha256": sha(AUDIT),
        "reserve_sha256": sha(RESERVE),
        "pair_count": dataset["pair_count"],
        "seed_group_count": dataset["distinct_seed_count"],
        "complete": True,
        "case_count": len(cases),
        "cases": cases,
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"V148 grouped audit failed; projected {len(cases)} evidence rows")


if __name__ == "__main__":
    main()
