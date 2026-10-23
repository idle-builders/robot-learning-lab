"""Freeze the conservative grouped planar candidate before new contact visits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    audit_v134_planar_rule as audit,
    v135_planar_selector as runtime,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
DATASET = BASE / "v131/failure-augmented-observable-pairs.json"
AUDIT = BASE / "v134/planar-rule-grouped-audit.json"
RESERVE = BASE / "v128/disjoint-contact-score-plan.json"
OUT = BASE / "v135/planar-selector.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = json.loads(DATASET.read_text())
    report = json.loads(AUDIT.read_text())
    reserve = json.loads(RESERVE.read_text())
    if (not data["complete"] or len(data["pairs"]) != 113 or
            len({row["seed_audit_only"] for row in data["pairs"]}) != 89 or
            report["source_sha256"] != {
                "dataset": sha(DATASET), "reserve": sha(RESERVE),
                "audit": sha(Path(audit.__file__))} or
            reserve["status"] != "sealed_no_contact_visits" or
            len(reserve["cases"]) != 24 or
            {row["seed_audit_only"] for row in data["pairs"]} &
            {case["seed"] for case in reserve["cases"]}):
        raise ValueError("Complete grouped training audit and sealed reserve required")
    selection = next(row for row in report["audits"]
                     if row["weighting"] == "equal_seed")
    if (selection["selected_full_gate_regressions"] or
            selection["selected_margin_losses"] or
            selection["selected_unresolved_failures"] or
            selection["selected_rescue_seed_count"] < 3):
        raise ValueError("Grouped development gate failed")
    fitted = audit.fit(data["pairs"], "equal_seed")
    if fitted != selection["full_training_rules_diagnostic_only"]:
        raise ValueError("Full training rule differs from grouped audit")
    model = {
        "schema": runtime.SCHEMA,
        "role": "frozen_development_candidate_not_qualified",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "output": "post_route_body_y_goal_shift_m",
        "possible_outputs_m": [0.0, -.002],
        "selection_timing": "Observed RGB reach entry, after automatic route selection",
        "training_pair_count": 113, "training_distinct_seed_count": 89,
        "group_crossfit_selected_rescue_seed_count":
            selection["selected_rescue_seed_count"],
        "group_crossfit_possible_rescue_seed_count":
            selection["possible_rescue_seed_count"],
        "source_sha256": {"training_dataset": sha(DATASET),
                          "grouped_audit": sha(AUDIT),
                          "sealed_reserve": sha(RESERVE),
                          "fitter": sha(Path(__file__)),
                          "inference": sha(Path(runtime.__file__))},
        "rules": {key: value for key, value in fitted.items()
                  if key != "_training_bounds"},
        "training_bounds": fitted["_training_bounds"],
        "qualification": ("Opened grouped development only. Older passing "
                          "starts and the independent V128 contact reserve "
                          "have not been scored for this candidate. The "
                          "historical Direct Frame is rejected."),
    }
    for row in data["pairs"]:
        expected = audit.predict(fitted, row["features"])
        actual = runtime.predict(model, row["features"])["shift_body_y_m"] < 0
        if actual != expected:
            raise ValueError("Serialized V135 runtime differs from fitted audit")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(model, indent=2) + "\n"
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Previously frozen V135 candidate changed")
    OUT.write_text(encoded)
    print(f"{OUT} sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
