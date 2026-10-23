"""Pin a grouped selector audit before V158 new-seed contact outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v160/new-seed-selector-audit-method-freeze.json"
SOURCES = {
    "audit_source": Path(__file__).with_name("audit_v160_new_seed_route_selector.py"),
    "method_source": Path(__file__),
    "v153_fit_source": Path(__file__).with_name("audit_v153_route_selector.py"),
    "v152_plan": BASE / "v152/paired-route-training-plan.json",
    "v152_matrix": BASE / "v152/native-paired-routes/matrix.json",
    "v152_summary": BASE / "v152/partial-source-checked-paired-route-summary.json",
    "v158_selection_method": BASE / "v158/new-seed-contact-selection-method-freeze.json",
    "reserve": BASE / "v147/untouched-score-reserve-plan.json",
    "pointer": ROOT / "robot/ape-current-design.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    old = json.loads(SOURCES["v152_matrix"].read_text())
    old_summary = json.loads(SOURCES["v152_summary"].read_text())
    selection = json.loads(SOURCES["v158_selection_method"].read_text())
    reserve = json.loads(SOURCES["reserve"].read_text())
    pointer = json.loads(SOURCES["pointer"].read_text())
    if (old["case_count"] != 47 or old["complete"] or
            old_summary["source_checked_pair_count"] != 47 or
            old_summary["source_sha256"]["matrix"] !=
            sha(SOURCES["v152_matrix"]) or
            old_summary["numeric_rgb_entry_failure_count"] != 0 or
            selection["contains_contact_results"] or
            selection["held_seed_ids"] !=
            [24606, 24612, 24615, 24618, 24621, 24624] or
            reserve["status"] != "sealed_no_contact_visits" or
            pointer["status"] != "rejected_no_selected_mechanical_design" or
            pointer["selected_design_id"] is not None or
            (BASE / "v158/native-development-contact/matrix.json").exists()):
        raise ValueError("V160 method must freeze before V158 development contact")
    value = {
        "schema": "robot-learning-lab.cove50-v160-new-seed-selector-audit-method.v1",
        "role": "predeclared_route_selector_audit_not_contact_or_score_result",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {key: sha(path) for key, path in SOURCES.items()},
        "prior_visibility": ("All 47 source-valid V152 historical paired routes, "
                             "their three full-gate rescues and nine regressions, "
                             "and the partial V156 camera route screen were known. "
                             "No V158 new-seed contact or held contact was opened."),
        "training_source": ("Use all 47 source-checked V152 paired historical "
                            "seeds plus source-valid V158 development pairs "
                            "from the other new seed families in each fold. "
                            "Never use invalid RGB pairs as training targets."),
        "rule_family": ("Use the already frozen V153 deterministic per-route, "
                        "per-RGB-source one-axis threshold search over ten "
                        "observed numeric entry and motor features. It requires "
                        "two training margin repairs, zero selected training "
                        "gate or margin failures, and feature bounds; otherwise "
                        "abstain to unchanged automatic route. No seed or "
                        "cohort is a runtime input."),
        "audit": ("Leave out an entire V158 development simulator seed, refit "
                  "from historical plus other valid development seeds, and "
                  "score the held seed exactly once. A selected invalid RGB "
                  "pair is unresolved; it is never assigned an alternative "
                  "contact result."),
        "development_gate": ("At least two independent selected full-gate "
                             "and 1.2 s margin rescues spanning both roll "
                             "and pitch physical cohorts; zero selected full "
                             "gate regressions, margin losses or unresolved "
                             "alternatives. A missing failure cohort is "
                             "insufficient support, not a passing selector."),
        "held_seed_ids": selection["held_seed_ids"],
        "held_boundary": ("If development passes, freeze the exact fitted "
                          "model and thresholds before opening any of the six "
                          "preassigned V158 held contacts. Older-start and "
                          "online retention and the separate V147 score remain "
                          "later gates. Rejected-body results cannot qualify "
                          "a selected mechanical design."),
        "contains_contact_results": False,
        "contains_policy_result": False,
    }
    encoded = json.dumps(value, indent=2) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("V160 audit method changed after freeze")
    OUT.write_text(encoded)
    print(f"V160 grouped selector method frozen sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
