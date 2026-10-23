"""Check whether frozen V150 offers enough held-seed repair opportunities for V153.

This uses no V152 prescribed-route outcomes and is not a selector evaluation.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
METHOD = BASE / "v153/route-selector-method-freeze.json"
V150_PLAN = BASE / "v150/paired-contact-training-plan.json"
V150_MATRIX = BASE / "v150/native-paired-contact/matrix.json"
V150_SUMMARY = BASE / "v150/source-checked-paired-summary.json"
V152_PLAN = BASE / "v152/paired-route-training-plan.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
POINTER = ROOT / "robot/ape-current-design.json"
AUDIT_SOURCE = Path(__file__).with_name("audit_v153_route_selector.py")
OUT = BASE / "v153/opportunity-support-preflight.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    method = json.loads(METHOD.read_text())
    refs = method["source_sha256"]
    v150 = json.loads(V150_MATRIX.read_text())
    summary = json.loads(V150_SUMMARY.read_text())
    v152_plan = json.loads(V152_PLAN.read_text())
    reserve = json.loads(RESERVE.read_text())
    pointer = json.loads(POINTER.read_text())
    if (method["status"] != "frozen_before_v152_paired_contact" or
            refs["audit_source"] != sha(AUDIT_SOURCE) or
            refs["v150_plan"] != sha(V150_PLAN) or
            refs["v152_plan"] != sha(V152_PLAN) or
            refs["reserve"] != sha(RESERVE) or
            refs["pointer"] != sha(POINTER) or
            v152_plan["case_count"] != 48 or
            v150["plan_sha256"] != sha(V150_PLAN) or
            v150["case_count"] != 48 or not v150["complete"] or
            summary["matrix_sha256"] != sha(V150_MATRIX) or
            summary["plan_sha256"] != sha(V150_PLAN) or
            summary["source_checked_seed_count"] != 48 or
            summary["source_checked_native_visit_count"] != 144 or
            summary["screen_entry_parity_failure_count"] or
            summary["paired_entry_parity_failure_count"] or
            not summary["complete"] or
            reserve["status"] != "sealed_no_contact_visits" or
            pointer["selected_design_id"] is not None):
        raise ValueError("Frozen V153 method or complete source-checked V150 differs")
    if len({row["seed"] for row in v150["cases"]}) != 48:
        raise ValueError("Whole-seed support needs 48 distinct V150 seeds")

    misses = [row for row in v150["cases"]
              if not row["baseline_margin_1p2_s_met"]]
    group = lambda row: (row["full_entry_observed_features"]["selection_mode"],
                         row["full_entry_observed_features"]["route_goal_source"])
    counts = Counter(group(row) for row in misses)
    # The pinned V153 fit requires two *other* repairs in the same route/source
    # group. Each repair must start as a V150 baseline margin miss. Withholding
    # one seed leaves at most group_size - 1 such training opportunities.
    min_other_repairs = 2
    if ("at least two independent training repairs" not in method["rule_family"] or
            "Leave out every row from one simulator seed" not in method["withholding"]):
        raise ValueError("V153 frozen rule support contract changed")
    cases = []
    for row in misses:
        mode, source = group(row)
        available_other = counts[(mode, source)] - 1
        cases.append({
            "id": f"held-support-{row['seed']}",
            "seed_audit_only": row["seed"],
            "cohort_audit_only": row["cohort"],
            "automatic_route": mode,
            "rgb_goal_source": source,
            "baseline_contact_s": row["baseline_contact_s"],
            "baseline_full_gate_pass": row["baseline_pass"],
            "same_group_margin_misses_total": counts[(mode, source)],
            "maximum_other_training_repairs": available_other,
            "minimum_other_training_repairs_required": min_other_repairs,
            "held_rule_can_exist_even_if_all_alternatives_repair":
                available_other >= min_other_repairs,
            "pass": available_other >= min_other_repairs,
            "faults": ([] if available_other >= min_other_repairs else
                       [f"Only {available_other} other same-group baseline "
                        "margin misses; frozen fit needs two repairs"]),
        })
    eligible = sum(case["held_rule_can_exist_even_if_all_alternatives_repair"]
                   for case in cases)
    eligible_full = sum(case["held_rule_can_exist_even_if_all_alternatives_repair"]
                        and not case["baseline_full_gate_pass"] for case in cases)
    result = {
        "schema": "robot-learning-lab.cove50-v153-opportunity-support-preflight.v1",
        "role": "source_checked_preflight_not_contact_outcome_or_selector_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {
            "method": sha(METHOD), "audit_source": sha(AUDIT_SOURCE),
            "v150_plan": sha(V150_PLAN), "v150_matrix": sha(V150_MATRIX),
            "v150_summary": sha(V150_SUMMARY), "v152_plan": sha(V152_PLAN),
            "reserve": sha(RESERVE), "pointer": sha(POINTER),
            "preflight_source": sha(Path(__file__)),
        },
        "v152_prescribed_outcomes_used": False,
        "case_count": len(cases), "planned_count": len(cases),
        "complete": True, "pass_count": eligible,
        "cases": cases,
        "v150_baseline_margin_miss_count": len(misses),
        "v150_baseline_full_gate_failure_count": sum(
            not row["baseline_pass"] for row in v150["cases"]),
        "minimum_other_training_repairs_per_group": min_other_repairs,
        "maximum_held_margin_repairs_under_frozen_rule_family": eligible,
        "required_held_margin_repairs": 2,
        "maximum_held_full_gate_rescues_under_frozen_rule_family": eligible_full,
        "required_held_full_gate_rescues": 1,
        "frozen_v153_training_gate_mathematically_possible": eligible >= 2 and
            eligible_full >= 1 and
            len({case["cohort_audit_only"] for case in cases
                 if case["held_rule_can_exist_even_if_all_alternatives_repair"]}) == 2,
        "baseline_miss_cases": cases,
        "interpretation": (
            "Even if every V152 prescribed alternative repairs its matching V150 "
            "baseline miss, no held seed can be selected by the frozen V153 fit: "
            "each route/RGB-source group has fewer than two other training "
            "repair opportunities after that seed is withheld. The V153 held "
            "gate cannot pass on these 48 starts. V152 route comparisons may "
            "still be useful training evidence; this is not a V152 outcome, "
            "an automatic policy, an independent score, or body qualification."),
        "next_data_need": (
            "Collect additional source-matched baseline margin misses with "
            "varied observed RGB and physical conditions in both cohorts, "
            "then freeze a new grouped method and independent reserve before "
            "contact-informed fitting. Do not alter the V153 freeze or open "
            "the V147 score reserve."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUT.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temp, OUT)
    print(json.dumps({key: result[key] for key in (
        "v150_baseline_margin_miss_count",
        "maximum_held_margin_repairs_under_frozen_rule_family",
        "frozen_v153_training_gate_mathematically_possible")}))


if __name__ == "__main__":
    main()
