"""Freeze the next contact-selection rule before V156 RGB entries are opened.

This writes a method, not a contact plan or result. The selected physical
starts are determined later from complete, source-checked camera observations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
V156_PLAN = BASE / "v156/independent-seed-entry-family-plan.json"
V156_MATRIX = BASE / "v156/native-entry-only/matrix.json"
V157_PLAN = BASE / "v157/exact-rgb-repeatability-plan.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
POINTER = ROOT / "robot/ape-current-design.json"
OUT = BASE / "v158/new-seed-contact-selection-method-freeze.json"
HELD_SEEDS = (24606, 24612, 24615, 24618, 24621, 24624)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not OUT.exists() and V156_MATRIX.exists():
        raise ValueError("V158 method must be frozen before V156 entries")
    entry = json.loads(V156_PLAN.read_text())
    repeat = json.loads(V157_PLAN.read_text())
    reserve = json.loads(RESERVE.read_text())
    pointer = json.loads(POINTER.read_text())
    families = entry["families"]
    held = set(HELD_SEEDS)
    if (entry["family_count"] != 24 or entry["case_count"] != 168 or
            entry["contains_contact_visits"] or
            repeat["case_count"] != 24 or repeat["contains_contact_visits"] or
            repeat["source_sha256"]["v156_plan"] != sha(V156_PLAN) or
            reserve["status"] != "sealed_no_contact_visits" or
            pointer["selected_design_id"] is not None or
            pointer["status"] != "rejected_no_selected_mechanical_design" or
            len(families) != 24 or
            held - {family["seed"] for family in families} or
            {family["stratum"] for family in families
             if family["seed"] in held} !=
            {"roll_live", "roll_locked", "pitch_locked"} or
            len({family["seed"] for family in families}) != 24):
        raise ValueError("V158 source or held seed split changed")
    plan = {
        "schema": "robot-learning-lab.cove50-v158-new-seed-contact-selection-method.v1",
        "role": "predeclared_new_seed_training_and_held_contact_method_not_results",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {
            "method_source": sha(Path(__file__)),
            "v156_entry_plan": sha(V156_PLAN),
            "v157_repeatability_plan": sha(V157_PLAN),
            "untouched_score_reserve": sha(RESERVE),
            "mechanical_pointer": sha(POINTER),
        },
        "prior_visibility": (
            "Frozen after V150/V152 development contacts and the V152 "
            "seed-24526 one-pixel RGB parity incident. The V155 one-seed "
            "camera map had partial results, but no V156 new-seed RGB entry, "
            "V157 exact repeat, or V147 reserved contact was opened."),
        "prerequisites": (
            "Complete source-checked V156 168 camera entries from 24 new "
            "seed families, complete source-checked V157 exact RGB repeat "
            "diagnostic, unchanged evaluator/body/actor/pointer sources, and "
            "the sealed V147 independent score reserve. Inspect V157 faults "
            "before authorizing contact collection; this method does not "
            "declare camera stability from entry counts alone."),
        "one_contact_start_per_seed_family": True,
        "entry_selection_rule": (
            "For each V156 family, consider only source-checked entries with "
            "a used controlled RGB goal. Prefer entries whose observed "
            "automatic route equals the family's frozen target route. Within "
            "that pool choose minimum squared toy_delta_x_m plus "
            "toy_delta_y_m, breaking ties by lexicographic entry id. If no "
            "target-route entry exists, use the same nearest-center ordering "
            "over all valid RGB entries and record target_route_absent. If a "
            "family has no valid RGB entry, record it unselected; never swap "
            "in a second contact start from that family after a failure."),
        "held_seed_ids": list(HELD_SEEDS),
        "development_seed_count": 18,
        "held_seed_count": 6,
        "held_split_rule": (
            "The six exact seed IDs above, two per physical stratum, are "
            "reserved before RGB and contact observations. Collect/freeze "
            "development contact first. Freeze any fitted selector and its "
            "thresholds before opening held contact. Never tune on held "
            "contact, and never count grid points within one seed as "
            "independent examples."),
        "paired_native_rule": (
            "On each selected development start, compare unchanged "
            "automatic full-native contact to exactly one prescribed route: "
            "force specialist if automatic is scoped, otherwise force "
            "scoped. Use the same seed, physical start and RGB goal source. "
            "Require screen-to-automatic and automatic-to-prescribed numeric "
            "RGB entry parity at the existing 0.1 mm tolerance and the saved "
            "full-native support, effort, collision, action and fingertip "
            "contact gates. Mark a parity mismatch invalid with its original "
            "source; a diagnostic repeat cannot replace the frozen pair. "
            "Held starts are opened only after selector freeze and evaluate "
            "automatic versus selected route under the same rules."),
        "development_gate": (
            "A selector may proceed to held contact only if a whole-seed "
            "withheld audit on development families demonstrates at least "
            "two independent full-gate and 1.2 s margin rescues, coverage "
            "across both roll and pitch physical cohorts, zero selected "
            "full-gate regressions, zero margin losses, and zero unmeasured "
            "selected alternatives. If training lacks held baseline failures "
            "in either cohort, record insufficient support and collect new "
            "development seeds under a separate freeze; do not relax this "
            "gate after seeing contact."),
        "held_gate": (
            "Report all six prespecified held seed outcomes, including "
            "missing RGB, parity faults and absent target routes. Require "
            "at least one selected full-gate rescue and no selected "
            "regression or 1.2 s margin loss before older-start/online "
            "retention. A held set with no baseline failure is inconclusive. "
            "The sealed V147 score and selected-body transfer remain later "
            "separate gates."),
        "contains_contact_results": False,
        "contains_policy_result": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(plan, indent=2) + "\n"
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Frozen V158 selection method changed")
    OUT.write_text(encoded)
    print(f"V158 contact method frozen before new-seed entry; sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
