"""Group-audit a compact observed-RGB reach correction rule family.

The specialist can use entry image-goal Y error; scoped-Y can use X error.
Thresholds are fitted from paired training contact only. Seed is used only to
withhold whole groups, never at inference. This is not a contact score.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
DATASET = BASE / "v131/failure-augmented-observable-pairs.json"
RESERVE = BASE / "v128/disjoint-contact-score-plan.json"
OUT = BASE / "v133/two-feature-grouped-audit.json"
MODE_FEATURES = {
    "v16_specialist_locked": "entry_error_y_mm",
    "v21_scoped_y_locked": "entry_error_x_mm",
}
WEIGHT_SCHEMES = ("uniform_pair", "equal_seed")
NUMERIC_FEATURES = (
    "entry_error_x_mm", "entry_error_y_mm", "camera_x_mm", "camera_y_mm",
    "orange_pixels", "rail_v", "delay_ms", "gaze_head_yaw_rad",
    "pitch_offset_rad", "roll_clearance_rad",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_rules(rows: list[dict], weighting: str) -> dict:
    counts = Counter(row["seed_audit_only"] for row in rows)
    rules = {}
    for mode, feature in MODE_FEATURES.items():
        cases = [row for row in rows
                 if row["features"]["selection_mode"] == mode and
                 row["features"]["route_goal_source"] == "selected_image_goal"]
        values = sorted({row["features"][feature] for row in cases})
        thresholds = ([None] +
                      [(left + right) / 2 for left, right in zip(values, values[1:])] +
                      ([values[0] - 1e-6] if values else []))
        best = (0.0, 0, 0.0)
        chosen_threshold = None
        for threshold in thresholds:
            selected = ([] if threshold is None else
                        [row for row in cases if row["features"][feature] >= threshold])
            if any((row["baseline_pass"] and not row["shifted_pass"]) or
                   (row["baseline_margin_1p2_s_met"] and
                    not row["shifted_margin_1p2_s_met"])
                   for row in selected):
                continue
            weight = lambda row: (1.0 if weighting == "uniform_pair" else
                                  1.0 / counts[row["seed_audit_only"]])
            rescued = sum(weight(row) for row in selected
                          if not row["baseline_pass"] and row["shifted_pass"])
            contact_gain = sum(weight(row) * row["contact_delta_s"]
                               for row in selected)
            score = (round(rescued, 8), -len(selected), round(contact_gain, 8))
            if score > best:
                best = score
                chosen_threshold = threshold
        rules[mode] = {"feature": feature, "threshold": chosen_threshold,
                       "condition": "observed_feature_gte_threshold",
                       "training_weighting": weighting,
                       "weighted_rescues": best[0],
                       "selected_training_pairs": -best[1]}
    rules["_training_bounds"] = {
        key: [min(row["features"][key] for row in rows),
              max(row["features"][key] for row in rows)]
        for key in NUMERIC_FEATURES
    }
    return rules


def predict(rules: dict, features: dict) -> bool:
    if features["route_goal_source"] != "selected_image_goal":
        return False
    if any(not low <= features[key] <= high
           for key, (low, high) in rules["_training_bounds"].items()):
        return False
    rule = rules.get(features["selection_mode"])
    return (rule is not None and rule["threshold"] is not None and
            features[rule["feature"]] >= rule["threshold"])


def summarize(rows: list[dict], selected: list[bool]) -> dict:
    chosen = [row for row, use in zip(rows, selected) if use]
    possible = {row["seed_audit_only"] for row in rows
                if not row["baseline_pass"] and row["shifted_pass"]}
    rescued = {row["seed_audit_only"] for row in chosen
               if not row["baseline_pass"] and row["shifted_pass"]}
    return {
        "selected_pair_count": len(chosen),
        "possible_rescue_pair_count": sum(not row["baseline_pass"] and
                                          row["shifted_pass"] for row in rows),
        "selected_rescue_pair_count": sum(not row["baseline_pass"] and
                                          row["shifted_pass"] for row in chosen),
        "possible_rescue_seed_count": len(possible),
        "selected_rescue_seed_count": len(rescued),
        "selected_full_gate_regressions": sum(row["baseline_pass"] and
                                              not row["shifted_pass"]
                                              for row in chosen),
        "selected_margin_losses": sum(row["baseline_margin_1p2_s_met"] and
                                      not row["shifted_margin_1p2_s_met"]
                                      for row in chosen),
        "selected_contact_change_s": round(sum(row["contact_delta_s"]
                                               for row in chosen), 3),
        "rescued_seeds_audit_only": sorted(rescued),
        "selected_failing_seeds_audit_only": sorted({
            row["seed_audit_only"] for row in chosen if not row["shifted_pass"]}),
    }


def main() -> None:
    data = json.loads(DATASET.read_text())
    reserve = json.loads(RESERVE.read_text())
    rows = data["pairs"]
    seeds = {row["seed_audit_only"] for row in rows}
    if (not data["complete"] or len(rows) != 113 or len(seeds) != 89 or
            data["source_sha256"]["v128_sealed_reserve"] != sha(RESERVE) or
            reserve["status"] != "sealed_no_contact_visits" or
            {case["seed"] for case in reserve["cases"]} & seeds):
        raise ValueError("Complete training and disjoint sealed reserve required")
    audits = []
    for weighting in WEIGHT_SCHEMES:
        selected = [False] * len(rows)
        thresholds_by_held_seed = {}
        for seed in sorted(seeds):
            training = [row for row in rows if row["seed_audit_only"] != seed]
            rules = fit_rules(training, weighting)
            thresholds_by_held_seed[str(seed)] = {
                mode: rules[mode]["threshold"] for mode in MODE_FEATURES}
            for i, held in enumerate(rows):
                if held["seed_audit_only"] == seed:
                    selected[i] = predict(rules, held["features"])
        audits.append({"weighting": weighting,
                       **summarize(rows, selected),
                       "held_decisions": [{
                           "seed_audit_only": row["seed_audit_only"],
                           "case_id_audit_only": row.get("case_id_audit_only"),
                           "cohort": row["cohort"],
                           "selection_mode": row["features"]["selection_mode"],
                           "selected_shift": selected[i],
                           "baseline_contact_s": row["baseline_contact_s"],
                           "shifted_contact_s": row["shifted_contact_s"],
                           "baseline_pass": row["baseline_pass"],
                           "shifted_pass": row["shifted_pass"],
                           "baseline_result": row["baseline_result"],
                           "shifted_result": row["shifted_result"],
                       } for i, row in enumerate(rows)],
                       "thresholds_by_held_seed_audit_only": thresholds_by_held_seed,
                       "full_training_rules_diagnostic_only": fit_rules(rows, weighting)})
    report = {
        "schema": "robot-learning-lab.cove50-v133-two-feature-group-audit.v1",
        "role": "opened_training_group_crossfit_not_independent_contact_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {"dataset": sha(DATASET), "reserve": sha(RESERVE),
                          "audit": sha(Path(__file__))},
        "method": ("Predeclared two-mode family: specialist shifts when "
                   "observed image-goal Y exceeds a fitted threshold; scoped-Y "
                   "shifts when observed X exceeds a fitted threshold; other "
                   "routes stay unchanged. For each training fold, fit each "
                   "threshold to maximize weighted full-gate rescues with zero "
                   "selected full-gate or 1.2 s margin regressions, then "
                   "prefer fewer interventions. Withhold all physical "
                   "variations of the held seed. Compare uniform-pair and "
                   "equal-seed weighting. Seed and simulator truth never enter "
                   "inference. Abstain when any observed numeric input lies "
                   "outside the current fold's training bounds. This "
                   "training audit cannot promote the router."),
        "pair_count": len(rows), "seed_group_count": len(seeds),
        "audits": audits,
        "cases": [{"id": row["weighting"],
                   **{key: value for key, value in row.items()
                      if key not in ("held_decisions",
                                     "thresholds_by_held_seed_audit_only",
                                     "full_training_rules_diagnostic_only")}}
                  for row in audits],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2) + "\n"
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Prior V133 grouped rule audit changed")
    OUT.write_text(encoded)
    print(f"{OUT} sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
