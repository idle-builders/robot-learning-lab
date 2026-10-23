"""Group-audit a small observed-image-error correction rule family.

The thresholds are learned only from opened paired contact. Repeated physical
conditions of one seed always remain in the same fold. This is a development
audit, not a contact score or a selected-body policy qualification.
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
OUT = BASE / "v134/planar-rule-grouped-audit.json"
SPECIALIST = "v16_specialist_locked"
SCOPED = "v21_scoped_y_locked"
NUMERIC_FEATURES = (
    "entry_error_x_mm", "entry_error_y_mm", "camera_x_mm", "camera_y_mm",
    "orange_pixels", "rail_v", "delay_ms", "gaze_head_yaw_rad",
    "pitch_offset_rad", "roll_clearance_rad",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def thresholds(values: list[float]) -> list[float]:
    ordered = sorted(set(values))
    return ([ordered[0] - 1e-6] +
            [(left + right) / 2 for left, right in zip(ordered, ordered[1:])]
            if ordered else [])


def fit_mode(rows: list[dict], mode: str, weighting: str) -> dict:
    cases = [row for row in rows
             if row["features"]["selection_mode"] == mode and
             row["features"]["route_goal_source"] == "selected_image_goal"]
    seed_counts = Counter(row["seed_audit_only"] for row in rows)
    x_values = thresholds([row["features"]["entry_error_x_mm"] for row in cases])
    y_values = (thresholds([row["features"]["entry_error_y_mm"] for row in cases])
                if mode == SPECIALIST else [None])
    best_score = (0.0, 0.0, 0.0, float("-inf"), float("-inf"))
    best_rule = {"entry_error_x_mm_gte": None, "entry_error_y_mm_gte": None}
    for x_threshold in x_values:
        x_cases = [row for row in cases
                   if row["features"]["entry_error_x_mm"] >= x_threshold]
        for y_threshold in y_values:
            selected = (x_cases if y_threshold is None else
                        [row for row in x_cases
                         if row["features"]["entry_error_y_mm"] >= y_threshold])
            if any((row["baseline_pass"] and not row["shifted_pass"]) or
                   (row["baseline_margin_1p2_s_met"] and
                    not row["shifted_margin_1p2_s_met"]) or
                   (not row["baseline_pass"] and not row["shifted_pass"])
                   for row in selected):
                continue
            weight = lambda row: (1.0 if weighting == "uniform_pair" else
                                  1.0 / seed_counts[row["seed_audit_only"]])
            rescue = sum(weight(row) for row in selected
                         if not row["baseline_pass"] and row["shifted_pass"])
            gain = sum(weight(row) * row["contact_delta_s"] for row in selected)
            score = (round(rescue, 8), -len(selected), round(gain, 8),
                     x_threshold, y_threshold if y_threshold is not None else 0.0)
            if score > best_score:
                best_score = score
                best_rule = {"entry_error_x_mm_gte": x_threshold,
                             "entry_error_y_mm_gte": y_threshold}
    return {**best_rule, "training_weighting": weighting,
            "weighted_rescues": best_score[0],
            "weighted_contact_gain_s": best_score[2],
            "selected_training_pairs": -best_score[1]}


def fit(rows: list[dict], weighting: str) -> dict:
    rules = {mode: fit_mode(rows, mode, weighting)
             for mode in (SPECIALIST, SCOPED)}
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
    return (rule is not None and rule["entry_error_x_mm_gte"] is not None and
            features["entry_error_x_mm"] >= rule["entry_error_x_mm_gte"] and
            (rule["entry_error_y_mm_gte"] is None or
             features["entry_error_y_mm"] >= rule["entry_error_y_mm_gte"]))


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
        "selected_unresolved_failures": sum(not row["shifted_pass"]
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
        raise ValueError("Complete grouped training and disjoint sealed reserve required")
    audits = []
    for weighting in ("uniform_pair", "equal_seed"):
        selected = [False] * len(rows)
        for seed in sorted(seeds):
            training = [row for row in rows if row["seed_audit_only"] != seed]
            rules = fit(training, weighting)
            for i, held in enumerate(rows):
                if held["seed_audit_only"] == seed:
                    selected[i] = predict(rules, held["features"])
        audits.append({"weighting": weighting, **summarize(rows, selected),
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
                       "full_training_rules_diagnostic_only": fit(rows, weighting)})
    report = {
        "schema": "robot-learning-lab.cove50-v134-planar-rule-group-audit.v1",
        "role": "opened_training_group_crossfit_not_independent_contact_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {"dataset": sha(DATASET), "reserve": sha(RESERVE),
                          "audit": sha(Path(__file__))},
        "method": ("Predeclared observed-image-error rule: specialist shifts "
                   "when entry X and Y exceed fitted lower bounds; scoped-Y "
                   "shifts when entry X exceeds a fitted lower bound; other "
                   "routes stay unchanged. Fit thresholds on paired opened "
                   "contact, forbidding selected full-gate regressions, 1.2 s "
                   "margin losses and unresolved failures. Optimize weighted "
                   "rescues, then fewer interventions, then net contact time; "
                   "identical training predictions use the stricter bound. "
                   "Compare pair and equal-seed weighting; withhold every "
                   "physical variation of the held seed. Abstain outside "
                   "observed numeric training bounds. Seed and simulator "
                   "truth are audit-only and never enter inference. V128 "
                   "contact remains sealed; this does not promote a router."),
        "pair_count": len(rows), "seed_group_count": len(seeds),
        "audits": audits,
        "cases": [{"id": row["weighting"],
                   **{key: value for key, value in row.items()
                      if key not in ("held_decisions",
                                     "full_training_rules_diagnostic_only")}}
                  for row in audits],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2) + "\n"
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Prior V134 grouped rule audit changed")
    OUT.write_text(encoded)
    print(f"{OUT} sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
