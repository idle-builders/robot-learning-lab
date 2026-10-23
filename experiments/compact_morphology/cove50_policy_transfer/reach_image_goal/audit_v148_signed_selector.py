"""Whole-seed audit of a small observed-input signed reach-target rule family.

Rule family and gates are fixed before V147 paired collection finishes. A rule
fitted here is diagnostic until older-start retention, online integration and
an untouched native contact score are complete.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
DATA = BASE / "v148/signed-observed-training.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
OUT = BASE / "v148/signed-rule-grouped-audit.json"
MODES = ("v16_specialist_locked", "v21_scoped_y_locked")
NUMERIC_FEATURES = (
    "entry_error_x_mm", "entry_error_y_mm", "camera_x_mm", "camera_y_mm",
    "orange_pixels", "rail_v", "delay_ms", "gaze_head_yaw_rad",
    "pitch_offset_rad", "roll_clearance_rad",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meets_margin(action: dict) -> bool:
    return action["pass"] and action["margin_1p2_s_met"]


def bounds(rows: list[dict]) -> dict:
    return {key: [min(row["features"][key] for row in rows),
                  max(row["features"][key] for row in rows)]
            for key in NUMERIC_FEATURES}


def match(rule: dict | None, features: dict, action: str) -> bool:
    if (rule is None or features["selection_mode"] != rule["selection_mode"] or
            features["route_goal_source"] != "selected_image_goal" or
            any(not low <= features[key] <= high
                for key, (low, high) in rule["training_bounds"].items())):
        return False
    if action == "neg2":
        return (features["pitch_offset_rad"] >= .105 - 1e-9 and
                features["entry_error_x_mm"] >= rule["x_threshold_mm"] and
                (rule["y_threshold_mm"] is None or
                 features["entry_error_y_mm"] >= rule["y_threshold_mm"]))
    return (features["pitch_offset_rad"] <= .105 + 1e-9 and
            features["entry_error_x_mm"] <= rule["x_threshold_mm"] and
            features["entry_error_y_mm"] >= rule["y_threshold_mm"])


def fit_mode(rows: list[dict], action: str, mode: str) -> dict | None:
    available = [row for row in rows if row[action] is not None]
    if not available:
        return None
    rescue = [row for row in available
              if row["features"]["selection_mode"] == mode and
              row["features"]["route_goal_source"] == "selected_image_goal" and
              not meets_margin(row["unchanged"]) and
              meets_margin(row[action]) and
              ((row["features"]["pitch_offset_rad"] >= .105 - 1e-9)
               if action == "neg2" else
               (row["features"]["pitch_offset_rad"] <= .105 + 1e-9))]
    if len({row["seed_audit_only"] for row in rescue}) < 2:
        return None
    all_bounds = bounds(available)
    x_values = sorted({row["features"]["entry_error_x_mm"] for row in rescue})
    y_values = (sorted({row["features"]["entry_error_y_mm"] for row in rescue})
                if action == "pos2" or mode == "v16_specialist_locked" else [None])
    best = None
    for x in x_values:
        for y in y_values:
            rule = {"selection_mode": mode, "x_threshold_mm": x,
                    "y_threshold_mm": y, "training_bounds": all_bounds}
            chosen = [row for row in rows if match(rule, row["features"], action)]
            known = [row for row in chosen if row[action] is not None]
            rescued = {row["seed_audit_only"] for row in known
                       if not meets_margin(row["unchanged"]) and
                       meets_margin(row[action])}
            if len(rescued) < 2 or any(
                    not meets_margin(row[action]) or
                    (row["unchanged"]["margin_1p2_s_met"] and
                     not row[action]["margin_1p2_s_met"])
                    for row in known):
                continue
            unknown = len(chosen) - len(known)
            gain = sum(row[action]["contact_s"] - row["unchanged"]["contact_s"]
                       for row in known)
            score = (len(rescued), -unknown, -len(chosen), round(gain, 3),
                     x if action == "neg2" else -x, y or 0)
            if best is None or score > best[0]:
                best = (score, rule)
    return best[1] if best else None


def fit(rows: list[dict]) -> dict:
    return {action: {mode: fit_mode(rows, action, mode) for mode in MODES}
            for action in ("neg2", "pos2")}


def predict(model: dict, features: dict) -> str:
    selected = [action for action in ("neg2", "pos2")
                if match(model[action].get(features["selection_mode"]), features,
                         action)]
    return selected[0] if selected else "unchanged"


def summarize(rows: list[dict], decisions: list[str]) -> dict:
    selected = [(row, action) for row, action in zip(rows, decisions)
                if action != "unchanged"]
    known = [(row, action) for row, action in selected if row[action] is not None]
    rescue_rows = [(row, action) for row, action in known
                   if not meets_margin(row["unchanged"]) and
                   meets_margin(row[action])]
    rescue = {row["seed_audit_only"] for row, _ in rescue_rows}
    possible = {row["seed_audit_only"] for row in rows
                if not meets_margin(row["unchanged"]) and
                any(row[action] is not None and meets_margin(row[action])
                    for action in ("neg2", "pos2"))}
    opened_resolved = sorted({row["seed_audit_only"] for row, _ in rescue_rows
                              if row["cohort"] == "opened_score_miss"})
    return {
        "selected_pair_count": len(selected),
        "selected_negative_count": sum(action == "neg2" for _, action in selected),
        "selected_positive_count": sum(action == "pos2" for _, action in selected),
        "selected_unmeasured_positive_count": len(selected) - len(known),
        "possible_rescue_seed_count": len(possible),
        "rescued_seed_count": len(rescue),
        "rescued_v147_seed_count": len({row["seed_audit_only"]
                                        for row, _ in rescue_rows
                                        if row["cohort"] == "v147"}),
        "rescued_v147_negative_seed_count": len({row["seed_audit_only"]
                                                 for row, action in rescue_rows
                                                 if row["cohort"] == "v147" and
                                                 action == "neg2"}),
        "rescued_v147_positive_seed_count": len({row["seed_audit_only"]
                                                 for row, action in rescue_rows
                                                 if row["cohort"] == "v147" and
                                                 action == "pos2"}),
        "opened_score_misses_resolved_audit_only": opened_resolved,
        "full_gate_regressions": sum(row["unchanged"]["pass"] and
                                     not row[action]["pass"] for row, action in known),
        "margin_losses": sum(row["unchanged"]["margin_1p2_s_met"] and
                             not row[action]["margin_1p2_s_met"]
                             for row, action in known),
        "selected_unresolved_or_margin_misses": sum(
            not row[action]["pass"] or not row[action]["margin_1p2_s_met"]
            for row, action in known),
        "selected_contact_change_s": round(sum(
            row[action]["contact_s"] - row["unchanged"]["contact_s"]
            for row, action in known), 3),
        "selected_harm_or_unknown_seeds_audit_only": sorted({
            row["seed_audit_only"] for row, action in selected
            if row[action] is None or not row[action]["pass"] or
            not row[action]["margin_1p2_s_met"]}),
    }


def main() -> None:
    data, reserve = json.loads(DATA.read_text()), json.loads(RESERVE.read_text())
    rows = data["rows"]
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["seed_audit_only"]].append(index)
    if (data["pair_count"] != 182 or len(groups) != 142 or
            data["positive_pair_count"] != 68 or
            data["source_sha256"]["v147_score_reserve"] != sha(RESERVE) or
            reserve["status"] != "sealed_no_contact_visits" or
            set(groups) & {case["seed"] for case in reserve["cases"]}):
        raise ValueError("Complete paired training and sealed score reserve required")
    for row in rows:
        for action in ("unchanged", "neg2", "pos2"):
            saved = row[action]
            if saved is None:
                continue
            if (sha(ROOT / saved["report"]) != saved["report_sha256"] or
                    (saved.get("trace") and
                     sha(ROOT / saved["trace"]) != saved["trace_sha256"])):
                raise ValueError(f"Source report or trace changed: {row['id']} {action}")
    held_decisions = ["unchanged"] * len(rows)
    for seed, indexes in groups.items():
        training = [row for row in rows if row["seed_audit_only"] != seed]
        model = fit(training)
        for index in indexes:
            held_decisions[index] = predict(model, rows[index]["features"])
    crossfit = summarize(rows, held_decisions)
    fitted = fit(rows)
    fitted_decisions = [predict(fitted, row["features"]) for row in rows]
    report = {
        "schema": "robot-learning-lab.cove50-v148-signed-rule-grouped-audit.v1",
        "role": "whole_seed_grouped_training_audit_not_independent_score",
        "body_role": data["body_role"],
        "source_sha256": {"dataset": sha(DATA), "reserve": sha(RESERVE),
                          "audit": sha(Path(__file__))},
        "rule_family": ("For each observed route, negative shift uses X >= a "
                        "fitted rescue boundary and specialist Y >= a fitted "
                        "boundary, with the previously opened pitch >=0.105 "
                        "guard. Positive shift uses X <= and Y >= fitted "
                        "rescue boundaries, pitch <=0.105. Negative takes "
                        "precedence if both rules match. Fit only candidates "
                        "covering at least two independent rescue seed groups "
                        "and no measured gate or margin failures. Abstain on "
                        "unknown modes or outside training bounds. Inputs are "
                        "observed RGB and state only."),
        "development_gate": ("At least one held V147 rescue seed group for "
                             "each signed action, both opened V139 score misses "
                             "repaired under whole-seed withholding, and zero "
                             "held full-gate regressions, margin losses, "
                             "unresolved selections and unmeasured positive "
                             "selections. Passing this only permits frozen "
                             "older-start and online native retention before "
                             "the separate score reserve; it does not qualify "
                             "a learned controller or new body."),
        "crossfit": crossfit,
        "crossfit_gate_met": (
            crossfit["rescued_v147_negative_seed_count"] >= 1 and
            crossfit["rescued_v147_positive_seed_count"] >= 1 and
            crossfit["opened_score_misses_resolved_audit_only"] == [24111, 24119] and
            not any(crossfit[key] for key in (
                "full_gate_regressions", "margin_losses",
                "selected_unresolved_or_margin_misses",
                "selected_unmeasured_positive_count"))),
        "full_fit_diagnostic_only": fitted,
        "full_fit_training_summary": summarize(rows, fitted_decisions),
        "held_decisions": [{"id": row["id"],
                            "seed_audit_only": row["seed_audit_only"],
                            "cohort": row["cohort"], "action": action,
                            "unchanged_contact_s": row["unchanged"]["contact_s"],
                            "chosen_contact_s": (row[action]["contact_s"]
                                                 if row[action] is not None else None)}
                           for row, action in zip(rows, held_decisions)],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2) + "\n"
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Saved V148 grouped audit changed")
    OUT.write_text(encoded)
    print(f"V148 grouped audit: {crossfit['rescued_v147_seed_count']} held "
          "V147 rescues; "
          f"{crossfit['full_gate_regressions']} regressions; "
          f"gate={report['crossfit_gate_met']}; sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
