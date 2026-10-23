"""Frozen whole-seed audit of one-axis observed-input arm-route rules.

Run only after complete, source-checked V150 and V152 CUDA collections. A
passing training audit permits retention and online integration, not a score.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v139_reserved_contact as score_runner,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
METHOD = BASE / "v153/route-selector-method-freeze.json"
V150 = BASE / "v150/native-paired-contact/matrix.json"
V150_PLAN = BASE / "v150/paired-contact-training-plan.json"
V150_SUMMARY = BASE / "v150/source-checked-paired-summary.json"
V152_PLAN = BASE / "v152/paired-route-training-plan.json"
V152 = BASE / "v152/native-paired-routes/matrix.json"
V152_SUMMARY = BASE / "v152/source-checked-paired-route-summary.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
POINTER = ROOT / "robot/ape-current-design.json"
OUT = BASE / "v153/route-selector-grouped-audit.json"
FEATURES = (
    "entry_error_x_mm", "entry_error_y_mm", "camera_x_mm", "camera_y_mm",
    "orange_pixels", "rail_v", "delay_ms", "gaze_head_yaw_rad",
    "pitch_offset_rad", "roll_clearance_rad",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_summary(path: Path, matrix: Path, expected: int) -> dict:
    summary = json.loads(path.read_text())
    if (not summary["complete"] or
            summary["source_checked_pair_count"] != expected or
            summary["source_sha256"]["matrix"] != sha(matrix) or
            summary["source_checked_native_source_count"] != expected * 2):
        raise ValueError(f"Incomplete or changed source summary: {path}")
    return summary


def load_rows() -> tuple[list[dict], dict]:
    method = json.loads(METHOD.read_text())
    refs = method["source_sha256"]
    if (refs["audit_source"] != sha(Path(__file__)) or
            refs["v150_plan"] != sha(V150_PLAN) or
            refs["v152_plan"] != sha(V152_PLAN) or
            refs["reserve"] != sha(RESERVE) or
            refs["pointer"] != sha(POINTER) or
            json.loads(RESERVE.read_text())["status"] !=
            "sealed_no_contact_visits"):
        raise ValueError("Frozen method, source or sealed reserve changed")
    v150 = json.loads(V150.read_text())
    v152 = json.loads(V152.read_text())
    s150 = json.loads(V150_SUMMARY.read_text())
    s152 = verified_summary(V152_SUMMARY, V152, 48)
    if (not v150["complete"] or v150["case_count"] != 48 or
            not v152["complete"] or v152["case_count"] != 48 or
            s150["matrix_sha256"] != sha(V150) or
            not s150["complete"] or
            s150["source_checked_seed_count"] != 48 or
            s150["source_checked_native_visit_count"] != 144 or
            s150["screen_entry_parity_failure_count"] or
            s150["paired_entry_parity_failure_count"] or
            s152["numeric_rgb_entry_failure_count"] or
            v152["plan_sha256"] != sha(V152_PLAN)):
        raise ValueError("Complete source-checked V150 and V152 required")
    p152 = json.loads(V152_PLAN.read_text())
    plan = {case["id"]: case for case in p152["cases"]}
    if (len(plan) != 48 or len({case["seed"] for case in plan.values()}) != 48 or
            {case["seed"] for case in plan.values()} &
            {case["seed"] for case in json.loads(RESERVE.read_text())["cases"]} or
            {row["id"] for row in v152["cases"]} != set(plan)):
        raise ValueError("V152 cases differ from frozen plan")
    rows = []
    for pair in v152["cases"]:
        case = plan[pair["id"]]
        if (pair["seed"] != case["seed"] or
                score_runner.entry_parity(case["entry_observed_features"],
                                          pair["entry_observed_features"]) or
                pair["numeric_rgb_entry_faults"] or
                pair["prescribed_route"] != case["prescribed_route"]):
            raise ValueError(f"Changed paired source: {pair['id']}")
        rows.append({
            "seed_audit_only": case["seed"], "cohort_audit_only": case["cohort"],
            "features": pair["entry_observed_features"],
            "prescribed_route": case["prescribed_route"],
            "baseline_contact_s": pair["baseline_contact_s"],
            "baseline_pass": pair["baseline_pass"],
            "baseline_margin": pair["baseline_margin_1p2_s_met"],
            "alternate_contact_s": pair["prescribed_contact_s"],
            "alternate_pass": pair["prescribed_pass"],
            "alternate_margin": pair["prescribed_margin_1p2_s_met"],
        })
    return rows, {"method": sha(METHOD), "v150_matrix": sha(V150),
                  "v150_summary": sha(V150_SUMMARY), "v152_plan": sha(V152_PLAN),
                  "v152_matrix": sha(V152), "v152_summary": sha(V152_SUMMARY),
                  "reserve": sha(RESERVE)}


def repaired(row: dict) -> bool:
    return (not row["baseline_margin"] and row["alternate_pass"] and
            row["alternate_margin"])


def safe(row: dict) -> bool:
    return row["alternate_pass"] and row["alternate_margin"]


def matches(rule: dict, row: dict) -> bool:
    obs = row["features"]
    if (obs["selection_mode"] != rule["selection_mode"] or
            obs["route_goal_source"] != rule["route_goal_source"] or
            any(not limits[0] <= obs[key] <= limits[1]
                for key, limits in rule["training_bounds"].items())):
        return False
    value = obs[rule["feature"]]
    return value <= rule["threshold"] if rule["direction"] == "le" else value >= rule["threshold"]


def fit(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        obs = row["features"]
        groups[(obs["selection_mode"], obs["route_goal_source"])].append(row)
    rules = []
    for mode, goal_source in sorted(groups):
        group = groups[(mode, goal_source)]
        bounds = {key: [min(row["features"][key] for row in group),
                        max(row["features"][key] for row in group)] for key in FEATURES}
        best = None
        for feature in FEATURES:
            for direction in ("le", "ge"):
                for threshold in sorted({row["features"][feature] for row in group}):
                    rule = {"selection_mode": mode, "route_goal_source": goal_source,
                            "feature": feature, "direction": direction,
                            "threshold": threshold, "training_bounds": bounds}
                    selected = [row for row in group if matches(rule, row)]
                    rescue_count = sum(repaired(row) for row in selected)
                    if rescue_count < 2 or any(not safe(row) for row in selected):
                        continue
                    gain = round(sum(row["alternate_contact_s"] -
                                     row["baseline_contact_s"] for row in selected), 3)
                    key = (rescue_count, -len(selected), gain, -FEATURES.index(feature),
                           1 if direction == "le" else 0, -threshold)
                    if best is None or key > best[0]:
                        best = (key, rule)
        if best:
            rules.append(best[1])
    return rules


def predict(rules: list[dict], row: dict) -> str:
    for rule in rules:
        if matches(rule, row):
            return row["prescribed_route"]
    return "unchanged"


def summarize(rows: list[dict], decisions: list[str]) -> dict:
    chosen = [row for row, decision in zip(rows, decisions) if decision != "unchanged"]
    rescued = [row for row in chosen if repaired(row)]
    return {
        "selected_count": len(chosen),
        "held_margin_repairs": len(rescued),
        "held_full_gate_rescues": sum(not row["baseline_pass"] and
                                       row["alternate_pass"] and
                                       row["alternate_margin"] for row in chosen),
        "held_repair_cohorts": sorted({row["cohort_audit_only"] for row in rescued}),
        "selected_full_gate_regressions": sum(row["baseline_pass"] and
                                               not row["alternate_pass"] for row in chosen),
        "selected_margin_losses": sum(row["baseline_margin"] and
                                      not row["alternate_margin"] for row in chosen),
        "selected_unresolved": sum(not safe(row) for row in chosen),
        "selected_seeds_audit_only": [row["seed_audit_only"] for row in chosen],
        "repaired_seeds_audit_only": [row["seed_audit_only"] for row in rescued],
    }


def main() -> None:
    rows, source = load_rows()
    decisions = []
    for row in rows:
        held = [other for other in rows if other["seed_audit_only"] != row["seed_audit_only"]]
        decisions.append(predict(fit(held), row))
    held = summarize(rows, decisions)
    cohorts = {row["cohort_audit_only"] for row in rows}
    gate = (held["held_margin_repairs"] >= 2 and
            held["held_full_gate_rescues"] >= 1 and
            set(held["held_repair_cohorts"]) == cohorts and
            held["selected_full_gate_regressions"] == 0 and
            held["selected_margin_losses"] == 0 and
            held["selected_unresolved"] == 0)
    fitted = fit(rows)
    report = {
        "schema": "robot-learning-lab.cove50-v153-route-selector-grouped-audit.v1",
        "role": "whole_seed_grouped_training_audit_not_independent_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": source,
        "method": "Per automatic route and RGB goal source, fit at most one axis-aligned <= or >= threshold across the ten frozen observed numeric features. Require at least two training repairs and zero selected full-gate or 1.2 s margin failures; otherwise abstain. Select by repair count, smaller coverage, contact gain and fixed feature/direction/threshold order. Whole-seed withholding refits from scratch, with physical cohorts used only for the audit.",
        "held_audit": held,
        "development_gate_met": gate,
        "full_fit_rules_diagnostic_only": fitted,
        "full_fit_summary_diagnostic_only": summarize(rows, [predict(fitted, row) for row in rows]),
        "next_step": ("Freeze older-start native retention and online integration before opening the independent reserve" if gate else
                      "No selector promotion or independent score; collect more source-matched route variance or revise method on separate development data"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"development_gate_met": gate, **held}))


if __name__ == "__main__":
    main()
