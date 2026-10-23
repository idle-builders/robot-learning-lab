"""Audit the old selector recipe with each of 89 opened seeds withheld."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    v118_observable_selector as selector,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
DATASET = BASE / "v129/challenging-observable-pairs.json"
RESERVE = BASE / "v128/disjoint-contact-score-plan.json"
OUT = BASE / "v129/leave-one-seed-out-recipe-diagnostic.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = json.loads(DATASET.read_text())
    reserve = json.loads(RESERVE.read_text())
    rows = data["pairs"]
    if (not data["complete"] or len(rows) != 89 or
            len({row["seed_audit_only"] for row in rows}) != 89 or
            data["source_sha256"]["v128_sealed_reserve"] != sha(RESERVE) or
            reserve["status"] != "sealed_no_contact_visits"):
        raise ValueError("Complete 89-seed training and sealed V128 required")
    modes = sorted({row["features"]["selection_mode"] for row in rows})
    vectors = np.array([
        [float(row["features"][key]) for key in selector.NUMERIC_FEATURES] +
        [float(row["features"]["selection_mode"] == mode) for mode in modes]
        for row in rows])
    target = np.array([max(-2.0, min(2.0, row["contact_delta_s"]))
                       for row in rows])
    audit = []
    for i, held in enumerate(rows):
        train = [j for j in range(len(rows)) if j != i]
        fit = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                    random_state=4, n_jobs=1)
        fit.fit(vectors[train], target[train])
        prediction = float(fit.predict(vectors[i:i + 1])[0])
        features = held["features"]
        reasons = []
        if features["selection_mode"] not in {
                rows[j]["features"]["selection_mode"] for j in train}:
            reasons.append("selection_mode_outside_training")
        if features["route_goal_source"] not in {
                rows[j]["features"]["route_goal_source"] for j in train}:
            reasons.append("RGB_goal_source_outside_training")
        for k, name in enumerate(selector.NUMERIC_FEATURES):
            if not np.min(vectors[train, k]) <= vectors[i, k] <= np.max(vectors[train, k]):
                reasons.append(f"{name}_outside_training")
        audit.append({
            "seed_audit_only": held["seed_audit_only"],
            "cohort": held["cohort"],
            "selection_mode": features["selection_mode"],
            "predicted_clipped_gain_s": round(prediction, 4),
            "abstention_reasons": reasons,
            "baseline_contact_s": held["baseline_contact_s"],
            "shifted_contact_s": held["shifted_contact_s"],
            "baseline_pass": held["baseline_pass"],
            "shifted_pass": held["shifted_pass"],
            "baseline_margin_1p2_s_met": held["baseline_margin_1p2_s_met"],
            "shifted_margin_1p2_s_met": held["shifted_margin_1p2_s_met"],
        })
    summaries = []
    for threshold in (0.0, 0.4):
        chosen = [row for row in audit
                  if not row["abstention_reasons"] and
                  row["predicted_clipped_gain_s"] > threshold]
        summaries.append({
            "selection_threshold_s": threshold,
            "selected_count": len(chosen),
            "rescues": sum(not row["baseline_pass"] and row["shifted_pass"]
                           for row in chosen),
            "full_gate_regressions": sum(row["baseline_pass"] and
                                         not row["shifted_pass"] for row in chosen),
            "margin_losses": sum(row["baseline_margin_1p2_s_met"] and
                                 not row["shifted_margin_1p2_s_met"]
                                 for row in chosen),
            "selected_failing_seeds_audit_only": [
                row["seed_audit_only"] for row in chosen
                if not row["baseline_pass"]],
        })
    report = {
        "schema": "robot-learning-lab.cove50-v129-opened-recipe-audit.v1",
        "role": "opened_training_leave_one_seed_out_not_independent_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {"dataset": sha(DATASET), "reserve": sha(RESERVE),
                          "audit": sha(Path(__file__))},
        "method": ("Refit the preexisting 200-tree, two-leaf observed-input "
                   "recipe on the other 88 seeds, apply the same in-range "
                   "abstention, and compare 0.0/0.4 s choice thresholds. "
                   "No V128 contact label or score is used; this only "
                   "diagnoses whether new training coverage helps."),
        "pair_count": len(rows),
        "baseline_failure_count": sum(not row["baseline_pass"] for row in rows),
        "possible_rescue_count": sum(not row["baseline_pass"] and
                                     row["shifted_pass"] for row in rows),
        "summaries": summaries,
        "cases": audit,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2) + "\n"
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Prior V129 training audit changed")
    OUT.write_text(encoded)
    print(f"{OUT} sha256={sha(OUT)} summaries={summaries}")


if __name__ == "__main__":
    main()
