"""Audit selector recipes while withholding whole training seed groups."""

from __future__ import annotations

from collections import Counter
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
DATASET = BASE / "v131/failure-augmented-observable-pairs.json"
RESERVE = BASE / "v128/disjoint-contact-score-plan.json"
OUT = BASE / "v132/grouped-opened-recipe-audit.json"
THRESHOLDS = (0.0, 0.4)
WEIGHTS = ("uniform_pair", "equal_seed", "failure_emphasis",
           "equal_seed_failure_emphasis")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows: list[dict], selected: list[bool]) -> dict:
    chosen = [row for row, use in zip(rows, selected) if use]
    possible_rescue_seeds = {row["seed_audit_only"] for row in rows
                             if not row["baseline_pass"] and row["shifted_pass"]}
    rescued_seeds = {row["seed_audit_only"] for row in chosen
                     if not row["baseline_pass"] and row["shifted_pass"]}
    return {
        "selected_pair_count": len(chosen),
        "baseline_failure_pair_count": sum(not row["baseline_pass"] for row in rows),
        "possible_rescue_pair_count": sum(not row["baseline_pass"] and
                                          row["shifted_pass"] for row in rows),
        "selected_rescue_pair_count": sum(not row["baseline_pass"] and
                                          row["shifted_pass"] for row in chosen),
        "possible_rescue_seed_count": len(possible_rescue_seeds),
        "selected_rescue_seed_count": len(rescued_seeds),
        "selected_full_gate_regressions": sum(row["baseline_pass"] and
                                              not row["shifted_pass"]
                                              for row in chosen),
        "selected_margin_losses": sum(row["baseline_margin_1p2_s_met"] and
                                      not row["shifted_margin_1p2_s_met"]
                                      for row in chosen),
        "selected_contact_change_s": round(sum(row["contact_delta_s"]
                                               for row in chosen), 3),
        "rescued_seeds_audit_only": sorted(rescued_seeds),
        "selected_failing_seeds_audit_only": sorted({
            row["seed_audit_only"] for row in chosen if not row["shifted_pass"]}),
    }


def main() -> None:
    data = json.loads(DATASET.read_text())
    reserve = json.loads(RESERVE.read_text())
    rows = data["pairs"]
    if (not data["complete"] or len(rows) != 113 or
            len({row["seed_audit_only"] for row in rows}) != 89 or
            data["source_sha256"]["v128_sealed_reserve"] != sha(RESERVE) or
            reserve["status"] != "sealed_no_contact_visits"):
        raise ValueError("Complete grouped training and sealed reserve required")
    modes = sorted({row["features"]["selection_mode"] for row in rows})
    values = np.array([
        [float(row["features"][key]) for key in selector.NUMERIC_FEATURES] +
        [float(row["features"]["selection_mode"] == mode) for mode in modes]
        for row in rows], dtype=float)
    targets = np.array([max(-2.0, min(2.0, row["contact_delta_s"]))
                        for row in rows])
    seed_counts = Counter(row["seed_audit_only"] for row in rows)
    predictions = {weight: [None] * len(rows) for weight in WEIGHTS}
    inside = [False] * len(rows)
    for seed in sorted(seed_counts):
        train = [i for i, row in enumerate(rows)
                 if row["seed_audit_only"] != seed]
        held = [i for i, row in enumerate(rows)
                if row["seed_audit_only"] == seed]
        train_modes = {rows[i]["features"]["selection_mode"] for i in train}
        train_sources = {rows[i]["features"]["route_goal_source"] for i in train}
        lower = np.min(values[train, :len(selector.NUMERIC_FEATURES)], axis=0)
        upper = np.max(values[train, :len(selector.NUMERIC_FEATURES)], axis=0)
        for i in held:
            feature = rows[i]["features"]
            inside[i] = (feature["selection_mode"] in train_modes and
                         feature["route_goal_source"] in train_sources and
                         bool(np.all(lower <= values[i, :len(lower)])) and
                         bool(np.all(values[i, :len(upper)] <= upper)))
        for weight_name in WEIGHTS:
            weights = (None if weight_name == "uniform_pair" else
                       [(1.0 / seed_counts[rows[i]["seed_audit_only"]]
                         if weight_name.startswith("equal_seed") else 1.0) *
                        (5.0 if "failure_emphasis" in weight_name and
                         not rows[i]["baseline_pass"] else 1.0)
                        for i in train])
            fit = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                        random_state=4, n_jobs=1)
            fit.fit(values[train], targets[train], sample_weight=weights)
            outputs = fit.predict(values[held])
            for i, prediction in zip(held, outputs):
                predictions[weight_name][i] = float(prediction)
    audits = []
    for weight_name in WEIGHTS:
        for threshold in THRESHOLDS:
            selected = [inside[i] and predictions[weight_name][i] > threshold
                        for i in range(len(rows))]
            audits.append({"sample_weight": weight_name,
                           "selection_threshold_s": threshold,
                           **summary(rows, selected),
                           "held_decisions": [{
                               "seed_audit_only": row["seed_audit_only"],
                               "case_id_audit_only": row.get("case_id_audit_only"),
                               "cohort": row["cohort"],
                               "selection_mode": row["features"]["selection_mode"],
                               "predicted_clipped_gain_s": round(predictions[weight_name][i], 4),
                               "inside_training_bounds": inside[i],
                               "selected_shift": selected[i],
                               "baseline_contact_s": row["baseline_contact_s"],
                               "shifted_contact_s": row["shifted_contact_s"],
                               "baseline_pass": row["baseline_pass"],
                               "shifted_pass": row["shifted_pass"],
                               "baseline_result": row["baseline_result"],
                               "shifted_result": row["shifted_result"],
                           } for i, row in enumerate(rows)],
                           "by_cohort": {
                               cohort: summary([row for row in rows
                                                if row["cohort"] == cohort],
                                               [selected[i] for i, row in enumerate(rows)
                                                if row["cohort"] == cohort])
                               for cohort in sorted({row["cohort"] for row in rows})}})
    report = {
        "schema": "robot-learning-lab.cove50-v132-grouped-recipe-audit.v1",
        "role": "opened_training_group_crossfit_not_independent_contact_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {"dataset": sha(DATASET), "reserve": sha(RESERVE),
                          "audit": sha(Path(__file__))},
        "method": ("Compare four predeclared weighting schemes and two gain "
                   "thresholds for the same 200-tree, two-leaf selector. "
                   "Withhold every row of a seed group together, including "
                   "the V130 physical perturbations of three opened seeds. "
                   "Failure-emphasis schemes give baseline-failed pairs 5× "
                   "weight; equal-seed schemes divide a seed's total weight "
                   "across its physical variations. "
                   "Abstain outside the remaining observed feature bounds. "
                   "Neither this group crossfit nor recipe selection uses "
                   "V128 contact labels; no candidate is promoted by it."),
        "pair_count": len(rows), "seed_group_count": len(seed_counts),
        "audits": audits,
        "cases": [{"id": f"{row['sample_weight']}@{row['selection_threshold_s']:g}s",
                   **{key: value for key, value in row.items()
                      if key not in ("by_cohort", "held_decisions")}}
                  for row in audits],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2) + "\n"
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Prior grouped recipe audit changed")
    OUT.write_text(encoded)
    print(f"{OUT} sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
