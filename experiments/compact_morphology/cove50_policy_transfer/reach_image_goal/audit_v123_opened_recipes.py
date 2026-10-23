"""Compare small contact-selector recipes on opened training pairs only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    v118_observable_selector as selector,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
DATASET = BASE / "v118/observable-pairs.json"
OUT = BASE / "v123/opened-recipe-comparison.json"
THRESHOLDS = (0.0, 0.2, 0.4)
RECIPES = (
    ("random_forest_leaf_1", RandomForestRegressor, 1),
    ("random_forest_leaf_2", RandomForestRegressor, 2),
    ("extra_trees_leaf_1", ExtraTreesRegressor, 1),
    ("extra_trees_leaf_2", ExtraTreesRegressor, 2),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = json.loads(DATASET.read_text())
    rows = data["pairs"]
    if (not data["complete"] or len(rows) != 65 or
            len({row["seed_audit_only"] for row in rows}) != 65):
        raise ValueError("The 65 independent opened training pairs are required")
    modes = sorted({row["features"]["selection_mode"] for row in rows})
    values = np.array([
        [float(row["features"][key]) for key in selector.NUMERIC_FEATURES] +
        [float(row["features"]["selection_mode"] == mode) for mode in modes]
        for row in rows], dtype=float)
    clipped_gain = np.array([max(-2.0, min(2.0, row["contact_delta_s"]))
                             for row in rows])
    results = []
    for name, recipe, leaf_size in RECIPES:
        predictions = []
        inside = []
        for index, held in enumerate(rows):
            train = [j for j in range(len(rows)) if j != index]
            fit = recipe(n_estimators=200, min_samples_leaf=leaf_size,
                         random_state=4, n_jobs=1)
            fit.fit(values[train], clipped_gain[train])
            predictions.append(float(fit.predict(values[index:index + 1])[0]))
            features = held["features"]
            inside.append(
                features["selection_mode"] in
                {rows[j]["features"]["selection_mode"] for j in train} and
                features["route_goal_source"] in
                {rows[j]["features"]["route_goal_source"] for j in train} and
                all(np.min(values[train, k]) <= values[index, k] <=
                    np.max(values[train, k])
                    for k in range(len(selector.NUMERIC_FEATURES))))
        for threshold in THRESHOLDS:
            chosen = [row for index, row in enumerate(rows)
                      if inside[index] and predictions[index] > threshold]
            results.append({
                "recipe": name, "threshold_s": threshold,
                "selected_count": len(chosen),
                "rescues": sum(not row["baseline_pass"] and row["shifted_pass"]
                               for row in chosen),
                "full_gate_regressions": sum(row["baseline_pass"] and
                                             not row["shifted_pass"]
                                             for row in chosen),
                "contact_margin_losses": sum(
                    row["baseline_margin_1p2_s_met"] and
                    not row["shifted_margin_1p2_s_met"] for row in chosen),
                "selected_contact_change_s": round(sum(
                    row["contact_delta_s"] for row in chosen), 3),
                "selected_baseline_failure_seeds": [
                    row["seed_audit_only"] for row in chosen
                    if not row["baseline_pass"]],
            })
    report = {
        "schema": "robot-learning-lab.cove50-v123-opened-recipe-audit.v1",
        "role": "opened_training_crossfit_only_not_frozen_contact_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {"dataset": sha(DATASET), "audit": sha(Path(__file__))},
        "method": ("Leave one of 65 distinct opened training seeds out, fit on "
                   "the other 64, preserve categorical and numeric in-range "
                   "abstention, and compare four prelisted tree recipes at "
                   "three thresholds. This uses no V116/V119/V123 or V128 "
                   "sealed contact label. Recipe search on opened data is "
                   "diagnostic and cannot establish generalization."),
        "opened_baseline_failure_count": sum(not row["baseline_pass"]
                                             for row in rows),
        "opened_possible_rescue_count": sum(not row["baseline_pass"] and
                                            row["shifted_pass"] for row in rows),
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2) + "\n"
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Prior opened recipe diagnostic changed")
    OUT.write_text(encoded)
    print(f"{OUT} sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
