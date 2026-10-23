"""Fit the unchanged observable selector recipe on all 65 varied pairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    v118_observable_selector as selector,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
DATASET = BASE / "v118/observable-pairs.json"
RESERVED = BASE / "v123/remaining-entry-contact-holdout-plan.json"
FIRST_HOLDOUT = BASE / "v116/reserved-contact-validation-plan.json"
OUT = BASE / "v123"
MODEL = OUT / "observable-selector.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    source = json.loads(DATASET.read_text())
    reserved = json.loads(RESERVED.read_text())
    prior = json.loads(FIRST_HOLDOUT.read_text())
    training = source["pairs"]
    if (source["schema"] != "robot-learning-lab.cove50-v118-observable-pairs.v1" or
            not source["complete"] or len(training) != 65 or
            reserved["contact_status"] != "not_run" or
            len(reserved["cases"]) != 14):
        raise ValueError("Complete 65-pair training data and reserved entries required")
    seeds = {row["seed_audit_only"] for row in training}
    if seeds & ({case["seed"] for case in reserved["cases"]} |
                {case["seed"] for case in prior["cases"]}):
        raise ValueError("Training contact pairs overlap reserved entries")
    modes = sorted({row["features"]["selection_mode"] for row in training})
    sources = sorted({row["features"]["route_goal_source"] for row in training})
    vector = lambda features: ([float(features[key]) for key in selector.NUMERIC_FEATURES] +
                               [float(features["selection_mode"] == mode)
                                for mode in modes])
    fit = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                random_state=4, n_jobs=1)
    fit.fit([vector(row["features"]) for row in training],
            [max(-2.0, min(2.0, row["contact_delta_s"])) for row in training])
    trees = []
    for estimate in fit.estimators_:
        tree = estimate.tree_
        trees.append({"left": tree.children_left.tolist(),
                      "right": tree.children_right.tolist(),
                      "feature": tree.feature.tolist(),
                      "threshold": tree.threshold.tolist(),
                      "value": tree.value[:, 0, 0].tolist()})
    model = {
        "schema": "robot-learning-lab.cove50-v118-observable-selector.v1",
        "model_id": "v123_complete_varied_pairs",
        "role": "frozen_development_candidate_not_qualified",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "inference_source_sha256": sha(Path(selector.__file__)),
        "fitter_source_sha256": sha(Path(__file__)),
        "training_dataset_sha256": sha(DATASET),
        "training_pair_count": len(training),
        "training_pair_sha256": digest(training),
        "training_cohorts": sorted({row["cohort"] for row in training}),
        "reserved_entry_plan_sha256": sha(RESERVED),
        "first_holdout_plan_sha256": sha(FIRST_HOLDOUT),
        "held_out_from_fit": ("All V116/V119 first-holdout and V123 remaining-entry "
                              "starts; neither contact matrix is an input"),
        "feature_contract": source["feature_contract"],
        "numeric_features": list(selector.NUMERIC_FEATURES),
        "selection_modes": modes,
        "route_goal_sources": sources,
        "training_bounds": {key: [min(row["features"][key] for row in training),
                                  max(row["features"][key] for row in training)]
                            for key in selector.NUMERIC_FEATURES},
        "target": "Contact difference between −2 mm and unchanged, clipped to ±2 s",
        "fit": {"algorithm": "RandomForestRegressor", "n_estimators": 200,
                "min_samples_leaf": 2, "random_state": 4},
        "selection_threshold_s": .4,
        "selection_rule": ("Select −2 mm only when predicted clipped contact "
                           "gain exceeds the unchanged 0.4 s threshold and every "
                           "observed feature is inside training bounds; otherwise "
                           "use unchanged."),
        "trees": trees,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(model, separators=(",", ":")) + "\n").encode()
    if MODEL.exists() and MODEL.read_bytes() != encoded:
        raise ValueError("Frozen V123 candidate or source changed")
    MODEL.write_bytes(encoded)
    print(f"frozen {MODEL} sha256={sha(MODEL)} pairs={len(training)}")


if __name__ == "__main__":
    main()
