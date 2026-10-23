"""Freeze a small RGB-entry selector using only pre-V113 paired native visits."""

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
DATASET = BASE / "v118/observable-pairs-partial.json"
OUT = BASE / "v118"
MODEL = OUT / "observable-selector.json"
DEVELOPMENT = OUT / "first-eight-opened-development.json"
HOLDOUT = BASE / "v116/reserved-contact-validation-plan.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def vector(features: dict, modes: list[str]) -> list[float]:
    return ([float(features[key]) for key in selector.NUMERIC_FEATURES] +
            [float(features["selection_mode"] == mode) for mode in modes])


def main() -> None:
    source = json.loads(DATASET.read_text())
    if source["schema"] != "robot-learning-lab.cove50-v118-observable-pairs.v1":
        raise ValueError("Unexpected paired native dataset")
    training = [row for row in source["pairs"]
                if not row["cohort"].startswith("V113")]
    development = [row for row in source["pairs"]
                   if row["cohort"].startswith("V113")]
    if len(training) != 41 or len(development) < 8:
        raise ValueError("Expected 41 older pairs and eight opened V113 development pairs")
    reserved = {row["seed"] for row in json.loads(HOLDOUT.read_text())["cases"]}
    if {row["seed_audit_only"] for row in training + development} & reserved:
        raise ValueError("Selector data touches the sealed contact holdout")
    modes = sorted({row["features"]["selection_mode"] for row in training})
    sources = sorted({row["features"]["route_goal_source"] for row in training})
    fitted = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                   random_state=4, n_jobs=1)
    fitted.fit([vector(row["features"], modes) for row in training],
               [max(-2.0, min(2.0, row["contact_delta_s"]))
                for row in training])
    trees = []
    for estimate in fitted.estimators_:
        tree = estimate.tree_
        trees.append({"left": tree.children_left.tolist(),
                      "right": tree.children_right.tolist(),
                      "feature": tree.feature.tolist(),
                      "threshold": tree.threshold.tolist(),
                      "value": tree.value[:, 0, 0].tolist()})
    model = {
        "schema": "robot-learning-lab.cove50-v118-observable-selector.v1",
        "role": "frozen_development_candidate_not_qualified",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "inference_source_sha256": sha(Path(selector.__file__)),
        "fitter_source_sha256": sha(Path(__file__)),
        "training_pair_count": len(training),
        "training_pair_sha256": digest(training),
        "training_cohorts": sorted({row["cohort"] for row in training}),
        "held_out_from_fit": "All V113 paired starts and sealed V116 contact starts",
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
                           "gain exceeds 0.4 s and every observed feature is "
                           "inside the training range; otherwise use unchanged."),
        "trees": trees,
    }
    encoded = (json.dumps(model, separators=(",", ":")) + "\n").encode()
    if MODEL.exists() and MODEL.read_bytes() != encoded:
        raise ValueError("Frozen V118 candidate or its sources changed")
    MODEL.write_bytes(encoded)
    rows = []
    for row in development[:8]:
        output = selector.predict(model, row["features"])
        rows.append({"seed": row["seed_audit_only"],
                     "decision": output,
                     "observed_contact_delta_s": row["contact_delta_s"],
                     "baseline_contact_s": row["baseline_contact_s"],
                     "shifted_contact_s": row["shifted_contact_s"],
                     "baseline_margin_1p2_s_met": row["baseline_margin_1p2_s_met"],
                     "shifted_margin_1p2_s_met": row["shifted_margin_1p2_s_met"]})
    result = {"schema": "robot-learning-lab.cove50-v118-first-eight-development.v1",
              "role": "opened_development_diagnostic_not_holdout_score",
              "model_sha256": sha(MODEL), "case_count": len(rows),
              "selected_count": sum(row["decision"]["shift_body_y_m"] < 0
                                    for row in rows),
              "selected_contact_gain_s": round(sum(row["observed_contact_delta_s"]
                                                    for row in rows
                                                    if row["decision"]["shift_body_y_m"] < 0), 3),
              "selected_margin_loss_count": sum(
                  row["baseline_margin_1p2_s_met"] and
                  not row["shifted_margin_1p2_s_met"] for row in rows
                  if row["decision"]["shift_body_y_m"] < 0),
              "cases": rows}
    DEVELOPMENT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"frozen {MODEL} sha256={sha(MODEL)}; opened first-eight development "
          f"selected={result['selected_count']}/8 contact_gain_s="
          f"{result['selected_contact_gain_s']}")


if __name__ == "__main__":
    main()
