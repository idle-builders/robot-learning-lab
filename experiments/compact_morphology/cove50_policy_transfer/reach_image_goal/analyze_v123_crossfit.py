"""Audit V123 recipe by withholding each training seed in turn."""

from __future__ import annotations

from collections import defaultdict
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
MODEL = BASE / "v123/observable-selector.json"
OUT = BASE / "v123/leave-one-seed-out-diagnostic.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source = json.loads(DATASET.read_text())
    model = json.loads(MODEL.read_text())
    pairs = source["pairs"]
    if (model["training_dataset_sha256"] != sha(DATASET) or
            len(pairs) != 65 or len({row["seed_audit_only"] for row in pairs}) != 65):
        raise ValueError("V123 crossfit requires 65 independent training seeds")
    rows = []
    for held in pairs:
        training = [row for row in pairs if row["seed_audit_only"] !=
                    held["seed_audit_only"]]
        modes = sorted({row["features"]["selection_mode"] for row in training})
        sources = sorted({row["features"]["route_goal_source"] for row in training})
        vector = lambda features: ([float(features[key])
                                    for key in selector.NUMERIC_FEATURES] +
                                   [float(features["selection_mode"] == mode)
                                    for mode in modes])
        fit = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                    random_state=4, n_jobs=1)
        fit.fit([vector(row["features"]) for row in training],
                [max(-2.0, min(2.0, row["contact_delta_s"]))
                 for row in training])
        features = held["features"]
        outside = []
        if features["selection_mode"] not in modes:
            outside.append("selection_mode")
        if features["route_goal_source"] not in sources:
            outside.append("route_goal_source")
        for key in selector.NUMERIC_FEATURES:
            lo = min(row["features"][key] for row in training)
            hi = max(row["features"][key] for row in training)
            if not lo <= features[key] <= hi:
                outside.append(key)
        predicted = float(fit.predict([vector(features)])[0])
        selected = not outside and predicted > model["selection_threshold_s"]
        rows.append({"seed": held["seed_audit_only"], "cohort": held["cohort"],
                     "selection_mode": features["selection_mode"],
                     "predicted_clipped_gain_s": round(predicted, 4),
                     "abstention_reasons": outside,
                     "selected_shift": selected,
                     "baseline_contact_s": held["baseline_contact_s"],
                     "shifted_contact_s": held["shifted_contact_s"],
                     "observed_contact_delta_s": held["contact_delta_s"],
                     "baseline_pass": held["baseline_pass"],
                     "shifted_pass": held["shifted_pass"],
                     "baseline_margin_1p2_s_met": held["baseline_margin_1p2_s_met"],
                     "shifted_margin_1p2_s_met": held["shifted_margin_1p2_s_met"]})
    groups = defaultdict(list)
    for row in rows:
        groups[row["cohort"]].append(row)

    def summary(items: list[dict]) -> dict:
        selected = [row for row in items if row["selected_shift"]]
        return {"case_count": len(items), "selected_count": len(selected),
                "rescues": sum(not row["baseline_pass"] and row["shifted_pass"]
                               for row in selected),
                "full_gate_regressions": sum(row["baseline_pass"] and
                                             not row["shifted_pass"]
                                             for row in selected),
                "margin_regressions": sum(row["baseline_margin_1p2_s_met"] and
                                          not row["shifted_margin_1p2_s_met"]
                                          for row in selected),
                "selected_contact_change_s": round(sum(
                    row["observed_contact_delta_s"] for row in selected), 3),
                "selected_still_failing": [row["seed"] for row in selected
                                            if not row["shifted_pass"]]}

    result = {"schema": "robot-learning-lab.cove50-v123-crossfit-diagnostic.v1",
              "role": "leave_one_training_seed_out_audit_not_contact_holdout_or_score",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "dataset_sha256": sha(DATASET), "model_sha256": sha(MODEL),
              "source_sha256": sha(Path(__file__)),
              "method": ("For each of 65 distinct training seeds, refit the same "
                         "200-tree, two-leaf recipe on the other 64 and apply the "
                         "unchanged 0.4 s threshold and in-range abstention. "
                         "No reserved V116/V119/V123 contact label is used. "
                         "This probes interpolation within opened data, not "
                         "future-scene or new-body performance."),
              "summary": summary(rows),
              "by_cohort": {name: summary(group) for name, group in groups.items()},
              "cases": rows}
    encoded = json.dumps(result, indent=2) + "\n"
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Frozen crossfit diagnostic or source changed")
    OUT.write_text(encoded)
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
