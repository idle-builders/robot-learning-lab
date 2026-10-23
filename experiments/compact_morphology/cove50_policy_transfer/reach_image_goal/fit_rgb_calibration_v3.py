"""Benchmark observable RGB correction fits on state-disjoint native views."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.rgb_calibration_model import RgbCalibration
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.rgb_goal_features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "calibration-v3"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    errors = np.linalg.norm(y - prediction, axis=1)
    return {"median_mm": round(float(np.median(errors)), 4),
            "p95_mm": round(float(np.percentile(errors, 95)), 4),
            "max_mm": round(float(np.max(errors)), 4)}


def main() -> None:
    plan_path = OUT / "plan.json"
    data_path = OUT / "rgb-calibration-data.npz"
    report_path = OUT / "data-report.json"
    plan = json.loads(plan_path.read_text())
    data_report = json.loads(report_path.read_text())
    if data_report["plan_sha256"] != sha(plan_path) or data_report["data_sha256"] != sha(data_path):
        raise ValueError("Calibration data changed after collection")
    with np.load(data_path) as saved:
        x = saved["x"].astype(np.float64)
        y = saved["y_mm"].astype(np.float64)
        split, seed, sample = (saved[key] for key in ("split", "seed", "sample"))
        if saved["feature_names"].tolist() != FEATURE_NAMES:
            raise ValueError("RGB feature contract changed")
    train, validation = split == "train", split == "validation"
    if set(seed[train]) & set(seed[validation]):
        raise ValueError("Training and validation source states overlap")
    old_model_path = BASE / "rgb-calibration-model.npz"
    old_model = RgbCalibration(old_model_path)
    old_prediction = np.asarray([old_model.correction_mm(row) for row in x[validation]])
    trials = []

    def add(name: str, predict) -> None:
        validation_prediction = predict(x[validation])
        correction_norm = np.linalg.norm(validation_prediction, axis=1)
        validation_prediction *= np.minimum(1, 10 / np.maximum(correction_norm, 1e-9))[:, None]
        training_index = np.flatnonzero((seed == 10604) & (sample == 0))[0]
        target_prediction = predict(x[training_index:training_index + 1])[0]
        old_views = validation & (seed < 10600)
        new_views = validation & (seed >= 10600)
        partition = {"all": metrics(y[validation], validation_prediction),
                     "historical": metrics(y[old_views], validation_prediction[seed[validation] < 10600]),
                     "opened_v21": metrics(y[new_views], validation_prediction[seed[validation] >= 10600])}
        trials.append({"name": name, "validation_error": partition,
                       "opened_10604_nominal_correction_target_mm": np.round(y[training_index], 4).tolist(),
                       "opened_10604_nominal_prediction_mm": np.round(target_prediction, 4).tolist(),
                       "opened_10604_nominal_error_mm": round(float(np.linalg.norm(y[training_index] - target_prediction)), 4)})
        print(name, partition["all"], partition["opened_v21"], flush=True)

    for leaf in (2, 5, 10):
        model = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=leaf,
                                    max_features=0.9, n_jobs=4, random_state=42).fit(x[train], y[train])
        add(f"extra_trees_leaf_{leaf}", model.predict)
    for leaf in (5, 15):
        models = [HistGradientBoostingRegressor(max_iter=200, min_samples_leaf=leaf,
                                                 l2_regularization=1, random_state=42).fit(x[train], y[train, i])
                  for i in range(2)]
        add(f"hist_gradient_leaf_{leaf}", lambda rows, models=models: np.stack(
            [model.predict(rows) for model in models], axis=1))
    for alpha in (1, 10):
        model = make_pipeline(StandardScaler(),
                              MLPRegressor(hidden_layer_sizes=(64, 64), alpha=alpha,
                                           max_iter=500, early_stopping=True, random_state=42)).fit(x[train], y[train])
        add(f"mlp_alpha_{alpha}", model.predict)
    passed = [trial for trial in trials if
              trial["validation_error"]["all"]["median_mm"] <= 1.5 and
              trial["validation_error"]["all"]["max_mm"] <= 3.0 and
              trial["opened_10604_nominal_error_mm"] <= 2.0]
    result = {"schema": "robot-learning-lab.cove50-rgb-calibration-v3-fit.v1",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "plan_sha256": sha(plan_path), "data_sha256": sha(data_path),
              "fit_source_sha256": sha(Path(__file__)),
              "old_model_sha256": sha(old_model_path),
              "train_images": int(train.sum()), "validation_images": int(validation.sum()),
              "coarse_validation_error": metrics(y[validation], np.zeros_like(y[validation])),
              "old_model_validation_error": metrics(y[validation], old_prediction),
              "trials": trials,
              "offline_gate_pass": bool(passed),
              "selected_model": min(passed, key=lambda trial: trial["validation_error"]["all"]["max_mm"])["name"]
                                 if passed else None,
              "qualification": "Opened V21 views are development data. A failed offline fit is not promoted to native control."}
    (OUT / "fit-report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"offline_gate_pass": result["offline_gate_pass"],
                      "selected_model": result["selected_model"]}))


if __name__ == "__main__":
    main()
