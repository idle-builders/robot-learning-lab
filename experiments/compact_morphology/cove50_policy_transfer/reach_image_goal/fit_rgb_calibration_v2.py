"""Fit bounded RGB correction with broadened opened historical poses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.rgb_calibration_model import design
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.rgb_goal_features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/calibration-v2"
LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1250.0,
           1350.0, 1400.0, 1450.0, 1500.0, 2000.0)
MAX_CORRECTION_MM = 10.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def errors_mm(y: np.ndarray, predicted: np.ndarray) -> dict:
    magnitude = np.linalg.norm(y - predicted, axis=1)
    return {"median": round(float(np.median(magnitude)), 4),
            "p95": round(float(np.percentile(magnitude, 95)), 4),
            "max": round(float(np.max(magnitude)), 4)}


def bounded(predicted: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(predicted, axis=1)
    return predicted * np.minimum(1.0, MAX_CORRECTION_MM / np.maximum(norm, 1e-9))[:, None]


def main() -> None:
    data_path = OUT / "rgb-calibration-data.npz"
    with np.load(data_path) as saved:
        x = saved["x"].astype(np.float64)
        y = saved["y_mm"].astype(np.float64)
        split = saved["split"]
        seeds = saved["seed"]
        names = saved["feature_names"].tolist()
    if names != FEATURE_NAMES or x.shape != (len(y), len(names)) or y.shape[1] != 2:
        raise ValueError("Unexpected RGB calibration dataset contract")
    train = split == "train"
    validation = split == "validation"
    if train.sum() < 2000 or validation.sum() < 300 or set(seeds[train]) & set(seeds[validation]):
        raise ValueError("Insufficient or overlapping calibration split")
    mean = x[train].mean(axis=0)
    scale = np.maximum(x[train].std(axis=0), 1e-5)
    train_x = design(x[train], mean, scale)
    valid_x = design(x[validation], mean, scale)
    gram = train_x.T @ train_x
    cross = train_x.T @ y[train]
    trials = []
    coefficients = {}
    for strength in LAMBDAS:
        regularizer = np.eye(gram.shape[0]) * strength
        regularizer[0, 0] = 0
        weight = np.linalg.solve(gram + regularizer, cross)
        coefficients[strength] = weight
        predicted = bounded(valid_x @ weight)
        trials.append({"ridge_lambda": strength,
                       "train_error_mm": errors_mm(y[train], bounded(train_x @ weight)),
                       "validation_error_mm": errors_mm(y[validation], predicted)})
    chosen = min(trials, key=lambda trial: (
        trial["validation_error_mm"]["max"],
        trial["validation_error_mm"]["median"]))
    coarse = errors_mm(y[validation], np.zeros_like(y[validation]))
    corrected = chosen["validation_error_mm"]
    passed = (corrected["median"] <= 1.5 and corrected["max"] <= 3.0 and
              corrected["median"] < coarse["median"] and corrected["max"] < coarse["max"])
    model_path = OUT / "rgb-calibration-model.npz"
    np.savez_compressed(model_path, mean=mean, scale=scale,
                        coefficients=coefficients[chosen["ridge_lambda"]],
                        max_correction_mm=MAX_CORRECTION_MM,
                        ridge_lambda=chosen["ridge_lambda"],
                        feature_names=np.asarray(FEATURE_NAMES))
    report = {"schema": "robot-learning-lab.cove50-rgb-calibration-v2-fit.v1",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "data": str(data_path.relative_to(ROOT)), "data_sha256": sha(data_path),
              "feature_source_sha256": sha(Path(__file__).with_name("rgb_goal_features.py")),
              "model_source_sha256": sha(Path(__file__).with_name("rgb_calibration_model.py")),
              "fit_source_sha256": sha(Path(__file__)),
              "train_seed_count": len(set(seeds[train])),
              "validation_seed_count": len(set(seeds[validation])),
              "coarse_validation_error_mm": coarse,
              "trials": trials, "selected_lambda": chosen["ridge_lambda"],
              "corrected_validation_error_mm": corrected,
              "offline_gate_pass": passed,
              "model": str(model_path.relative_to(ROOT)), "model_sha256": sha(model_path),
              "limit": "Synthetic orange cube and exact capture pose on rejected Direct Frame; no physical camera or new-body qualification."}
    (OUT / "rgb-calibration-fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"offline_gate_pass": passed, "coarse": coarse,
                      "corrected": corrected, "lambda": chosen["ridge_lambda"]}))


if __name__ == "__main__":
    main()
