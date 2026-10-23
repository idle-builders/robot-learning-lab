"""Small frozen regression adapter for the synthetic RGB goal diagnostic."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.rgb_goal_features import (
    FEATURE_NAMES,
)


def design(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    z = np.clip((x - mean) / scale, -5, 5)
    terms = [np.ones((len(z), 1)), z]
    terms.extend((z[:, i:i + 1] * z[:, j:j + 1])
                 for i in range(z.shape[1]) for j in range(i, z.shape[1]))
    return np.concatenate(terms, axis=1)


class RgbCalibration:
    def __init__(self, path: Path) -> None:
        with np.load(path) as saved:
            names = saved["feature_names"].tolist()
            if names != FEATURE_NAMES:
                raise ValueError("RGB calibration feature contract changed")
            self.mean = saved["mean"].astype(np.float64)
            self.scale = saved["scale"].astype(np.float64)
            self.coefficients = saved["coefficients"].astype(np.float64)
            self.max_correction_mm = float(saved["max_correction_mm"])
        if self.coefficients.shape != (design(self.mean, self.mean, self.scale).shape[1], 2):
            raise ValueError("RGB calibration coefficient shape changed")

    def correction_mm(self, row: np.ndarray) -> np.ndarray:
        correction = (design(row, self.mean, self.scale) @ self.coefficients)[0]
        if not np.isfinite(correction).all():
            raise ValueError("Nonfinite RGB calibration prediction")
        magnitude = float(np.linalg.norm(correction))
        if magnitude > self.max_correction_mm:
            correction *= self.max_correction_mm / magnitude
        return correction
