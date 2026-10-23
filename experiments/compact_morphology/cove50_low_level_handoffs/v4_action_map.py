"""Exact source-derived V4 physical joint target map for offline labels.

The saved native traces are checked against this map before use. The live
evaluator separately checks these values against the instantiated action term.
"""

from __future__ import annotations

import numpy as np


SCALE = {
    **{f"{side}_{joint}": value for side in ("left", "right")
       for joint, value in (("hip_yaw", 1.0), ("hip_roll", 1.0),
                            ("hip_pitch", 1.5), ("knee", 1.6),
                            ("ankle_pitch", 1.0), ("shoulder_pitch", 1.2))},
}
OFFSET = {
    **{f"{side}_{joint}": value for side in ("left", "right")
       for joint, value in (("hip_pitch", -0.2), ("knee", 0.4),
                            ("ankle_pitch", -0.2))},
}
SHOULDER_CLIP = (-1.52, 1.52)


def arrays(names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = np.asarray([SCALE.get(name, 0.0) for name in names], dtype=np.float32)
    offset = np.asarray([OFFSET.get(name, 0.0) for name in names], dtype=np.float32)
    lower = np.asarray([SHOULDER_CLIP[0] if name.endswith("shoulder_pitch") else -np.inf
                        for name in names], dtype=np.float32)
    upper = np.asarray([SHOULDER_CLIP[1] if name.endswith("shoulder_pitch") else np.inf
                        for name in names], dtype=np.float32)
    return scale, offset, lower, upper


def processed(raw: np.ndarray, names: list[str]) -> np.ndarray:
    scale, offset, lower, upper = arrays(names)
    return np.clip(raw * scale + offset, lower, upper)


def canonical_raw(target: np.ndarray, names: list[str]) -> np.ndarray:
    scale, offset, lower, upper = arrays(names)
    effective = np.clip(target, lower, upper)
    return np.where(scale != 0, (effective - offset) / np.where(scale != 0, scale, 1), 0)
