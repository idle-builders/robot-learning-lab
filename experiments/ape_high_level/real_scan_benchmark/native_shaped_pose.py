"""Transfer historical 50 cm joint/contact odometry error to a room step.

This is a simulator-side packet generator. The actor receives only the
resulting measured increment and health, never the historical root trajectory
or the room's evaluator pose. It is not a physical sensor qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT / "artifacts/ape-sensors-50cm/native-proprioceptive-odometry-v1"
SOURCES = (
    "planar200v3-forward",
    "varied150-forward-seed144",
    "robust10v200-forward-seed144",
)
ESTIMATE = "combined_illustrative_impairments"


class NativeShapedPose:
    def __init__(self, source: str) -> None:
        if source not in SOURCES:
            raise ValueError(f"Unknown native odometry source: {source}")
        self.path = SOURCE_DIR / f"{source}-trace.json"
        rows = json.loads(self.path.read_text())
        self.source = source
        self.source_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.truth = np.asarray([row["actual_xy_m_evaluator_only"]
                                 for row in rows], dtype=float)
        self.sensor = np.asarray([row["estimated_xy_m"][ESTIMATE]
                                  for row in rows], dtype=float)
        self.contacts = np.asarray([any(row["contact_found_idealized"])
                                    for row in rows], dtype=bool)
        self.arc = np.r_[0., np.cumsum(np.linalg.norm(
            np.diff(self.truth, axis=0), axis=1))]
        self.cursor = 0

    def advance(self, commanded_delta_world_m: np.ndarray) -> dict:
        command = np.asarray(commanded_delta_world_m, dtype=float)
        distance = float(np.linalg.norm(command))
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("Native-shaped packet needs an accepted translation")
        start = self.cursor
        end = int(np.searchsorted(self.arc, self.arc[start] + distance,
                                  side="left"))
        if end >= len(self.arc):
            return {"accepted": False, "reason": "native_source_exhausted",
                    "source": self.source, "source_start_frame": start}
        source_true = self.truth[end] - self.truth[start]
        source_measured = self.sensor[end] - self.sensor[start]
        traveled = float(self.arc[end] - self.arc[start])
        if traveled <= 0 or float(np.linalg.norm(source_true)) < 1e-5:
            return {"accepted": False, "reason": "native_source_no_net_step",
                    "source": self.source, "source_start_frame": start,
                    "source_end_frame": end}
        source_error = source_measured - source_true
        rotation = (math.atan2(command[1], command[0]) -
                    math.atan2(source_true[1], source_true[0]))
        c, s = math.cos(rotation), math.sin(rotation)
        aligned_error = np.array([[c, -s], [s, c]]) @ source_error
        measured = command + aligned_error * (distance / traveled)
        contact_samples = int(self.contacts[start + 1:end + 1].sum())
        healthy = bool(contact_samples > 0 and np.all(np.isfinite(measured)) and
                       np.linalg.norm(measured) <= .12)
        self.cursor = end
        return {"accepted": healthy,
                "reason": "contact_gated_joint_imu_increment" if healthy else
                          "native_contact_or_increment_unavailable",
                "source": self.source,
                "source_start_frame": start, "source_end_frame": end,
                "contact_samples": contact_samples,
                "measured_delta_world_m": measured.tolist(),
                "commanded_step_m": distance,
                "native_distance_consumed_m": traveled}
