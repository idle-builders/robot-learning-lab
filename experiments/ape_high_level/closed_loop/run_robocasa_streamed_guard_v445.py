"""Approximate a 10 Hz rotating lidar stream as five 20 ms angular sectors.

Each guard tick receives 100 rays over one 72-degree sector (5000 samples/s,
500 rays/revolution). This is still an ideal simultaneous-sector, zero-latency
measurement: it omits real streaming timestamps, beam divergence, reflectance,
packet transport and braking. The 0.4 s planner scan remains unchanged.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-streamed-guard-v445"
RAYS_PER_SECTOR = 100
SECTORS_PER_REVOLUTION = 5


def sector_guard(phase: int):
    tick = 0

    def measure(model, data, pose):
        nonlocal tick
        sector = (phase + tick) % SECTORS_PER_REVOLUTION
        tick += 1
        span = 2. * math.pi / SECTORS_PER_REVOLUTION
        angles = (sector * span +
                  np.arange(RAYS_PER_SECTOR, dtype=float) * span / RAYS_PER_SECTOR)
        origin = np.array([pose[0], pose[1], .30], dtype=float)
        nearest = math.inf
        for angle in angles:
            direction = np.array([math.cos(angle), math.sin(angle), 0.], dtype=float)
            geom_id = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(model, data, origin, direction,
                                     None, 1, -1, geom_id)
            if distance > 0:
                nearest = min(nearest, float(distance))
        return nearest

    return measure


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for phase in range(SECTORS_PER_REVOLUTION):
        trial.raw_nearest_range = sector_guard(phase)
        trial.OUT = OUT / f"phase{phase}"
        episode = trial.episode(True)
        rows.append({"phase_sector": phase, "episode": episode})
        print(phase, episode["outcome"], episode["steps"],
              round(episode["min_true_prop_gap_m"], 4),
              round(episode["end_time_s"], 3), flush=True)
    report = {"source_sha256": route.sha256(Path(__file__)),
              "frozen_trial_source_sha256": route.sha256(Path(trial.__file__)),
              "scope": "original east-pinch route, 10Hz ideal rotating-sector guard, five 20ms phase offsets, zero latency/braking",
              "sectors_per_revolution": SECTORS_PER_REVOLUTION,
              "sector_interval_s": trial.SUBSTEP_S,
              "rays_per_sector": RAYS_PER_SECTOR,
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
