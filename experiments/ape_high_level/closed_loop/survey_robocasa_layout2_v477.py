"""Save a sensor-derived layout-2 free/occupied grid before prop entry.

The survey reuses the range-localized actor, its commanded motion and noisy
dual-height scans. No simulator fixture geometry or true pose is written into
the map used by later actor decisions. True pose remains in the run's evaluator
trace only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_localized_route_v408 as actor
import run_robocasa_layout2_route_v407 as route
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout2-survey-v477"
CAPTURED = []


class CapturingGrid(route.SensorGrid):
    def __init__(self):
        super().__init__()
        CAPTURED.append(self)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actor.model_with_layout2 = route.model_with_layout2
    actor.planar_scan = height_scan((.05, .30))
    actor.SensorGrid = CapturingGrid
    actor.OUT = OUT
    route.MARGIN_M = .15
    episode = actor.episode(5., .30, "range_growing")
    grid = CAPTURED[-1]
    map_path = OUT / "sensor-derived-map.npz"
    np.savez_compressed(map_path, xs=grid.xs, ys=grid.ys,
                        free=grid.free, occupied=grid.occupied)
    diagnostic_cell = grid.cell(np.array([4.95, -3.34]))
    report = {"source_sha256": route.sha256(Path(__file__)),
              "frozen_actor_source_sha256": route.sha256(Path(actor.__file__)),
              "scene_source_sha256": route.sha256(Path(route.__file__)),
              "height_adapter_source_sha256": route.sha256(
                  Path(__file__).with_name("run_robocasa_scan_height_v433.py")),
              "map": {"path": str(map_path), "sha256": route.sha256(map_path),
                      "known_free_cells": int(grid.free.sum()),
                      "occupied_cells": int(grid.occupied.sum()),
                      "diagnostic_later_prop_cell_known_free": bool(grid.free[diagnostic_cell])},
              "episode": episode}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(episode["outcome"], episode["steps"],
          round(episode["end_pose_error_m"], 3),
          "prop_cell_free", bool(grid.free[diagnostic_cell]), flush=True)


if __name__ == "__main__":
    main()
