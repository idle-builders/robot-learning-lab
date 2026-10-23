"""Frozen V465 policy on the original late, previously unseen crossing.

The object looks stationary until t=14.0 s, then crosses during an action.
This is an explicit falsification challenge for velocity-only path forecasts;
the actor receives no simulator future trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_layout4_crossing_v447 import layout4_with_prop
from run_robocasa_layout4_directional_wait_v465 import grid_class
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("robocasa_layout4_crossing_v447.json")
PRIOR = ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-layout4-late-policy-v469"


def main():
    case = next(c for c in json.loads(CONFIG.read_text())["cases"]
                if c["id"] == "late_crossing")
    prior = np.load(PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    trial.model_with_layout2 = layout4_with_prop
    trial.planar_scan = height_scan((.05, .30))
    trial.obstacle_xy = motion(case)
    trial.START_X, trial.OBSTACLE_Y = case["from_xy_m"]
    trial.SensorGrid = grid_class(prior)
    trial.OUT = OUT
    route.MARGIN_M = .15
    episode = trial.episode(True)
    trial.SensorGrid = route.SensorGrid
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "policy_source_sha256": route.sha256(
                  Path(__file__).with_name("run_robocasa_layout4_directional_wait_v465.py")),
              "config_sha256": route.sha256(CONFIG),
              "prior_map_sha256": route.sha256(PRIOR),
              "scope": "frozen velocity/path policy on unchanged original late crossing, idealized guard and kinematic proxy",
              "cases": [{"scenario": case, "package": "directional_wait_v465",
                         "episode": episode}]}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(episode["outcome"], episode["steps"],
          round(episode["end_time_s"], 2),
          round(episode["min_true_prop_gap_m"], 3), flush=True)


if __name__ == "__main__":
    main()
