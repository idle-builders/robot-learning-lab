"""Paired frozen-route and V461 wait policy on a directionally safe mover."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from run_robocasa_layout4_crossing_v447 import layout4_with_prop
from run_robocasa_layout4_returning_prop_v460 import motion
from run_robocasa_layout4_wait_and_remap_v461 import grid_class
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("robocasa_layout4_away_motion_v464.json")
PRIOR = ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-layout4-away-motion-v464"


def main():
    case = json.loads(CONFIG.read_text())["cases"][0]
    prior = np.load(PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    trial.model_with_layout2 = layout4_with_prop
    trial.planar_scan = height_scan((.05, .30))
    trial.obstacle_xy = motion(case)
    trial.START_X, trial.OBSTACLE_Y = case["from_xy_m"]
    route.MARGIN_M = .15
    rows = []
    for label, grid_factory in (("baseline", route.SensorGrid),
                                ("wait_remap_v461", grid_class(prior))):
        trial.SensorGrid = grid_factory
        trial.OUT = OUT / label
        episode = trial.episode(True)
        rows.append({"scenario": case, "package": label, "episode": episode})
        print(label, episode["outcome"], episode["steps"],
              round(episode["end_time_s"], 2),
              round(episode["min_true_prop_gap_m"], 3), flush=True)
    trial.SensorGrid = route.SensorGrid
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "config_sha256": route.sha256(CONFIG),
              "policy_source_sha256": route.sha256(
                  Path(__file__).with_name("run_robocasa_layout4_wait_and_remap_v461.py")),
              "tracker_source_sha256": route.sha256(
                  Path(__file__).with_name("novel_range_object.py")),
              "prior_map_sha256": route.sha256(PRIOR),
              "scope": "previously unseen westward trajectory above the route, unchanged V461 motion trigger versus frozen route",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
