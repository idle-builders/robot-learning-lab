"""Transfer V473 safe-side rule to an independently range-surveyed layout 2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_layout4_safe_side_v473 import CAPTURED, grid_class
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("robocasa_layout2_safe_side_v478.json")
PRIOR = ROOT / "artifacts/ape-robocasa-layout2-survey-v477/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-layout2-safe-side-v478"


def main() -> None:
    policy = json.loads(CONFIG.read_text())
    prior = np.load(PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    trial.model_with_layout2 = route.model_with_layout2
    trial.planar_scan = height_scan((.05, .30))
    route.MARGIN_M = .15
    rows = []
    for case in policy["cases"]:
        trial.obstacle_xy = motion(case)
        trial.START_X, trial.OBSTACLE_Y = case["from_xy_m"]
        trial.SensorGrid = grid_class(prior, policy)
        trial.OUT = OUT / case["id"]
        episode = trial.episode(True)
        grid = CAPTURED[-1]
        rows.append({"scenario": case, "package": "safe_side_layout2_v478",
                     "episode": episode, "track_history": grid.track_history,
                     "safe_side_history": grid.safe_side_history,
                     "decision_history": grid.decision_history})
        print(case["id"], episode["outcome"], episode["steps"],
              round(episode["end_time_s"], 2),
              [round(v, 3) for v in episode["end_true_xy_m"]], flush=True)
    trial.SensorGrid = route.SensorGrid
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "config_sha256": route.sha256(CONFIG),
              "prior_map_sha256": route.sha256(PRIOR),
              "frozen_policy_source_sha256": route.sha256(
                  Path(__file__).with_name("run_robocasa_layout4_safe_side_v473.py")),
              "scope": "same V473 rule, new range-built layout2 prior, previously authored dynamic paths; no owner turn",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
