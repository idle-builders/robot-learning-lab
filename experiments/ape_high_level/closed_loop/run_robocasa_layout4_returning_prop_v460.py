"""Frozen actor baseline for predeclared static, one-way and returning props."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from run_robocasa_layout4_crossing_v447 import layout4_with_prop
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("robocasa_layout4_returning_prop_v460.json")
OUT = ROOT / "artifacts/ape-robocasa-layout4-returning-prop-v460"


def motion(case):
    waypoints = [(float(t), np.array([x, y], dtype=float))
                 for t, x, y in case["waypoints"]]

    def at(time_s):
        if time_s <= waypoints[0][0]:
            return waypoints[0][1].copy()
        for (start_t, start), (end_t, end) in zip(waypoints, waypoints[1:]):
            if time_s <= end_t:
                return start + (time_s - start_t) / (end_t - start_t) * (end - start)
        return waypoints[-1][1].copy()

    return at


def main():
    cases = json.loads(CONFIG.read_text())["cases"]
    OUT.mkdir(parents=True, exist_ok=True)
    trial.model_with_layout2 = layout4_with_prop
    trial.planar_scan = height_scan((.05, .30))
    route.MARGIN_M = .15
    rows = []
    for case in cases:
        trial.obstacle_xy = motion(case)
        trial.START_X, trial.OBSTACLE_Y = case["from_xy_m"]
        trial.OUT = OUT / case["id"]
        episode = trial.episode(True)
        rows.append({"scenario": case, "package": "frozen_dual05_30_margin15",
                     "episode": episode})
        print(case["id"], episode["outcome"], episode["steps"],
              round(episode["end_time_s"], 2),
              round(episode["min_true_prop_gap_m"], 3), flush=True)
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "config_sha256": route.sha256(CONFIG),
              "frozen_trial_source_sha256": route.sha256(Path(trial.__file__)),
              "scope": "predeclared same-initial-prop static, one-way, returning layouts; frozen dual-height actor and ideal 20ms guard",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
