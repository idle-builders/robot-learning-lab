"""Observe actor-side novel-object shape and velocity without changing route.

This diagnostic checks whether the predeclared slow crossing becomes visible
in time for a future occupancy planner. Object truth is added only after the
closed-loop actor has produced its scan, localization and action history.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from novel_range_object import NovelRangeObject
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_layout4_crossing_v447 import layout4_with_prop
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("robocasa_layout4_slow_motion_v457.json")
PRIOR = ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-layout4-object-track-v459"
CAPTURED = []


def grid_class(prior):
    class TrackingGrid(route.SensorGrid):
        def __init__(self):
            super().__init__()
            assert np.array_equal(self.xs, prior["xs"])
            assert np.array_equal(self.ys, prior["ys"])
            self.tracker = NovelRangeObject(self.xs, self.ys,
                                            prior["free"], prior["occupied"])
            self.track_history = []
            CAPTURED.append(self)

        def update(self, ranges, pose, max_range_m):
            time_s = len(self.track_history) * trial.ACTION_S
            self.track_history.append(self.tracker.update(ranges, pose, time_s))
            super().update(ranges, pose, max_range_m)

    return TrackingGrid


def main():
    prior = np.load(PRIOR)
    cases = json.loads(CONFIG.read_text())["cases"]
    OUT.mkdir(parents=True, exist_ok=True)
    trial.model_with_layout2 = layout4_with_prop
    trial.planar_scan = height_scan((.05, .30))
    route.MARGIN_M = .15
    rows = []
    for case in cases:
        trial.obstacle_xy = motion(case)
        trial.START_X, trial.OBSTACLE_Y = case["from_xy_m"]
        trial.SensorGrid = grid_class(prior)
        trial.OUT = OUT / case["id"]
        episode = trial.episode(True)
        history = CAPTURED[-1].track_history
        truth = motion(case)
        for row in history:
            center = row["center_xy_m"]
            row["evaluator_true_center_xy_m"] = truth(row["capture_time_s"]).tolist()
            row["evaluator_center_error_m"] = (
                float(np.linalg.norm(np.asarray(center) - truth(row["capture_time_s"])))
                if center is not None else None)
        rows.append({"scenario": case, "package": "frozen_route_with_track_diagnostic",
                     "episode": episode, "track_history": history})
        valid = [row for row in history if row["center_xy_m"] is not None]
        moving = [row for row in history if row["velocity_xy_m_s"] is not None
                  and np.linalg.norm(row["velocity_xy_m_s"]) > .04]
        print(case["id"], episode["outcome"], "confirmed", len(valid),
              "mean_center_error", round(float(np.mean([row["evaluator_center_error_m"]
                                                         for row in valid])), 3) if valid else None,
              "first_speed_gt4cms", moving[0]["capture_time_s"] if moving else None,
              flush=True)
    trial.SensorGrid = route.SensorGrid
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "tracker_source_sha256": route.sha256(
                  Path(__file__).with_name("novel_range_object.py")),
              "config_sha256": route.sha256(CONFIG),
              "prior_map_sha256": route.sha256(PRIOR),
              "frozen_trial_source_sha256": route.sha256(Path(trial.__file__)),
              "scope": "actor-side compact-object fit and velocity diagnostic; frozen route unchanged; truth only evaluator scores",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
