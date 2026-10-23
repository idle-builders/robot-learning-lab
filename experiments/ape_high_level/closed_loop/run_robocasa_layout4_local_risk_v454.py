"""Development sweep: local risk around a sensor-detected novel obstacle.

The detector compares live dual-height range endpoints in the actor's
estimated frame with a saved, sensor-derived prior map. It never reads the
simulator prop pose or evaluator geometry. This is a revealed layout-4 case,
so it is tuning evidence, not held-out validation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_layout4_crossing_v447 import layout4_with_prop
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("robocasa_layout4_crossing_v447.json")
PRIOR = ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-layout4-local-risk-v454"
CAPTURED = []


def grid_class(prior, radius_m):
    class LocalRiskGrid(route.SensorGrid):
        def __init__(self):
            super().__init__()
            assert np.array_equal(self.xs, prior["xs"])
            assert np.array_equal(self.ys, prior["ys"])
            self.radius_m = radius_m
            self.candidate_streak = 0
            self.risk_center = None
            self.risk_history = []
            CAPTURED.append(self)

        def update(self, ranges, pose, max_range_m):
            novel = []
            for angle, distance in zip(np.linspace(-math.pi, math.pi, len(ranges), endpoint=False), ranges):
                if distance is None:
                    continue
                point = pose + distance * np.array([math.cos(angle), math.sin(angle)])
                if not (self.xs[0] <= point[0] <= self.xs[-1] and
                        self.ys[0] <= point[1] <= self.ys[-1]):
                    continue
                cell = self.cell(point)
                if prior["free"][cell] and not prior["occupied"][cell]:
                    novel.append(point)
            self.candidate_streak = self.candidate_streak + 1 if len(novel) >= 3 else 0
            if self.candidate_streak >= 2:
                self.risk_center = np.mean(novel, axis=0)
            self.risk_history.append({"candidate_count": len(novel),
                                      "candidate_streak": self.candidate_streak,
                                      "risk_center_xy_m": self.risk_center.tolist() if self.risk_center is not None else None})
            super().update(ranges, pose, max_range_m)

        def plan(self, origin_xy, goal_xy):
            if self.risk_center is None:
                return super().plan(origin_xy, goal_xy)
            mask = np.hypot(self.xx - self.risk_center[0],
                            self.yy - self.risk_center[1]) < self.radius_m
            original_free = self.free
            self.free = original_free & ~mask
            try:
                if mask[self.cell(origin_xy)]:
                    return None, "inside_novel_object_risk_region", []
                return super().plan(origin_xy, goal_xy)
            finally:
                self.free = original_free

    return LocalRiskGrid


def main() -> None:
    prior = np.load(PRIOR)
    scenarios = [case for case in json.loads(CONFIG.read_text())["cases"]
                 if case["id"] in ("stationary", "early_crossing", "late_crossing")]
    OUT.mkdir(parents=True, exist_ok=True)
    trial.model_with_layout2 = layout4_with_prop
    trial.planar_scan = height_scan((.05, .30))
    route.MARGIN_M = .15
    rows = []
    for radius in (.40, .50, .60, .70, .80):
        for case in scenarios:
            trial.obstacle_xy = motion(case)
            trial.START_X, trial.OBSTACLE_Y = case["from_xy_m"]
            trial.SensorGrid = grid_class(prior, radius)
            trial.OUT = OUT / f"radius_{int(radius * 100):02d}cm" / case["id"]
            episode = trial.episode(True)
            risks = CAPTURED[-1].risk_history
            rows.append({"scenario": case, "risk_radius_m": radius,
                         "episode": episode, "risk_history": risks})
            print(radius, case["id"], episode["outcome"], episode["steps"],
                  round(episode["end_time_s"], 2),
                  round(episode["min_true_prop_gap_m"], 3),
                  "first_risk_step", next((i for i, row in enumerate(risks)
                                           if row["risk_center_xy_m"] is not None), None), flush=True)
    trial.SensorGrid = route.SensorGrid
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "config_sha256": route.sha256(CONFIG),
              "prior_map_sha256": route.sha256(PRIOR),
              "frozen_trial_source_sha256": route.sha256(Path(trial.__file__)),
              "scope": "revealed layout4 development sweep, actor-estimated dual-height range novelty with local route mask, ideal 20ms proximity guard",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
