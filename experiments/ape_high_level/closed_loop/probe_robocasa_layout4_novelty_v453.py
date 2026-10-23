"""Compare range endpoints to a saved sensor-derived free-space survey.

The novelty flag uses only current noisy range, estimated actor pose and the
previous survey grid. Prop coordinates are used solely for evaluator scoring.
This probes whether a new movable-object candidate can be distinguished from
normal scan/map mismatch before adding a dynamic occupancy planner.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from run_robocasa_layout4_crossing_v447 import layout4_with_prop
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
MAP = ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-layout4-novelty-v453"
CAPTURED = []


def grid_class(prior):
    class NoveltyGrid(route.SensorGrid):
        def __init__(self):
            super().__init__()
            assert np.array_equal(self.xs, prior["xs"])
            assert np.array_equal(self.ys, prior["ys"])
            self.novel_history = []
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
                    novel.append(point.tolist())
            self.novel_history.append({"pose_xy_m": pose.tolist(),
                                       "novel_endpoints_xy_m": novel,
                                       "novel_endpoint_count": len(novel)})
            super().update(ranges, pose, max_range_m)

    return NoveltyGrid


def main() -> None:
    prior = np.load(MAP)
    OUT.mkdir(parents=True, exist_ok=True)
    trial.model_with_layout2 = layout4_with_prop
    trial.planar_scan = height_scan((.05, .30))
    route.MARGIN_M = .15
    reports = []
    for label, xy in (("absent", (-10., -10.)),
                      ("stationary_prop", (3., -1.05))):
        trial.START_X, trial.OBSTACLE_Y = xy
        trial.obstacle_xy = lambda _time, point=np.asarray(xy): point.copy()
        trial.SensorGrid = grid_class(prior)
        trial.OUT = OUT / label
        episode = trial.episode(True)
        history = CAPTURED[-1].novel_history
        counts = [row["novel_endpoint_count"] for row in history]
        near_prop = [int(sum(np.linalg.norm(np.asarray(point) - [3., -1.05]) < .40
                             for point in row["novel_endpoints_xy_m"]))
                     for row in history]
        reports.append({"case": label, "episode": episode,
                        "novelty": history,
                        "max_novel_endpoints_per_scan": max(counts),
                        "scans_with_three_novel_endpoints": sum(n >= 3 for n in counts),
                        "max_novel_endpoints_near_prop_evaluator": max(near_prop),
                        "scans_with_three_near_prop_evaluator": int(sum(n >= 3 for n in near_prop))})
        print(label, episode["outcome"], episode["steps"],
              "max_novel", max(counts),
              "scans3", sum(n >= 3 for n in counts),
              "near_prop3", sum(n >= 3 for n in near_prop), flush=True)
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    trial.SensorGrid = route.SensorGrid
    report = {"source_sha256": route.sha256(Path(__file__)),
              "prior_map_sha256": route.sha256(MAP),
              "frozen_trial_source_sha256": route.sha256(Path(trial.__file__)),
              "scope": "sensor-derived prior-free grid versus actor-estimated current dual-height range endpoints; prop location evaluator-only",
              "cases": reports}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
