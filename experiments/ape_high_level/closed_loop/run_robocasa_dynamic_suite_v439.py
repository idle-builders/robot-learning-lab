"""Run frozen moving-prop trajectories with two actor range configurations.

The actor sees only its scans and current RGB render. Simulator prop motion
never enters its map or motion choice except through sensed rays. This still
uses a kinematic eye rig and an ideal 20 ms full-scan proximity guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
SUITE = Path(__file__).with_name("robocasa_dynamic_suite_v439.json")
OUT = ROOT / "artifacts/ape-robocasa-dynamic-suite-v439"
ORIGINAL_SCAN = route.planar_scan


def motion(case: dict):
    start = np.asarray(case["from_xy_m"], dtype=float)
    end = np.asarray(case["to_xy_m"], dtype=float)
    begin, finish = float(case["begin_s"]), float(case["end_s"])

    def xy(time_s: float) -> np.ndarray:
        fraction = min(1., max(0., (time_s - begin) / (finish - begin)))
        return start + fraction * (end - start)

    return xy


def main() -> None:
    suite = json.loads(SUITE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in suite["cases"]:
        trial.obstacle_xy = motion(case)
        trial.START_X, trial.OBSTACLE_Y = case["from_xy_m"]
        for package, margin, heights in (("mid30_margin10", .10, (.30,)),
                                         ("dual05_30_margin15", .15, (.05, .30))):
            route.MARGIN_M = margin
            route.LIDAR_Z_M = .30
            trial.planar_scan = (ORIGINAL_SCAN if len(heights) == 1
                                 else height_scan(heights))
            trial.OUT = OUT / case["id"] / package
            episode = trial.episode(True)
            rows.append({"scenario": case, "package": package,
                         "map_margin_m": margin, "scan_heights_m": heights,
                         "episode": episode})
            print(case["id"], package, episode["outcome"],
                  episode["steps"], round(episode["min_true_prop_gap_m"], 3), flush=True)
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    trial.planar_scan = ORIGINAL_SCAN
    report = {"source_sha256": route.sha256(Path(__file__)),
              "suite_sha256": route.sha256(SUITE),
              "frozen_trial_source_sha256": route.sha256(Path(trial.__file__)),
              "height_adapter_source_sha256": route.sha256(
                  Path(__file__).with_name("run_robocasa_scan_height_v433.py")),
              "scope": "four predeclared moving-prop layout2 cases x two fixed sensor/map packages; ideal kinematic body and 20ms full-range guard",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
