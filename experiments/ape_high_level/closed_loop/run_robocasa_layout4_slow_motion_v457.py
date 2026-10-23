"""Challenge the frozen layout-4 actor with visible slow prop motion.

The static and moving scenes begin with identical geometry. Motion is
predeclared in the adjacent JSON. Actor input remains only live range/RGB and
estimated pose; the prop trajectory is evaluator-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_layout4_crossing_v447 import layout4_with_prop
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("robocasa_layout4_slow_motion_v457.json")
OUT = ROOT / "artifacts/ape-robocasa-layout4-slow-motion-v457"


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
              "scope": "layout4 predeclared static versus slow 8-16s continuous crossing; dual 5+30cm range, 15cm margin, ideal 20ms guard",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
