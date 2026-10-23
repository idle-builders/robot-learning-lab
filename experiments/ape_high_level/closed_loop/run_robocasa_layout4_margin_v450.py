"""Develop an earlier-route clearance response to the revealed late crossing.

This is post-hoc tuning on V447's revealed failure. It cannot establish a
final margin or dynamic-person safety; it measures progress versus clearance
for static and late-moving versions with unchanged sensing and guard.
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
CONFIG = Path(__file__).with_name("robocasa_layout4_crossing_v447.json")
OUT = ROOT / "artifacts/ape-robocasa-layout4-margin-v450"


def main() -> None:
    source = json.loads(CONFIG.read_text())
    scenarios = [case for case in source["cases"]
                 if case["id"] in ("stationary", "late_crossing")]
    OUT.mkdir(parents=True, exist_ok=True)
    trial.model_with_layout2 = layout4_with_prop
    trial.planar_scan = height_scan((.05, .30))
    rows = []
    for margin in (.15, .25, .35, .45, .55, .65):
        route.MARGIN_M = margin
        for case in scenarios:
            trial.obstacle_xy = motion(case)
            trial.START_X, trial.OBSTACLE_Y = case["from_xy_m"]
            trial.OUT = OUT / f"margin_{int(margin * 100):02d}cm" / case["id"]
            episode = trial.episode(True)
            rows.append({"scenario": case, "package": f"margin_{int(margin * 100):02d}cm",
                         "map_margin_m": margin, "episode": episode})
            print(margin, case["id"], episode["outcome"], episode["steps"],
                  round(episode["min_true_prop_gap_m"], 3), flush=True)
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "config_sha256": route.sha256(CONFIG),
              "frozen_trial_source_sha256": route.sha256(Path(trial.__file__)),
              "scope": "post-hoc global map-margin development sweep 15-65cm, layout4 static and late-crossing prop, dual 5+30cm range",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
