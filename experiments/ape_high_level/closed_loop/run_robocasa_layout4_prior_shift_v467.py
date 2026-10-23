"""Stress the actor's novelty track with a 10 cm misregistered prior map.

The live range and localizer are unchanged. Only the old survey's free and
occupied arrays shift by one 10 cm grid cell, with newly exposed edges marked
unknown. This is a synthetic map error, not measured localization noise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from run_robocasa_layout4_crossing_v447 import layout4_with_prop
from run_robocasa_layout4_directional_wait_v465 import grid_class
from run_robocasa_layout4_returning_prop_v460 import motion
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("robocasa_layout4_prior_shift_v467.json")
SCENARIOS = Path(__file__).with_name("robocasa_layout4_returning_prop_v460.json")
PRIOR = ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-layout4-prior-shift-v467"


def shifted(prior, dx_cells, dy_cells):
    arrays = {"xs": prior["xs"], "ys": prior["ys"]}
    for name in ("free", "occupied"):
        value = np.roll(prior[name], shift=(dy_cells, dx_cells), axis=(0, 1))
        if dx_cells > 0:
            value[:, :dx_cells] = False
        elif dx_cells < 0:
            value[:, dx_cells:] = False
        if dy_cells > 0:
            value[:dy_cells, :] = False
        elif dy_cells < 0:
            value[dy_cells:, :] = False
        arrays[name] = value
    return arrays


def main():
    config = json.loads(CONFIG.read_text())
    scenarios = {case["id"]: case for case in json.loads(SCENARIOS.read_text())["cases"]}
    prior = np.load(PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    trial.model_with_layout2 = layout4_with_prop
    trial.planar_scan = height_scan((.05, .30))
    route.MARGIN_M = .15
    rows = []
    for dx, dy in config["prior_offsets_cells"]:
        altered = shifted(prior, dx, dy)
        for name in config["cases"]:
            case = scenarios[name]
            trial.obstacle_xy = motion(case)
            trial.START_X, trial.OBSTACLE_Y = case["from_xy_m"]
            trial.SensorGrid = grid_class(altered)
            trial.OUT = OUT / f"prior_dx{dx}_dy{dy}" / name
            episode = trial.episode(True)
            rows.append({"scenario": case, "package": f"prior_dx{dx}_dy{dy}",
                         "offset_cells_xy": [dx, dy], "episode": episode})
            print(dx, dy, name, episode["outcome"], episode["steps"],
                  round(episode["end_time_s"], 2),
                  round(episode["min_true_prop_gap_m"], 3), flush=True)
    trial.SensorGrid = route.SensorGrid
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "policy_source_sha256": route.sha256(
                  Path(__file__).with_name("run_robocasa_layout4_directional_wait_v465.py")),
              "tracker_source_sha256": route.sha256(
                  Path(__file__).with_name("novel_range_object.py")),
              "config_sha256": route.sha256(CONFIG),
              "scenario_config_sha256": route.sha256(SCENARIOS),
              "prior_map_sha256": route.sha256(PRIOR),
              "scope": "frozen one-cell old-map translation stress, live actor and scene unchanged",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
