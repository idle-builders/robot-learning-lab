"""Replay the articulated-owner route with persistent multi-object range tracks.

This changes only the tracker/planner wrapper from V491. The person and prop
remain unlabeled range objects to the actor; the prior and motion are frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import robocasa_owner_person as person
import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_layout2_multitrack_v485 import CAPTURED, grid_class
from multi_novel_range_objects import MultiNovelRangeObjects
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("robocasa_layout4_safe_side_v473.json")
PRIOR = ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-person-multitrack-v492"
TORSO_XY = [1., -2.4]
HAND_XYZ = [1.10, -2.24, .43]


def scene_with_owner(*, obstacle_xy):
    model = person.scene_with_person(prop_xy=obstacle_xy)
    person.set_owner_pose(model, TORSO_XY, HAND_XYZ, show_can=False)
    return model


def main():
    policy = json.loads(CONFIG.read_text())
    prior = np.load(PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    trial.model_with_layout2 = scene_with_owner
    trial.planar_scan = height_scan((.05, .30))
    trial.SensorGrid = grid_class(prior, policy)
    route.MARGIN_M = .15
    rows = []
    for case in policy["cases"]:
        trial.obstacle_xy = motion(case)
        trial.START_X, trial.OBSTACLE_Y = case["from_xy_m"]
        trial.OUT = OUT / case["id"]
        episode = trial.episode(True)
        grid = CAPTURED[-1]
        rows.append({"scenario": case, "episode": episode,
                     "track_history": grid.track_history,
                     "safe_side_history": grid.safe_side_history,
                     "decision_history": grid.decision_history,
                     "multi_history": grid.multi_history,
                     "multi_decisions": grid.multi_decisions})
        print(case["id"], episode["outcome"], episode["steps"],
              round(episode["end_time_s"], 2),
              [round(x, 3) for x in episode["end_true_xy_m"]], flush=True)
    trial.SensorGrid = route.SensorGrid
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "person_source_sha256": route.sha256(Path(person.__file__)),
              "multi_tracker_source_sha256": route.sha256(Path(__file__).with_name("multi_novel_range_objects.py")),
              "multi_planner_source_sha256": route.sha256(Path(__file__).with_name("run_robocasa_layout2_multitrack_v485.py")),
              "config_sha256": route.sha256(CONFIG),
              "prior_map_sha256": route.sha256(PRIOR),
              "scope": "furnished layout4 range/localized route with static articulated owner already present, taught can hidden, persistent multi-object tracker; 50cm kinematic ape proxy",
              "owner_torso_xy_m": TORSO_XY, "owner_hand_xyz_m": HAND_XYZ,
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
