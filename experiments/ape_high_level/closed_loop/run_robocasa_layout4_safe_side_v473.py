"""Choose a sensor-supported safe-side owner view before a possible pinch.

This is a revealed-case conservative response, not a general collision
guarantee. It assumes the newly observed compact object's total displacement
is bounded by a scenario-supplied 60 cm. The actor compares that envelope
with its own known route and current observed-free pose. If the route enters
the envelope but the current pose has reserve, it asks the owner to present
the item here instead of entering the narrow corridor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_layout4_crossing_v447 import layout4_with_prop
from run_robocasa_layout4_directional_wait_v465 import grid_class as directional_grid_class
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("robocasa_layout4_safe_side_v473.json")
PRIOR = ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-layout4-safe-side-v473"
CAPTURED = []


def grid_class(prior, policy):
    BaseGrid = directional_grid_class(prior)

    class SafeSideGrid(BaseGrid):
        def __init__(self):
            super().__init__()
            self.safe_side_history = []
            CAPTURED.append(self)

        def plan(self, origin_xy, goal_xy):
            track = self.track_history[-1]
            center = track["center_xy_m"]
            radius = track["radius_m"]
            if center is not None and radius is not None and self.mode == "navigate":
                proposed, status, path = route.SensorGrid.plan(self, origin_xy, goal_xy)
                if path:
                    center = np.asarray(center)
                    envelope = (radius + policy["actor_body_radius_m"] +
                                policy["required_gap_m"] +
                                policy["object_shape_allowance_m"] +
                                policy["assumed_object_max_total_displacement_m"])
                    route_nearest = min(float(np.linalg.norm(self.xy(cell) - center))
                                        for cell in path)
                    current_reserve = float(np.linalg.norm(origin_xy - center) - envelope)
                    should_stage = (route_nearest < envelope and
                                    current_reserve >= policy["minimum_staging_reserve_m"])
                    self.safe_side_history.append({
                        "time_s": track["capture_time_s"],
                        "object_center_xy_m": center.tolist(),
                        "fit_radius_m": radius,
                        "assumed_exclusion_radius_m": envelope,
                        "nearest_planned_route_center_m": route_nearest,
                        "current_staging_reserve_m": current_reserve,
                        "should_stage": should_stage})
                    if should_stage:
                        decision = "request_owner_show_here_safe_side"
                        self.decision_history.append({
                            "scan_index": len(self.track_history) - 1,
                            "time_s": track["capture_time_s"],
                            "mode": "safe_side_owner_request", "decision": decision,
                            "requested_xy_m": None})
                        return None, decision, []
            return super().plan(origin_xy, goal_xy)

    return SafeSideGrid


def main():
    config = json.loads(CONFIG.read_text())
    prior = np.load(PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    trial.model_with_layout2 = layout4_with_prop
    trial.planar_scan = height_scan((.05, .30))
    route.MARGIN_M = .15
    rows = []
    for case in config["cases"]:
        trial.obstacle_xy = motion(case)
        trial.START_X, trial.OBSTACLE_Y = case["from_xy_m"]
        trial.SensorGrid = grid_class(prior, config)
        trial.OUT = OUT / case["id"]
        episode = trial.episode(True)
        grid = CAPTURED[-1]
        rows.append({"scenario": case, "package": "safe_side_v473",
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
              "tracker_source_sha256": route.sha256(
                  Path(__file__).with_name("novel_range_object.py")),
              "scope": "actor-estimated new object plus scenario-bounded possible displacement chooses current-side owner view before late pinch; kinematic proxy",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
