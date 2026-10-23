"""Compare persistent multi-object belief with V482's one-object selection.

The base occupancy/planner is frozen. A multi-track check stages before any
route enters any confirmed object's motion region, or refuses movement if
the current pose already lies in one. This run tests belief persistence and
decision behavior in the authored two-prop scenes, not general tracking.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_two_props_v482 as two
import run_robocasa_moving_prop_v423 as trial
from multi_novel_range_objects import MultiNovelRangeObjects
from run_robocasa_layout4_safe_side_v473 import grid_class as single_grid_class
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("robocasa_layout2_two_props_v482.json")
PRIOR = ROOT / "artifacts/ape-robocasa-layout2-survey-v477/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-layout2-multitrack-v485"
CAPTURED = []


def grid_class(prior, policy):
    BaseGrid = single_grid_class(prior, policy)

    class MultiSafeSideGrid(BaseGrid):
        def __init__(self):
            super().__init__()
            self.multiple = MultiNovelRangeObjects(prior["xs"], prior["ys"],
                                                   prior["free"], prior["occupied"])
            self.multi_history = []
            self.multi_decisions = []
            CAPTURED.append(self)

        def update(self, ranges, pose, max_range_m):
            super().update(ranges, pose, max_range_m)
            time_s = self.track_history[-1]["capture_time_s"]
            self.multi_history.append(self.multiple.update(ranges, pose, time_s))

        def plan(self, origin_xy, goal_xy):
            multi = self.multi_history[-1]
            confirmed = [track for track in multi["tracks"] if track["hits"] >= 2]
            if confirmed and self.mode == "navigate":
                proposed, status, path = route.SensorGrid.plan(self, origin_xy, goal_xy)
                if path:
                    conflicts = []
                    current_reserves = []
                    for track in confirmed:
                        center = np.asarray(track["center_xy_m"])
                        envelope = (track["radius_m"] + policy["actor_body_radius_m"] +
                                    policy["required_gap_m"] +
                                    policy["object_shape_allowance_m"] +
                                    policy["assumed_object_max_total_displacement_m"])
                        route_nearest = min(float(np.linalg.norm(self.xy(cell) - center))
                                            for cell in path)
                        reserve = float(np.linalg.norm(origin_xy - center) - envelope)
                        current_reserves.append(reserve)
                        if route_nearest < envelope:
                            conflicts.append({"track_id": track["id"],
                                              "route_nearest_m": route_nearest,
                                              "envelope_m": envelope,
                                              "track_age_s": track["age_s"]})
                    decision = "continue_clear_route"
                    if conflicts:
                        decision = ("request_owner_show_here_safe_side"
                                    if min(current_reserves) >=
                                    policy["minimum_staging_reserve_m"] else
                                    "stop_current_pose_inside_object_region")
                    self.multi_decisions.append({"time_s": multi["capture_time_s"],
                                                 "confirmed_track_ids": [t["id"] for t in confirmed],
                                                 "conflicts": conflicts,
                                                 "minimum_current_reserve_m": min(current_reserves),
                                                 "decision": decision})
                    if conflicts:
                        return None, decision, []
            return super().plan(origin_xy, goal_xy)

    return MultiSafeSideGrid


def main() -> None:
    policy = json.loads(CONFIG.read_text())
    prior = np.load(PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    trial.model_with_layout2 = two.scene_with_two_props
    trial.obstacle_xy = two.primary_and_secondary_xy
    trial.image_frame = two.dual_prop_frame
    trial.planar_scan = height_scan((.05, .30))
    trial.SensorGrid = grid_class(prior, policy)
    route.MARGIN_M = .15
    rows = []
    for case in policy["cases"]:
        two.ACTIVE_CASE = {**case, "primary_xy_m": policy["primary_xy_m"]}
        trial.START_X, trial.OBSTACLE_Y = policy["primary_xy_m"]
        trial.OUT = OUT / case["id"]
        episode = trial.episode(True)
        grid = CAPTURED[-1]
        rows.append({"scenario": two.ACTIVE_CASE, "package": "multi_safe_side_v485",
                     "episode": episode, "track_history": grid.track_history,
                     "multi_history": grid.multi_history,
                     "multi_decisions": grid.multi_decisions,
                     "safe_side_history": grid.safe_side_history})
        print(case["id"], episode["outcome"], episode["steps"],
              round(episode["end_time_s"], 2),
              "confirmed_ids", [t["id"] for t in grid.multi_history[-1]["tracks"]
                                if t["hits"] >= 2], flush=True)
    trial.SensorGrid = route.SensorGrid
    trial.image_frame = two.BASE_IMAGE_FRAME
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "multi_tracker_source_sha256": route.sha256(
                  Path(__file__).with_name("multi_novel_range_objects.py")),
              "config_sha256": route.sha256(CONFIG),
              "prior_map_sha256": route.sha256(PRIOR),
              "two_prop_scene_source_sha256": route.sha256(Path(two.__file__)),
              "scope": "persistent compact-object range tracks and multi-region route check in authored two-prop layout2 scenes",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
