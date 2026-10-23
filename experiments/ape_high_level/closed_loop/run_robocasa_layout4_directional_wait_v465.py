"""Use observed object velocity and the planned route to decide when to hold.

This revealed-case revision corrects V461's false owner prompt for an object
moving above, rather than into, the route. It remains a constant-velocity
prediction from 0.4 s scans with an ideal 20 ms near-body guard; it has no
physical person or stopping model and is not a safety certificate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from run_robocasa_layout4_crossing_v447 import layout4_with_prop
from run_robocasa_layout4_returning_prop_v460 import motion
from run_robocasa_layout4_wait_and_remap_v461 import grid_class as wait_grid_class
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIGS = [Path(__file__).with_name("robocasa_layout4_returning_prop_v460.json"),
           Path(__file__).with_name("robocasa_layout4_away_motion_v464.json")]
PRIOR = ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-layout4-directional-wait-v465"
CAPTURED = []
HORIZON_S = 6.
CAPTURE_S = .4
SPEED_TRIGGER_M_S = .05
BODY_RADIUS_M = .17
REQUIRED_GAP_M = .05
PREDICTION_ALLOWANCE_M = .03


def grid_class(prior):
    BaseGrid = wait_grid_class(prior)

    class DirectionalWaitGrid(BaseGrid):
        def __init__(self):
            super().__init__()
            self.risk_history = []
            CAPTURED.append(self)

        def update(self, ranges, pose, max_range_m):
            previous_mode = self.mode
            super().update(ranges, pose, max_range_m)
            time_s = self.track_history[-1]["capture_time_s"]
            # V461 triggers on speed alone. Defer that transition until the
            # velocity has been compared with the actor-selected path.
            if (previous_mode in ("navigate", "navigate_after_fresh_remap") and
                    self.mode == "wait_reobserve" and self.wait_start_s == time_s):
                self.mode = previous_mode
                self.wait_start_s = None

        def plan(self, origin_xy, goal_xy):
            if self.mode in ("navigate", "navigate_after_fresh_remap"):
                proposed, status, path = route.SensorGrid.plan(self, origin_xy, goal_xy)
                track = self.track_history[-1]
                velocity = track["velocity_xy_m_s"]
                if (velocity is not None and track["center_xy_m"] is not None and
                        track["radius_m"] is not None and
                        np.linalg.norm(velocity) > SPEED_TRIGGER_M_S and path):
                    center = np.asarray(track["center_xy_m"])
                    velocity = np.asarray(velocity)
                    radius = track["radius_m"]
                    samples = []
                    for index, cell in enumerate(path[:int(HORIZON_S / CAPTURE_S) + 1]):
                        future_s = index * CAPTURE_S
                        prop_center = center + future_s * velocity
                        gap = float(np.linalg.norm(self.xy(cell) - prop_center) - radius - BODY_RADIUS_M)
                        samples.append((gap, future_s))
                    minimum_gap, encounter_s = min(samples)
                    hold_gaps = [float(np.linalg.norm(origin_xy -
                                          (center + future_s * velocity)) -
                                       radius - BODY_RADIUS_M)
                                 for future_s in np.arange(0., HORIZON_S + .01, CAPTURE_S)]
                    hold_min = min(hold_gaps)
                    risky_route = minimum_gap < REQUIRED_GAP_M + PREDICTION_ALLOWANCE_M
                    self.risk_history.append({"time_s": track["capture_time_s"],
                                              "route_min_predicted_gap_m": minimum_gap,
                                              "route_encounter_after_s": encounter_s,
                                              "hold_min_predicted_gap_m": hold_min,
                                              "risky_route": risky_route})
                    if risky_route:
                        if hold_min < REQUIRED_GAP_M + PREDICTION_ALLOWANCE_M:
                            self.mode = "ask_owner_clear_corridor"
                            return super().plan(origin_xy, goal_xy)
                        self.mode = "wait_reobserve"
                        self.wait_start_s = track["capture_time_s"]
            return super().plan(origin_xy, goal_xy)

    return DirectionalWaitGrid


def main():
    prior = np.load(PRIOR)
    cases = [case for config in CONFIGS
             for case in json.loads(config.read_text())["cases"]]
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
        grid = CAPTURED[-1]
        rows.append({"scenario": case, "package": "directional_wait_v465",
                     "episode": episode, "track_history": grid.track_history,
                     "risk_history": grid.risk_history,
                     "decision_history": grid.decision_history})
        print(case["id"], episode["outcome"], episode["steps"],
              round(episode["end_time_s"], 2),
              round(episode["min_true_prop_gap_m"], 3),
              "holds", sum(d["decision"] == "hold_and_reobserve_moving_object"
                           for d in grid.decision_history), flush=True)
    trial.SensorGrid = route.SensorGrid
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "wait_policy_source_sha256": route.sha256(
                  Path(__file__).with_name("run_robocasa_layout4_wait_and_remap_v461.py")),
              "tracker_source_sha256": route.sha256(
                  Path(__file__).with_name("novel_range_object.py")),
              "config_sha256": {str(path): route.sha256(path) for path in CONFIGS},
              "prior_map_sha256": route.sha256(PRIOR),
              "prediction": {"horizon_s": HORIZON_S,
                             "capture_s": CAPTURE_S,
                             "speed_trigger_m_s": SPEED_TRIGGER_M_S,
                             "body_radius_m": BODY_RADIUS_M,
                             "required_gap_m": REQUIRED_GAP_M,
                             "prediction_allowance_m": PREDICTION_ALLOWANCE_M},
              "scope": "actor-side constant-velocity planned-path screen before V461 wait/remap; same seeded layout4 and predeclared directions",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
