"""Close a bounded motion cue -> safe hold -> fresh map -> resume/help loop.

The same actor-side range/pose-derived object track is used for all three
predeclared cases. A moving compact object pauses translation at the current
observed-free pose; the actor keeps sensing during the hold. If the object
returns near its first observed center and stays still twice, accumulated
occupancy is rebuilt from the current scan before routing again. Otherwise
the ape reports that the corridor needs owner help after a bounded wait.
Neither object script nor evaluator geometry enters the decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as trial
from novel_range_object import NovelRangeObject
from run_robocasa_layout4_crossing_v447 import layout4_with_prop
from run_robocasa_layout4_returning_prop_v460 import motion
from run_robocasa_scan_height_v433 import height_scan


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(__file__).with_name("robocasa_layout4_returning_prop_v460.json")
PRIOR = ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"
OUT = ROOT / "artifacts/ape-robocasa-layout4-wait-remap-v461"
CAPTURED = []
SPEED_TRIGGER_M_S = .05
RETURN_CENTER_M = .07
STILL_SPEED_M_S = .04
MAX_WAIT_S = 10.


def grid_class(prior):
    class WaitAndRemapGrid(route.SensorGrid):
        def __init__(self):
            super().__init__()
            assert np.array_equal(self.xs, prior["xs"])
            assert np.array_equal(self.ys, prior["ys"])
            self.tracker = NovelRangeObject(self.xs, self.ys,
                                            prior["free"], prior["occupied"])
            self.track_history = []
            self.decision_history = []
            self.anchor_center = None
            self.mode = "navigate"
            self.wait_start_s = None
            self.return_streak = 0
            CAPTURED.append(self)

        def update(self, ranges, pose, max_range_m):
            capture_time_s = getattr(self, "next_capture_time_s", None)
            time_s = (len(self.track_history) * trial.ACTION_S
                      if capture_time_s is None else float(capture_time_s))
            self.next_capture_time_s = None
            track = self.tracker.update(ranges, pose, time_s)
            self.track_history.append(track)
            center = track["center_xy_m"]
            velocity = track["velocity_xy_m_s"]
            if self.anchor_center is None and center is not None:
                self.anchor_center = np.asarray(center)
            if (self.mode == "navigate" and velocity is not None and
                    np.linalg.norm(velocity) > SPEED_TRIGGER_M_S):
                self.mode = "wait_reobserve"
                self.wait_start_s = time_s
            if self.mode == "wait_reobserve":
                returned = (center is not None and velocity is not None and
                            np.linalg.norm(np.asarray(center) - self.anchor_center) < RETURN_CENTER_M and
                            np.linalg.norm(velocity) < STILL_SPEED_M_S)
                self.return_streak = self.return_streak + 1 if returned else 0
                if self.return_streak >= 2:
                    self.free.fill(False)
                    self.occupied.fill(False)
                    self.visits.fill(0)
                    self.mode = "navigate_after_fresh_remap"
                elif time_s - self.wait_start_s >= MAX_WAIT_S:
                    self.mode = "ask_owner_clear_corridor"
            super().update(ranges, pose, max_range_m)

        def plan(self, origin_xy, goal_xy):
            if self.mode == "wait_reobserve":
                decision = (origin_xy.copy(), "hold_and_reobserve_moving_object", [])
            elif self.mode == "ask_owner_clear_corridor":
                decision = (None, "ask_owner_clear_corridor", [])
            else:
                decision = super().plan(origin_xy, goal_xy)
            self.decision_history.append({
                "scan_index": len(self.track_history) - 1,
                "time_s": self.track_history[-1]["capture_time_s"],
                "mode": self.mode, "decision": decision[1],
                "requested_xy_m": (decision[0].tolist()
                                   if decision[0] is not None else None)})
            return decision

    return WaitAndRemapGrid


def main():
    prior = np.load(PRIOR)
    cases = json.loads(CONFIG.read_text())["cases"]
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
        rows.append({"scenario": case, "package": "actor_track_wait_remap_v461",
                     "episode": episode, "track_history": grid.track_history,
                     "decision_history": grid.decision_history})
        holds = sum(row["decision"] == "hold_and_reobserve_moving_object"
                    for row in grid.decision_history)
        print(case["id"], episode["outcome"], episode["steps"],
              round(episode["end_time_s"], 2),
              round(episode["min_true_prop_gap_m"], 3),
              "holds", holds, "final_mode", grid.mode, flush=True)
    trial.SensorGrid = route.SensorGrid
    route.MARGIN_M = .10
    route.LIDAR_Z_M = .30
    report = {"source_sha256": route.sha256(Path(__file__)),
              "tracker_source_sha256": route.sha256(
                  Path(__file__).with_name("novel_range_object.py")),
              "config_sha256": route.sha256(CONFIG),
              "prior_map_sha256": route.sha256(PRIOR),
              "frozen_trial_source_sha256": route.sha256(Path(trial.__file__)),
              "policy": {"speed_trigger_m_s": SPEED_TRIGGER_M_S,
                         "return_center_m": RETURN_CENTER_M,
                         "still_speed_m_s": STILL_SPEED_M_S,
                         "return_consecutive_scans": 2,
                         "max_wait_s": MAX_WAIT_S},
              "scope": "actor-only track cue, zero-translation holds, fresh map rebuilding, bounded owner-help outcome; 50cm kinematic proxy and ideal 20ms guard",
              "cases": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
