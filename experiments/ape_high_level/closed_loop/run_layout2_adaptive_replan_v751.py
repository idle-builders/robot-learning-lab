"""Revalidate the actor-selected route on fresh 10 Hz track packets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import actor_range_refuge_v744 as selector
import run_layout2_adaptive_emergency_v747 as emergency
import run_layout2_adaptive_exit_turn_v749 as turn
import run_layout2_adaptive_refuge_v744 as adaptive
import run_robocasa_layout2_route_v407 as route
import run_robocasa_moving_prop_v423 as moving


SOURCE = Path(__file__)
OUT = adaptive.west.side.ROOT / "artifacts/ape-robocasa-layout2-adaptive-replan-v751"
MIN_REPLAN_GAIN_M = .08
MIN_REPLAN_INTERVAL_S = .25


class ReplanningEmergencyMonitor(emergency.EmergencyMonitor):
    def __init__(self, model, data, actual, script, grid, estimate):
        super().__init__(model, data, actual, script, grid, estimate)
        self.hazard_id = None
        self.hazard_centers = []
        self.last_replan_s = None

    def select_response(self, track, time_s, result):
        goal, evidence = super().select_response(track, time_s, result)
        if track is not None:
            self.hazard_id = track["id"]
            self.hazard_centers.append((time_s, np.asarray(track["center_xy_m"])))
        return goal, evidence

    def __call__(self, time_s):
        result = super().__call__(time_s)
        if (self.trigger_time_s is None or "actor_confirmed_tracks" not in result or
                time_s - self.trigger_time_s < MIN_REPLAN_INTERVAL_S):
            return result
        current_track = next((track for track in result["actor_confirmed_tracks"]
                              if track["id"] == self.hazard_id), None)
        if current_track is None:
            result["actor_replan"] = {"status": "hazard_track_missing",
                                      "capture_time_s": time_s}
            return result
        self.hazard_centers.append((time_s,
                                    np.asarray(current_track["center_xy_m"])))
        if len(self.hazard_centers) < 3:
            return result
        older_t, older_center = self.hazard_centers[-3]
        velocity = ((self.hazard_centers[-1][1] - older_center) /
                    max(.001, time_s - older_t))
        proposed, evidence = selector.choose(
            self.grid, self.estimate[:2], current_track, velocity,
            adaptive.full.FIRST_SURFACE, allow_best_effort=True,
            consider_goal_xy_m=self.response_goal)
        current = evidence["considered_goal_evidence"]
        current_gap = (-float("inf") if current is None else
                       current["minimum_predicted_gap_lower_m"])
        proposed_gap = evidence["chosen_minimum_gap_lower_m"]
        decision = {"status": "keep_current_route",
                    "capture_time_s": time_s,
                    "track_id": current_track["id"],
                    "velocity_xy_m_s": velocity.tolist(),
                    "current_goal_xy_m": (None if self.response_goal is None else
                                          self.response_goal.tolist()),
                    "current_goal_predicted_gap_lower_m": (
                        None if current is None else current_gap),
                    "proposed_goal_xy_m": None if proposed is None else proposed.tolist(),
                    "proposed_goal_predicted_gap_lower_m": proposed_gap,
                    "selection_status": evidence["status"],
                    "selection_wall_s": evidence["selection_wall_s"]}
        result["actor_replan"] = decision
        if (proposed is None or proposed_gap is None or
                self.response_goal is not None and
                np.linalg.norm(proposed - self.response_goal) < .08 or
                self.last_replan_s is not None and
                time_s - self.last_replan_s < MIN_REPLAN_INTERVAL_S or
                proposed_gap < current_gap + MIN_REPLAN_GAIN_M):
            return result
        if result["actor_guard_range_m"] < moving.GUARD_CENTER_RANGE_M:
            decision["status"] = "raw_guard_blocks_replan"
            return result
        requested, status, path = route.SensorGrid.plan(
            self.grid, self.estimate[:2], proposed)
        decision["actor_plan_status"] = status
        if requested is None or status != "known_goal":
            decision["status"] = "new_route_not_known_free"
            return result
        start_estimated = self.estimate[:2].copy()
        start_actual = self.actual.copy()
        self.estimated_path = np.asarray(
            [start_estimated] + [self.grid.xy(cell) for cell in path[1:]])
        self.actual_path = (start_actual +
                            (self.estimated_path - start_estimated) *
                            (1. + .30 / 5.89))
        self.path_length = float(np.sum(np.linalg.norm(
            np.diff(self.actual_path, axis=0), axis=1)))
        self.dispatch_time_s = time_s
        self.stop_time_s = None
        self.response_goal = proposed.copy()
        self.last_replan_s = time_s
        decision["status"] = "switched_to_fresh_route"
        decision["new_path_xy_m"] = self.estimated_path.tolist()
        return result


def main(*, prop_factory=turn.ExitTurnProp, out=OUT,
         profile="post-cue turn toward V747 northwest response"):
    out.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-adaptive-replan.v1",
                   "control_report_sha256": route.sha256(emergency.OUT / "report.json"),
                   "held_out_turn_report_sha256": route.sha256(turn.OUT / "report.json"),
                   "changed_variable": "On each new 10Hz capture after initial response, forecast the current route and radial alternatives from the same actor range track; switch only if predicted lower gap improves by >=8cm and the mapped route and raw guard permit it.",
                   "minimum_replan_interval_s": MIN_REPLAN_INTERVAL_S,
                   "minimum_replan_gain_m": MIN_REPLAN_GAIN_M,
                   "profile": profile,
                   "source_sha256": route.sha256(SOURCE),
                   "selector_source_sha256": route.sha256(Path(selector.__file__))}
    (out / "replan-declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    adaptive.main(prop_factory=prop_factory, out=out, profile=profile,
                  monitor_factory=ReplanningEmergencyMonitor)
    path = out / "report.json"
    report = json.loads(path.read_text())
    report["replan_declaration_sha256"] = route.sha256(out / "replan-declaration.json")
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
