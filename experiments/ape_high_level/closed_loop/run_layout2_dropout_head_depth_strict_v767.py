"""Use depth warning without dispatching an unsupported escape route."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import actor_range_refuge_v744 as selector
import run_layout2_adaptive_refuge_v744 as adaptive
import run_layout2_adaptive_exit_turn_v749 as turn
import run_layout2_dropout_head_depth_response_v765 as previous
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route


SOURCE = Path(__file__)
OUT = previous.dropout.trial.adaptive.west.side.ROOT / "artifacts/ape-robocasa-layout2-dropout-head-depth-strict-v767"


class StrictDepthMonitor(previous.DepthOffMonitor):
    output_branch = OUT

    def __call__(self, t):
        result = super().__call__(t)
        motion = result.get("actor_head_depth_motion")
        if (self.trigger_time_s is not None or motion is None or
                not motion["would_interrupt"]):
            return result
        center = self.depth_observations[-1][1]
        velocity = np.asarray(motion["velocity_xy_m_s"])
        track = {"id": self.head_target_track["id"],
                 "status": "confirmed_now",
                 "center_xy_m": center.tolist(),
                 "radius_m": self.head_target_track["radius_m"]}
        goal, evidence = selector.choose(
            self.grid, self.estimate[:2], track, velocity, full.FIRST_SURFACE,
            allow_best_effort=False)
        result["actor_head_depth_decision"] = {
            "status": evidence["status"],
            "candidate_count": evidence["candidate_count"],
            "supported_candidate_count": evidence["supported_candidate_count"],
            "chosen_goal_xy_m": evidence["chosen_goal_xy_m"],
            "depth_threat_but_range_unhealthy": True}
        if goal is None:
            return result
        requested, status, path = route.SensorGrid.plan(
            self.grid, self.estimate[:2], goal)
        if requested is None or status != "known_goal":
            result["actor_head_depth_decision"]["status"] = "route_not_known_free"
            return result
        self.trigger_time_s = t
        self.hazard_id = track["id"]
        self.hazard_centers = self.depth_observations.copy()
        self.start_estimated = self.estimate[:2].copy()
        self.start_actual = self.actual.copy()
        self.response_goal = goal.copy()
        self.plan_status = status
        self.estimated_path = np.asarray(
            [self.start_estimated] + [self.grid.xy(cell) for cell in path[1:]])
        self.actual_path = (self.start_actual +
                            (self.estimated_path - self.start_estimated) *
                            selector.ROUTE_SCALE)
        self.path_length = float(np.sum(np.linalg.norm(
            np.diff(self.actual_path, axis=0), axis=1)))
        self.dispatch_time_s = t
        result.update({"actor_response_trigger_time_s": t,
                       "actor_response_source": "supported_head_depth_during_range_dropout",
                       "actor_goal_xy_m": goal.tolist(),
                       "actor_response_selection": evidence,
                       "actor_plan_status": status,
                       "actor_response_qualification": evidence["status"],
                       "actor_estimated_path_xy_m": self.estimated_path.tolist(),
                       "invalidate_image": True})
        return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {
        "schema": "robot-learning-lab.layout2-dropout-head-depth-strict.v1",
        "comparison_report_sha256": route.sha256(previous.OUT / "report.json")
            if (previous.OUT / "report.json").exists() else None,
        "depth_on_report_sha256": route.sha256(previous.OUT / "depth_on/report.json"),
        "depth_off_report_sha256": route.sha256(previous.OUT / "depth_off/report.json"),
        "same_0p8s_range_tracker_outage": True,
        "changed_variable": "A head-depth warning rejects the item image but dispatches only if at least one timed route and held pose meet all V744 hypotheses; otherwise keep monitoring for a fresh range-supported continuation.",
        "source_sha256": route.sha256(SOURCE)}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    adaptive.main(prop_factory=turn.ExitTurnProp, out=OUT,
                  profile="V765 depth sensor with strict supported-route handoff",
                  monitor_factory=StrictDepthMonitor)
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["strict_depth_declaration_sha256"] = route.sha256(OUT / "predeclared.json")
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
