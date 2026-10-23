"""Live actor-selected refuge during the full west-station owner episode."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import actor_range_refuge_v744 as selector
import run_layout2_west_station_threat_gate_v732 as west
import run_robocasa_layout2_owner_inference_trigger_v716 as trigger
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined


SOURCE = Path(__file__)
OUT = west.side.ROOT / "artifacts/ape-robocasa-layout2-adaptive-refuge-v744"


class AdaptiveThreatMonitor(west.ThreatMonitor):
    allow_best_effort = False

    def __init__(self, model, data, actual, script, grid, estimate):
        super().__init__(model, data, actual, script, grid, estimate)
        self.response_selector = self.select_response

    def select_response(self, track, time_s, result):
        if track is None or "actor_track_velocity_xy_m_s" not in result:
            return None, {"status": "track_velocity_missing",
                          "capture_time_s": time_s}
        goal, evidence = selector.choose(
            self.grid, self.estimate[:2], track,
            result["actor_track_velocity_xy_m_s"], full.FIRST_SURFACE,
            allow_best_effort=self.allow_best_effort)
        evidence["capture_time_s"] = time_s
        evidence["actor_pose_xy_m"] = self.estimate[:2].tolist()
        evidence["actor_track_id"] = track["id"]
        return goal, evidence


def main(*, prop_factory=trigger.CaptureTriggeredProp, out=OUT,
         profile="capture-triggered straight side prop",
         monitor_factory=AdaptiveThreatMonitor):
    out.mkdir(parents=True, exist_ok=True)
    declaration = {
        "schema": "robot-learning-lab.layout2-adaptive-refuge.v1",
        "profile": profile,
        "station": "V731 RGB-D-person-associated west station",
        "motion_warning": "V732 two 10Hz moving-track warnings and closest-approach gate",
        "candidate_rule": "actor-known-free radial cells at 0.45/0.65/0.85m, one grid search, timed 0.30m/s route and renewable 0.80s hold",
        "hazard_hypotheses": ["straight", "1.25x straight", "stop after 0.7s",
                              "±45° turn after 0.7s"],
        "pose_track_reserve_m": selector.POSE_TRACK_RESERVE_M,
        "dispatch_delay_s": selector.DISPATCH_DELAY_S,
        "monitor_continues_after_image_worker": True,
        "allow_unqualified_best_effort": monitor_factory.allow_best_effort,
        "no_supported_goal": "invalidate captured item image, report no supported response, continue observing without calling a free cell safe",
        "development_notice": "Hypotheses and reserve are uncalibrated; candidate structure was devised after V736/V738 south-exit failures and V742 north development success.",
        "source_sha256": route.sha256(SOURCE),
        "selector_source_sha256": route.sha256(Path(selector.__file__)),
        "monitor_source_sha256": route.sha256(Path(west.__file__))}
    (out / "adaptive-declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    previous = joined.CONTINUE_INTERRUPTED_MONITOR
    try:
        joined.CONTINUE_INTERRUPTED_MONITOR = True
        west.main(prop_factory=prop_factory, out=out, profile=profile,
                  monitor_factory=monitor_factory)
    finally:
        joined.CONTINUE_INTERRUPTED_MONITOR = previous
    path = out / "report.json"
    report = json.loads(path.read_text())
    report["adaptive_declaration_sha256"] = route.sha256(out / "adaptive-declaration.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    ep = report["episode"]
    guard = next(row for row in ep["trace"] if row.get("stage") ==
                 "owner_inference_near_body_monitor")
    response = next((row for row in guard["monitor_samples"] if
                     "actor_response_trigger_time_s" in row), None)
    print(ep["outcome"], ep["final_actual_xy_m"],
          None if response is None else response["actor_response_selection"]["status"],
          None if response is None else response["actor_goal_xy_m"], flush=True)


if __name__ == "__main__":
    main()
