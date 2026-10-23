"""Full west-station owner turn with actor-only projected near-body track gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_layout2_person_associated_station_v731 as associated
import run_layout2_west_station_baseline_v729 as west
import run_robocasa_layout2_joint_station_v712 as station
import run_robocasa_layout2_live_inference_escape_v721 as baseline
import run_robocasa_layout2_owner_inference_trigger_v716 as trigger
import run_robocasa_layout2_owner_transfer_v695 as transfer
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side
import run_robocasa_person_joined_v497 as joined


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-west-station-threat-gate-v732"
HORIZON_S = 2.5
POSE_AND_TRACK_RESERVE_M = .10
GAP_M = .05


class ThreatMonitor(baseline.LiveMonitor):
    def __init__(self, model, data, actual, script, grid, estimate):
        super().__init__(model, data, actual, script, grid, estimate)
        self.admission_gate = self.admit_interrupt

    def admit_interrupt(self, track, time_s, result):
        if track is None or len(self.centers) < 3 or self.centers[-3][1] is None:
            result["actor_projected_threat_state"] = "insufficient_track"
            return False
        t0, p0 = self.centers[-3]
        velocity = (self.centers[-1][1] - p0) / max(.001, time_s - t0)
        relative = self.centers[-1][1] - self.estimate[:2]
        speed2 = float(np.dot(velocity, velocity))
        closest_t = (float(np.clip(-np.dot(relative, velocity) / speed2,
                                   0., HORIZON_S)) if speed2 > 1.e-6 else 0.)
        closest_center = self.centers[-1][1] + closest_t * velocity
        clearance = (float(np.linalg.norm(closest_center - self.estimate[:2])) -
                     float(track["radius_m"]) - .17)
        lower = clearance - POSE_AND_TRACK_RESERVE_M
        result.update({"actor_projected_threat_state":
                           "interrupt" if lower < GAP_M else "observed_clear",
                       "actor_projected_nearest_time_s": closest_t,
                       "actor_projected_gap_lower_m": lower,
                       "actor_track_velocity_xy_m_s": velocity.tolist(),
                       "actor_prediction_horizon_s": HORIZON_S})
        return lower < GAP_M


def main(*, prop_factory=trigger.CaptureTriggeredProp, out=OUT,
         profile="item-image-capture-triggered straight side prop",
         monitor_factory=ThreatMonitor):
    out.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-west-station-threat-gate.v1",
                   "station_rule": {"standoff_m": west.STANDOFF_M,
                                    "lateral_m": west.LATERAL_M},
                   "actor_person_association": "V731 first RGB-D person surface to range object <=0.45m",
                   "interrupt_gate": "two 10Hz track motion estimates >0.45m/s and 2.5s constant-velocity closest approach below 5cm modeled body gap plus 10cm uncalibrated pose/track reserve",
                   "profile": profile,
                   "monitor_factory": monitor_factory.__name__,
                   "limitation": "Constant velocity, virtual 72 beams, fixed 17cm surrogate and scripted prop; not a future-occupancy certificate for arbitrary turns.",
                   "source_sha256": route.sha256(SOURCE)}
    (out / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    prop = prop_factory()
    old = (full.OUT, full.PROP_SCRIPT_FACTORY, full.RUN_SCOPE,
           full.POST_STOP_UNTIL_S, full.STANDOFF_M, full.STATION_LATERAL_M,
           full.SEED_ROUTE_FROM_SURVEY, full.ACTIVE_VIEW_CORRECTION,
           transfer.PROP_START.copy(), joined.grid_class,
           joined.INFERENCE_MONITOR_FACTORY, joined.CAPTURE_EVENT_FACTORY,
           joined.INTERRUPTED_HOLD_UNTIL_S)
    try:
        full.OUT = out
        full.PROP_SCRIPT_FACTORY = lambda _: prop
        full.RUN_SCOPE = "layout2 full owner moving-prop turn at west station; actor person/range association and live 10Hz projected track-threat gate; 50cm kinematic proxy"
        full.POST_STOP_UNTIL_S = station.HORIZON_S
        full.STANDOFF_M, full.STATION_LATERAL_M = west.STANDOFF_M, west.LATERAL_M
        full.SEED_ROUTE_FROM_SURVEY = True
        full.ACTIVE_VIEW_CORRECTION = True
        transfer.PROP_START = side.START.copy()
        joined.grid_class = associated.associated_grid_class
        joined.INFERENCE_MONITOR_FACTORY = monitor_factory
        joined.CAPTURE_EVENT_FACTORY = lambda script: script.arm
        joined.INTERRUPTED_HOLD_UNTIL_S = station.HORIZON_S
        full.main()
    finally:
        (full.OUT, full.PROP_SCRIPT_FACTORY, full.RUN_SCOPE,
         full.POST_STOP_UNTIL_S, full.STANDOFF_M, full.STATION_LATERAL_M,
         full.SEED_ROUTE_FROM_SURVEY, full.ACTIVE_VIEW_CORRECTION,
         transfer.PROP_START, joined.grid_class,
         joined.INFERENCE_MONITOR_FACTORY, joined.CAPTURE_EVENT_FACTORY,
         joined.INTERRUPTED_HOLD_UNTIL_S) = old
    report_path = out / "report.json"
    report = json.loads(report_path.read_text())
    report["capture_trigger_time_s"] = prop.capture_time_s
    report["predeclaration_sha256"] = route.sha256(out / "predeclared.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    ep = report["episode"]
    print(ep["route_state"], ep["outcome"], ep["route_episode"]["end_true_xy_m"],
          ep["final_actual_xy_m"], flush=True)


if __name__ == "__main__":
    main()
