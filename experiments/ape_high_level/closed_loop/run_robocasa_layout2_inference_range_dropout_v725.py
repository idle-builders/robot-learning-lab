"""Withhold early 10Hz range-track packets during V721's owner inference."""

from __future__ import annotations

import json
from pathlib import Path

import run_robocasa_layout2_joint_station_v712 as station
import run_robocasa_layout2_live_inference_escape_v721 as baseline
import run_robocasa_layout2_owner_transfer_v695 as transfer
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side
import run_robocasa_person_joined_v497 as joined


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-inference-range-dropout-v725"
DROP_UNTIL_AFTER_CAPTURE_S = .8


class DropoutMonitor(baseline.LiveMonitor):
    def __init__(self, model, data, actual, script, grid, estimate):
        super().__init__(model, data, actual, script, grid, estimate)
        self.resumed = False

    def __call__(self, t):
        capture = self.script.capture_time_s
        assert capture is not None
        missing = t - capture < DROP_UNTIL_AFTER_CAPTURE_S
        if missing:
            self.next_scan_s = t + .1
        elif not self.resumed:
            self.next_scan_s = t
            self.resumed = True
        result = super().__call__(t)
        result["range_track_stream_health"] = (
            "missing" if missing else "healthy")
        return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-inference-range-dropout.v1",
                   "control_report_sha256": route.sha256(
                       baseline.OUT / "moving/report.json"),
                   "same_development_exit_and_speed_source_sha256": route.sha256(
                       Path(baseline.__file__)),
                   "missing_10hz_track_from_capture_until_s": DROP_UNTIL_AFTER_CAPTURE_S,
                   "independent_20ms_raw_nearest_monitor_retained": True,
                   "event_relative_prop_path": "same straight side path as V721",
                   "source_sha256": route.sha256(SOURCE)}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    prop = baseline.Prop(True)
    old = (full.OUT, full.PROP_SCRIPT_FACTORY, full.RUN_SCOPE,
           full.POST_STOP_UNTIL_S, full.STANDOFF_M, full.STATION_LATERAL_M,
           full.SEED_ROUTE_FROM_SURVEY, full.ACTIVE_VIEW_CORRECTION,
           transfer.PROP_START.copy(), joined.INFERENCE_MONITOR_FACTORY,
           joined.CAPTURE_EVENT_FACTORY, joined.INTERRUPTED_HOLD_UNTIL_S)
    try:
        full.OUT = OUT
        full.PROP_SCRIPT_FACTORY = lambda _: prop
        full.RUN_SCOPE = "layout2 full owner task with missing first 0.8s 10Hz inference tracker packets, separate raw 20ms guard retained; 50cm kinematic proxy"
        full.POST_STOP_UNTIL_S = station.HORIZON_S
        full.STANDOFF_M = station.STANDOFF_M
        full.STATION_LATERAL_M = station.LATERAL_M
        full.SEED_ROUTE_FROM_SURVEY = True
        full.ACTIVE_VIEW_CORRECTION = True
        transfer.PROP_START = side.START.copy()
        joined.INFERENCE_MONITOR_FACTORY = DropoutMonitor
        joined.CAPTURE_EVENT_FACTORY = lambda script: script.arm
        joined.INTERRUPTED_HOLD_UNTIL_S = station.HORIZON_S
        full.main()
    finally:
        (full.OUT, full.PROP_SCRIPT_FACTORY, full.RUN_SCOPE,
         full.POST_STOP_UNTIL_S, full.STANDOFF_M, full.STATION_LATERAL_M,
         full.SEED_ROUTE_FROM_SURVEY, full.ACTIVE_VIEW_CORRECTION,
         transfer.PROP_START, joined.INFERENCE_MONITOR_FACTORY,
         joined.CAPTURE_EVENT_FACTORY, joined.INTERRUPTED_HOLD_UNTIL_S) = old
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["capture_trigger_time_s"] = prop.capture_time_s
    report["dropout_source_sha256"] = route.sha256(SOURCE)
    report["predeclaration_sha256"] = route.sha256(OUT / "predeclared.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    ep = report["episode"]
    guard = next(row for row in ep["trace"] if row.get("stage") ==
                 "owner_inference_near_body_monitor")
    trigger = next((row for row in guard["monitor_samples"] if
                    "actor_response_trigger_time_s" in row), None)
    print(ep["route_state"], ep["outcome"],
          None if trigger is None else trigger["actor_response_trigger_time_s"],
          ep["final_actual_xy_m"], flush=True)


if __name__ == "__main__":
    main()
