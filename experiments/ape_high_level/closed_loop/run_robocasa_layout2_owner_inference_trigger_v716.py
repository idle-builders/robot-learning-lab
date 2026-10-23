"""Trigger a fixture-clear side prop from the owner item's image capture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_joint_station_v712 as station
import run_robocasa_layout2_owner_transfer_v695 as transfer
import run_robocasa_layout2_owner_inference_monitor_v715 as monitor
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_side_prop_v701 as side
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout2_route_v407 as route


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-owner-inference-trigger-v716"


class CaptureTriggeredProp:
    def __init__(self):
        self.capture_time_s = None

    def arm(self, capture_time_s):
        if self.capture_time_s is None:
            self.capture_time_s = capture_time_s

    def __call__(self, t):
        if self.capture_time_s is None:
            return side.START.copy()
        fraction = np.clip((t - self.capture_time_s - .15) / 2.5, 0., 1.)
        return side.START + fraction * (monitor.END - side.START)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {"schema": "robot-learning-lab.layout2-capture-triggered-prop.v1",
                   "same_station_as": str(station.OUT / "report.json"),
                   "path": "same fixture-clear start/end as V715",
                   "motion_begin_relative_to_owner_item_capture_s": .15,
                   "motion_duration_s": 2.5,
                   "actor_does_not_receive_trigger_time": True,
                   "test": "full owner episode with a moving low prop during image inference; main thread must sample range while image worker is busy and invalidate stale item evidence on warning",
                   "source_sha256": route.sha256(SOURCE)}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    prop = CaptureTriggeredProp()
    old = (full.OUT, full.PROP_SCRIPT_FACTORY, full.RUN_SCOPE,
           full.POST_STOP_UNTIL_S, full.STANDOFF_M, full.STATION_LATERAL_M,
           full.SEED_ROUTE_FROM_SURVEY, full.ACTIVE_VIEW_CORRECTION,
           transfer.PROP_START.copy(), joined.INFERENCE_MONITOR_FACTORY,
           joined.CAPTURE_EVENT_FACTORY,
           joined.INTERRUPTED_HOLD_UNTIL_S)
    try:
        full.OUT = OUT
        full.PROP_SCRIPT_FACTORY = lambda _: prop
        full.RUN_SCOPE = "layout2 full owner station, item-capture-triggered low prop and concurrent 20ms raw near-body monitor; 50cm kinematic proxy"
        full.POST_STOP_UNTIL_S = station.HORIZON_S
        full.STANDOFF_M = station.STANDOFF_M
        full.STATION_LATERAL_M = station.LATERAL_M
        full.SEED_ROUTE_FROM_SURVEY = True
        full.ACTIVE_VIEW_CORRECTION = True
        transfer.PROP_START = side.START.copy()
        joined.INFERENCE_MONITOR_FACTORY = monitor.inference_monitor
        joined.CAPTURE_EVENT_FACTORY = lambda script: script.arm
        joined.INTERRUPTED_HOLD_UNTIL_S = station.HORIZON_S
        full.main()
    finally:
        (full.OUT, full.PROP_SCRIPT_FACTORY, full.RUN_SCOPE,
         full.POST_STOP_UNTIL_S, full.STANDOFF_M, full.STATION_LATERAL_M,
         full.SEED_ROUTE_FROM_SURVEY, full.ACTIVE_VIEW_CORRECTION,
         transfer.PROP_START, joined.INFERENCE_MONITOR_FACTORY,
         joined.CAPTURE_EVENT_FACTORY,
         joined.INTERRUPTED_HOLD_UNTIL_S) = old
    path = OUT / "report.json"
    report = json.loads(path.read_text())
    report["capture_trigger_time_s"] = prop.capture_time_s
    report["moving_prop_source_sha256"] = route.sha256(SOURCE)
    report["predeclaration_sha256"] = route.sha256(OUT / "predeclared.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    ep = report["episode"]
    guard = next((row for row in ep["trace"] if row.get("stage") ==
                  "owner_inference_near_body_monitor"), None)
    print(ep["route_state"], ep["outcome"],
          None if guard is None else guard["first_alarm_time_s"],
          ep["hold_until_s"], flush=True)


if __name__ == "__main__":
    main()
