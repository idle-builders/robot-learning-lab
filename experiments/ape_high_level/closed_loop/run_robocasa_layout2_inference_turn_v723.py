"""Held-out prop turn toward V721's development exit after the actor cue."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from audit_robocasa_dynamic_suite_v440 import prop_fixture_gap
import robocasa_owner_person as person
import run_robocasa_layout2_joint_station_v712 as station
import run_robocasa_layout2_live_inference_escape_v721 as baseline
import run_robocasa_layout2_owner_inference_monitor_v715 as monitor
import run_robocasa_layout2_owner_transfer_v695 as transfer
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side
import run_robocasa_person_joined_v497 as joined


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-inference-turn-v723"
TURN_AFTER_CAPTURE_S = .8
FINAL_AFTER_CAPTURE_S = 2.65
END = np.array([1., -3.9])
MID = side.START + ((TURN_AFTER_CAPTURE_S - .15) / 2.5) * (
    monitor.END - side.START)


class TurnProp(baseline.Prop):
    def __init__(self):
        super().__init__(True)

    def __call__(self, t):
        if self.capture_time_s is None:
            return side.START.copy()
        elapsed = t - self.capture_time_s
        if elapsed <= TURN_AFTER_CAPTURE_S:
            return super().__call__(t)
        fraction = np.clip((elapsed - TURN_AFTER_CAPTURE_S) /
                           (FINAL_AFTER_CAPTURE_S - TURN_AFTER_CAPTURE_S), 0., 1.)
        return MID + fraction * (END - MID)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = person.scene_with_person(side.START, layout_id=2, style_id=1,
                                     clutter_mode=0)
    data = mujoco.MjData(model)
    prop_body = model.body("route_obstacle").id
    prop_geom = model.geom("route_obstacle_geom").id
    ape_geom = model.geom("audit_ape_geom").id
    gaps = []
    for elapsed in np.arange(0., FINAL_AFTER_CAPTURE_S + .01, .02):
        trial = TurnProp()
        trial.arm(0.)
        xy = trial(float(elapsed))
        model.body_pos[prop_body, :2] = xy
        mujoco.mj_forward(model, data)
        gap, near = prop_fixture_gap(model, data, prop_geom, ape_geom, xy)
        gaps.append((gap, float(elapsed), near))
    nearest = min(gaps)
    assert nearest[0] >= .05
    declaration = {"schema": "robot-learning-lab.layout2-inference-turn.v1",
                   "control_report_sha256": route.sha256(
                       baseline.OUT / "moving/report.json"),
                   "same_policy_source_sha256": route.sha256(
                       Path(baseline.__file__)),
                   "same_prop_path_until_after_capture_s": TURN_AFTER_CAPTURE_S,
                   "prior_actor_cue_after_capture_s": .4071,
                   "turned_prop_endpoint_xy_m": END.tolist(),
                   "turned_prop_arrival_after_capture_s": FINAL_AFTER_CAPTURE_S,
                   "fixture_minimum_gap_m_capped_at_30cm": nearest[0],
                   "held_out_bearing": "chosen after freezing V721 development exit, before this full run",
                   "source_sha256": route.sha256(SOURCE)}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    prop = TurnProp()
    old = (full.OUT, full.PROP_SCRIPT_FACTORY, full.RUN_SCOPE,
           full.POST_STOP_UNTIL_S, full.STANDOFF_M, full.STATION_LATERAL_M,
           full.SEED_ROUTE_FROM_SURVEY, full.ACTIVE_VIEW_CORRECTION,
           transfer.PROP_START.copy(), joined.INFERENCE_MONITOR_FACTORY,
           joined.CAPTURE_EVENT_FACTORY, joined.INTERRUPTED_HOLD_UNTIL_S)
    try:
        full.OUT = OUT
        full.PROP_SCRIPT_FACTORY = lambda _: prop
        full.RUN_SCOPE = "layout2 full owner task, same V721 live inference track/response with post-cue fixture-clear prop turn toward development exit; 50cm proxy"
        full.POST_STOP_UNTIL_S = station.HORIZON_S
        full.STANDOFF_M = station.STANDOFF_M
        full.STATION_LATERAL_M = station.LATERAL_M
        full.SEED_ROUTE_FROM_SURVEY = True
        full.ACTIVE_VIEW_CORRECTION = True
        transfer.PROP_START = side.START.copy()
        joined.INFERENCE_MONITOR_FACTORY = baseline.LiveMonitor
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
    report["predeclaration_sha256"] = route.sha256(OUT / "predeclared.json")
    report["turn_source_sha256"] = route.sha256(SOURCE)
    path.write_text(json.dumps(report, indent=2) + "\n")
    ep = report["episode"]
    guard = next(row for row in ep["trace"] if row.get("stage") ==
                 "owner_inference_near_body_monitor")
    print(ep["route_state"], ep["outcome"], guard["first_alarm_time_s"],
          ep["final_actual_xy_m"], flush=True)


if __name__ == "__main__":
    main()
