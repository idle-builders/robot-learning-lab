"""Retimed low prop during the slow owner-item recognition in layout 2."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from audit_robocasa_dynamic_suite_v440 import prop_fixture_gap
import robocasa_owner_person as person
import run_robocasa_layout2_joint_station_v712 as station
import run_robocasa_layout2_owner_transfer_v695 as transfer
import run_robocasa_layout2_person_route_v697 as full
import run_robocasa_layout2_side_prop_v701 as side
import run_robocasa_moving_prop_v423 as moving
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout2_route_v407 as route


SOURCE = Path(__file__)
OUT = side.ROOT / "artifacts/ape-robocasa-layout2-owner-inference-monitor-v715"
BEGIN_S = 13.3
END_S = 17.3
END = np.array([.8, -3.5])


def prop_at(t):
    fraction = np.clip((t - BEGIN_S) / (END_S - BEGIN_S), 0., 1.)
    return side.START + fraction * (END - side.START)


def inference_monitor(model, data, actual, script, **_):
    body = model.body("route_obstacle").id
    head = model.camera("head").id

    def sample(t):
        model.body_pos[body, :2] = script(t)
        model.cam_pos[head, :2] = actual
        mujoco.mj_forward(model, data)
        return moving.raw_nearest_range(model, data, actual)

    return sample


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = person.scene_with_person(side.START, layout_id=2, style_id=1,
                                     clutter_mode=0)
    data = mujoco.MjData(model)
    body = model.body("route_obstacle").id
    geom = model.geom("route_obstacle_geom").id
    ape = model.geom("audit_ape_geom").id
    fixture = []
    for t in np.arange(BEGIN_S, END_S + .0001, .02):
        xy = prop_at(float(t))
        model.body_pos[body, :2] = xy
        mujoco.mj_forward(model, data)
        gap, near = prop_fixture_gap(model, data, geom, ape, xy)
        fixture.append((gap, float(t), near))
    minimum = min(fixture)
    assert minimum[0] >= .05
    declaration = {"schema": "robot-learning-lab.layout2-owner-inference-monitor.v1",
                   "same_station_as": str(station.OUT / "report.json"),
                   "scene": "RoboCasa layout2/style1, RGB-D person station with one active correction",
                   "side_prop_start_xy_m": side.START.tolist(),
                   "side_prop_end_xy_m": END.tolist(),
                   "motion_begin_s": BEGIN_S, "motion_end_s": END_S,
                   "fixture_minimum_gap_m": minimum[0],
                   "comparison": "Static prop versus prop entering the held owner's station during the slow taught-item worker. Main thread samples raw near-body range at 20ms wall cadence, advances scripted scene, and invalidates a captured image on alarm. No escape skill yet.",
                   "source_sha256": route.sha256(SOURCE)}
    (OUT / "predeclared.json").write_text(json.dumps(declaration, indent=2) + "\n")
    old = (full.OUT, full.PROP_SCRIPT_FACTORY, full.RUN_SCOPE,
           full.POST_STOP_UNTIL_S, full.STANDOFF_M, full.STATION_LATERAL_M,
           full.SEED_ROUTE_FROM_SURVEY, full.ACTIVE_VIEW_CORRECTION,
           transfer.PROP_START.copy(), joined.INFERENCE_MONITOR_FACTORY,
           joined.INTERRUPTED_HOLD_UNTIL_S)
    try:
        full.POST_STOP_UNTIL_S = station.HORIZON_S
        full.STANDOFF_M = station.STANDOFF_M
        full.STATION_LATERAL_M = station.LATERAL_M
        full.SEED_ROUTE_FROM_SURVEY = True
        full.ACTIVE_VIEW_CORRECTION = True
        transfer.PROP_START = side.START.copy()
        joined.INFERENCE_MONITOR_FACTORY = inference_monitor
        joined.INTERRUPTED_HOLD_UNTIL_S = station.HORIZON_S
        full.RUN_SCOPE = "layout2 full owner station, concurrent item inference and independent 20ms raw range monitor; static/moving paired; 50cm kinematic proxy"
        for label, script in (("static", lambda t: side.START.copy()),
                              ("moving", prop_at)):
            full.OUT = OUT / label
            full.PROP_SCRIPT_FACTORY = lambda _, script=script: script
            full.main()
            report = json.loads((full.OUT / "report.json").read_text())
            ep = report["episode"]
            monitor = next((row for row in ep["trace"] if row.get("stage") ==
                            "owner_inference_near_body_monitor"), None)
            print(label, ep["route_state"], ep["outcome"],
                  None if monitor is None else monitor["first_alarm_time_s"],
                  ep["hold_until_s"], flush=True)
    finally:
        (full.OUT, full.PROP_SCRIPT_FACTORY, full.RUN_SCOPE,
         full.POST_STOP_UNTIL_S, full.STANDOFF_M, full.STATION_LATERAL_M,
         full.SEED_ROUTE_FROM_SURVEY, full.ACTIVE_VIEW_CORRECTION,
         transfer.PROP_START, joined.INFERENCE_MONITOR_FACTORY,
         joined.INTERRUPTED_HOLD_UNTIL_S) = old


if __name__ == "__main__":
    main()
