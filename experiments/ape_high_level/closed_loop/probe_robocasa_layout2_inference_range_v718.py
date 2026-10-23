"""Paired virtual range coverage at V716's captured owner pose."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import run_robocasa_layout2_owner_inference_monitor_v715 as monitor
import run_robocasa_layout2_owner_inference_trigger_v716 as trial
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_side_prop_v701 as side
from run_robocasa_scan_height_v433 import height_scan


SOURCE = Path(__file__)
OUT = trial.OUT / "range-coverage-v718.json"


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    ep = report["episode"]
    guard = next(row for row in ep["trace"] if row.get("stage") ==
                 "owner_inference_near_body_monitor")
    capture = report["capture_trigger_time_s"]
    xy = np.asarray(ep["final_actual_xy_m"])
    model = person.scene_with_person(side.START, layout_id=2, style_id=1,
                                     clutter_mode=0)
    torso = interaction.TORSO_XY + [-.3, 0.]
    hand = interaction.SHOW_HAND + [-.3, 0., 0.]
    person.set_owner_pose(model, torso, hand, show_can=True)
    direction = (hand[:2] - torso) / np.linalg.norm(hand[:2] - torso)
    model.body_pos[model.body("shown_can").id, :2] = hand[:2] + .14 * direction
    data = mujoco.MjData(model)
    prop_body = model.body("route_obstacle").id
    rows = []
    for tick in range(25):
        t = capture + tick * .1
        fraction = np.clip((t - capture - .15) / 2.5, 0., 1.)
        prop_xy = side.START + fraction * (monitor.END - side.START)
        record = {"time_s": round(t, 4), "prop_xy_evaluator_m": prop_xy.tolist(),
                  "paired_counterfactuals": {}}
        for label, heights in (("low_5cm", (.05,)), ("mid_30cm", (.30,)),
                               ("fused_5_30cm", (.05, .30))):
            scan = height_scan(heights)
            packets = []
            for prop in (side.START, prop_xy):
                model.body_pos[prop_body, :2] = prop
                mujoco.mj_forward(model, data)
                packets.append(scan(model, data, xy, 5.,
                                    np.random.default_rng(71800 + tick)))
            stationary, moving = packets
            shorter = [index for index, (a, b) in enumerate(zip(stationary, moving))
                       if b is not None and (a is None or a - b >= .10)]
            record["paired_counterfactuals"][label] = {
                "stationary_72_beams_m": stationary,
                "moving_72_beams_m": moving,
                "moving_shorter_by_at_least_10cm_beams": shorter,
                "changed_beam_count": len(shorter)}
        rows.append(record)
    summary = {}
    for label in ("low_5cm", "mid_30cm", "fused_5_30cm"):
        first_one = next((row for row in rows if row["paired_counterfactuals"][label]
                          ["changed_beam_count"] >= 1), None)
        first_three = next((row for row in rows if row["paired_counterfactuals"][label]
                            ["changed_beam_count"] >= 3), None)
        summary[label] = {"first_one_beam_time_s": None if first_one is None else first_one["time_s"],
                          "first_three_beams_time_s": None if first_three is None else first_three["time_s"]}
    result = {"schema": "robot-learning-lab.layout2-inference-range-coverage.v1",
              "source_sha256": route.sha256(SOURCE),
              "trial_report_sha256": route.sha256(trial.OUT / "report.json"),
              "fixed_actor_pose_evaluator_m": xy.tolist(),
              "capture_time_s": capture,
              "actor_raw_nearest_alarm_time_s": guard["first_alarm_time_s"],
              "paired_diagnostic_not_actor_track": True,
              "scan_heights_m": {"low_5cm": [.05], "mid_30cm": [.30],
                                 "fused_5_30cm": [.05, .30]},
              "summary": summary, "rows": rows}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
