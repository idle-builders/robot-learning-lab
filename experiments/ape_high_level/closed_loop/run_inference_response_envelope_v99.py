"""V99: evaluator-only whole-body motion screen from V98's inference alarm.

This checks whether a bounded response even exists before asking actor sensors
to authorize it. Candidate direction is chosen by the experimenter, never by
future scene geometry in the owner controller.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from articulated_person_scene import FAR_HAND_XYZ_M, add_person, set_person_pose
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v99"
V98 = ROOT / "artifacts/ape-closed-loop-cove50-v98"
ALARM_TIME_S = 17.4
END_TIME_S = 18.0
STEP_M = .08
SLIP = .55
DIRECTIONS = {
    "hold": (0., 0.),
    "right": (0., -1.),
    "left": (0., 1.),
    "back": (-1., 0.),
    "forward": (1., 0.),
    "back_right": (-math.sqrt(.5), -math.sqrt(.5)),
    "forward_right": (math.sqrt(.5), -math.sqrt(.5)),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rows = []
    try:
        for seed in (170, 172):
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            packet_path = (V98 / "moving" / f"seed{seed}-present" /
                           f"seed{seed}-present/owner-guard-12.npz")
            with np.load(packet_path) as packet:
                qpos = packet["actor_qpos"].copy()
                estimate = packet["estimated_pose_xyyaw"].copy()
                time_s = float(packet["capture_time_s"])
            if abs(time_s - ALARM_TIME_S) > 1e-5:
                raise AssertionError("V98 alarm packet time changed")
            actual = np.array((qpos[0], qpos[1],
                               2*math.atan2(qpos[6], qpos[3])))
            for case in CASES:
                room = OfferDriftRoom(case=tuple(prior["case"]),
                                      pose_source="command", branch=OUT / "scene")
                try:
                    room.layout(seed, absent=False, furnished=True)
                    obstacle = CASES[case]
                    if obstacle is not None:
                        position, size = obstacle
                        room.model.body_pos[room.model.body("loop_box").id] = position
                        room.model.geom("loop_box_geom").size[:] = size
                    room.data.qpos[:] = qpos
                    room.estimated[:] = estimate
                    room.head = float(qpos[room.head_addr])
                    set_person_pose(room.model, FAR_HAND_XYZ_M,
                                    carried_item="loop_TomatoSauce")
                    for direction, (dx, dy) in DIRECTIONS.items():
                        path = []
                        minimum = (float("inf"), None, None, None)
                        first_below = None
                        for tick in range(31):
                            time_s = ALARM_TIME_S + .02*tick
                            fraction = min(1., (time_s-ALARM_TIME_S)/.4)
                            room.actual[:] = actual
                            room.actual[0] += STEP_M*SLIP*dx*fraction
                            room.actual[1] += STEP_M*SLIP*dy*fraction
                            room.time_s = time_s
                            prop_fraction = min(1., max(0.,
                                (time_s-v98["delayed_prop"]["start_s"]) /
                                v98["delayed_prop"]["duration_s"]))
                            y = (v98["delayed_prop"]["start_y_m"] +
                                 prop_fraction*(v98["delayed_prop"]["end_y_m"]-
                                                v98["delayed_prop"]["start_y_m"]))
                            room.obstacle_xy[3] = (0., y)
                            room.model.body_pos[room.model.body(
                                "loop_stool").id] = (0., y, STOOL_SIZE[2])
                            room._forward()
                            gap, robot_geom, scene_geom = (
                                room._current_obstacle_clearance())
                            path.append({"time_s": round(time_s, 4),
                                         "actual_pose_xyyaw":
                                             room.actual.round(5).tolist(),
                                         "evaluator_gap_m": round(float(gap), 5)})
                            if gap < minimum[0]:
                                minimum = (float(gap), time_s,
                                           robot_geom, scene_geom)
                            if first_below is None and gap < .05:
                                first_below = time_s
                        rows.append({
                            "seed": seed, "case": case,
                            "direction": direction,
                            "minimum_scene_gap_m": round(minimum[0], 5),
                            "minimum_time_s": round(minimum[1], 4),
                            "closest_pair": list(minimum[2:]),
                            "first_below_5cm_s": (round(first_below, 4)
                                                  if first_below is not None else None),
                            "path": path,
                        })
                finally:
                    room.close()
    finally:
        structural.structural_model = old_model
    report = {
        "schema": "robot-learning-lab.ape-inference-response-envelope.v1",
        "source_sha256": sha(SOURCE),
        "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
        "selected_visual_sha256": sha(selected_visual()),
        "alarm_time_s": ALARM_TIME_S,
        "end_time_s": END_TIME_S,
        "command_step_m": STEP_M,
        "imposed_actual_motion_factor": SLIP,
        "directions_xy": DIRECTIONS,
        "cases": CASES,
        "scope": "Evaluator-only 20ms whole-body geometry screen from V98's saved inference-time alarm posture, with the same articulated owner and moving low prop plus V96 right-side blocks. Seven fixed commands, two seeds, four scenes. Candidate feasibility is not an actor choice, observed-free permission or selected-body skill result.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for case in CASES:
        vals = [row for row in rows if row["seed"] == 170 and row["case"] == case]
        print(case, [(row["direction"], row["minimum_scene_gap_m"])
                     for row in vals])
    print(path)


if __name__ == "__main__":
    main()
