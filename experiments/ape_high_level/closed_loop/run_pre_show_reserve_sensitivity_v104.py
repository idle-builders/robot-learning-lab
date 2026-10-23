"""V104: sensitivity of V100's fixed pre-show reserve to pose and slip.

The actor's nominal command sequence and estimated start pose are unchanged.
Only evaluator actual start pose and executed fraction vary. This is a local
uncertainty screen, not an estimated covariance or selected-body dynamics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from articulated_person_scene import (FAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
import run_live_articulated_owner_v343 as owner
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
from run_pre_show_reserve_screen_v100 import (COMMAND_STEP_M, DIRECTIONS,
                                              INFERENCE_S, LOOK_S,
                                              OWNER_MOVE_S, SEQUENCES,
                                              START_S, STEP_DURATION_S,
                                              V98, commanded_offset, prop_y)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v104"
SEQUENCE_NAMES = ("back2_diag2", "back2_diag3")
SLIPS = (.45, .55, .70)
START_PERTURBATIONS = {
    "nominal": (0., 0., 0.),
    "right_1cm": (0., -.01, 0.),
    "right_2cm": (0., -.02, 0.),
    "left_1cm": (0., .01, 0.),
    "back_1cm": (-.01, 0., 0.),
    "forward_1cm": (.01, 0., 0.),
    "yaw_left_2deg": (0., 0., math.radians(2.)),
    "yaw_right_2deg": (0., 0., -math.radians(2.)),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    prior = next(row for row in v98["rows"]
                 if row["condition"] == "moving" and row["case"][0] == 170)
    source = (V98 / "moving/seed170-present/seed170-present/owner-guard-00.npz")
    with np.load(source) as packet:
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
    base_actual = np.array((qpos[0], qpos[1],
                            2*math.atan2(qpos[6], qpos[3])))
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rows = []
    try:
        for case in CASES:
            room = OfferDriftRoom(case=tuple(prior["case"]),
                                  pose_source="command", branch=OUT / "scene")
            try:
                room.layout(170, absent=False, furnished=True)
                obstacle = CASES[case]
                if obstacle is not None:
                    position, size = obstacle
                    room.model.body_pos[room.model.body("loop_box").id] = position
                    room.model.geom("loop_box_geom").size[:] = size
                for sequence_name in SEQUENCE_NAMES:
                    sequence = SEQUENCES[sequence_name]
                    owner_start = START_S + len(sequence)*STEP_DURATION_S
                    owner_end = owner_start + OWNER_MOVE_S
                    image_time = owner_end + LOOK_S
                    end_time = max(18., image_time + INFERENCE_S)
                    total_ticks = int(round((end_time-START_S)/.02))
                    for label, (dx, dy, dyaw) in START_PERTURBATIONS.items():
                        start = base_actual + np.array((dx, dy, dyaw))
                        changed_qpos = qpos.copy()
                        changed_qpos[:2] = start[:2]
                        changed_qpos[3:7] = (math.cos(start[2]/2), 0., 0.,
                                                math.sin(start[2]/2))
                        for slip in SLIPS:
                            room.data.qpos[:] = changed_qpos
                            start_head = float(changed_qpos[room.head_addr])
                            minimum = (float("inf"), None, None, None)
                            first_below = None
                            for tick in range(total_ticks+1):
                                time_s = START_S + .02*tick
                                offset = commanded_offset(
                                    sequence, time_s-START_S)
                                room.actual[:] = start
                                room.actual[:2] += slip*offset
                                room.estimated[:] = estimate
                                room.estimated[:2] += offset
                                room.time_s = time_s
                                look_fraction = min(1., max(0.,
                                    (time_s-owner_end)/LOOK_S))
                                room.head = start_head*(1.-look_fraction)
                                fraction = min(1., max(0.,
                                    (time_s-owner_start)/OWNER_MOVE_S))
                                hand = (START_HAND_XYZ_M + fraction *
                                        (FAR_HAND_XYZ_M - START_HAND_XYZ_M))
                                set_person_pose(room.model, hand,
                                                carried_item="loop_TomatoSauce")
                                if time_s >= owner_end - 1e-6:
                                    room.model.body_pos[room.model.body(
                                        "loop_TomatoSauce").id] = (
                                            owner.OWNER_SHOW_XYZ_M)
                                y = prop_y(time_s)
                                room.obstacle_xy[3] = (0., y)
                                room.model.body_pos[room.model.body(
                                    "loop_stool").id] = (0., y, STOOL_SIZE[2])
                                room._forward()
                                gap, robot_geom, obstacle_geom = (
                                    room._current_obstacle_clearance())
                                if gap < minimum[0]:
                                    minimum = (float(gap), time_s,
                                               robot_geom, obstacle_geom)
                                if first_below is None and gap < .05:
                                    first_below = time_s
                            rows.append({
                                "case": case, "sequence": sequence_name,
                                "start_perturbation": label,
                                "actual_start_offset_xyyaw": (dx, dy, dyaw),
                                "executed_motion_factor": slip,
                                "minimum_scene_gap_m": round(minimum[0], 5),
                                "minimum_time_s": round(minimum[1], 4),
                                "closest_pair": list(minimum[2:]),
                                "first_below_5cm_s": (round(first_below, 4)
                                                      if first_below is not None
                                                      else None),
                                "end_time_s": round(end_time, 4),
                            })
            finally:
                room.close()
    finally:
        structural.structural_model = old_model
    report = {
        "schema": "robot-learning-lab.ape-pre-show-reserve-sensitivity.v1",
        "source_sha256": sha(SOURCE),
        "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
        "source_packet_path": str(source.relative_to(ROOT)),
        "source_packet_sha256": sha(source),
        "selected_visual_sha256": sha(selected_visual()),
        "sequences": SEQUENCE_NAMES,
        "start_perturbations_xyyaw": START_PERTURBATIONS,
        "executed_motion_factors": SLIPS,
        "scope": "Evaluator-only local uncertainty screen at V98's 16.2s pre-show state, with actor command sequence and estimated pose held fixed. Two development sequences, four scenes, eight start-pose perturbations, three imposed execution factors,20ms full-body geometry through item presentation and complete prop approach. No probability model, actuator dynamics, actor clearance decision or new owner image inference is implied.",
        "rows": rows,
    }
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for sequence in SEQUENCE_NAMES:
        values = [row for row in rows if row["sequence"] == sequence]
        print(sequence, "clear", sum(row["minimum_scene_gap_m"] >= .05
                                     for row in values), "/", len(values),
              "minimum", min(row["minimum_scene_gap_m"] for row in values))
        print("hand close nominal slip", [
            (row["start_perturbation"], row["minimum_scene_gap_m"])
            for row in values if row["case"] == "hand_close" and
            row["executed_motion_factor"] == .55])
    print(path)


if __name__ == "__main__":
    main()
