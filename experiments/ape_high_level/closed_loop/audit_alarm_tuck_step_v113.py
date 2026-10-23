"""V113 evaluator screen: can arm retraction make the V98 side exit feasible?

This is a counterfactual on the historical Direct Frame body, not an actor
choice or a qualified combined motion skill. No evaluator geometry is exposed
to the controller.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from articulated_person_scene import FAR_HAND_XYZ_M, add_person, set_person_pose
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_inference_response_envelope_v99 import (ALARM_TIME_S, END_TIME_S,
                                                  SLIP, STEP_M, V98)
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
from audit_protective_postures_v327 import self_clearance


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v113"
# A small predeclared set; all targets are inside the model's joint ranges.
POSTURES = {
    "plain_right": {},
    "elbow_tuck": {"right_elbow_pitch": -.8},
    "shoulder_back": {"right_shoulder_pitch": -.8},
    "shoulder_back_elbow": {"right_shoulder_pitch": -.8,
                            "right_elbow_pitch": -.8},
    "shoulder_roll_in": {"right_shoulder_roll": .8},
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
                capture_time = float(packet["capture_time_s"])
            if abs(capture_time-ALARM_TIME_S) > 1e-5:
                raise AssertionError("Saved alarm state changed")
            actual = np.array((qpos[0], qpos[1],
                               2*math.atan2(qpos[6], qpos[3])))
            for case, obstacle in CASES.items():
                room = OfferDriftRoom(case=tuple(prior["case"]),
                                      pose_source="command", branch=OUT)
                try:
                    room.layout(seed, absent=False, furnished=True)
                    if obstacle is not None:
                        position, size = obstacle
                        room.model.body_pos[room.model.body("loop_box").id] = position
                        room.model.geom("loop_box_geom").size[:] = size
                    room.estimated[:] = estimate
                    room.head = float(qpos[room.head_addr])
                    set_person_pose(room.model, FAR_HAND_XYZ_M,
                                    carried_item="loop_TomatoSauce")
                    for name, target in POSTURES.items():
                        scene_min = (float("inf"), None, None, None)
                        self_min = (float("inf"), None, None, None)
                        first_below = None
                        path = []
                        video_path = OUT / f"seed{seed}-{case}-{name}.mp4"
                        record = seed == 170 and case == "hand_close"
                        writer = (imageio.get_writer(video_path, fps=5,
                                                     codec="libx264", quality=7)
                                  if record else None)
                        try:
                            for tick in range(31):
                                time_s = ALARM_TIME_S + .02*tick
                                fraction = min(1., (time_s-ALARM_TIME_S)/.4)
                                room.data.qpos[:] = qpos
                                for joint_name, radians in target.items():
                                    joint = room.model.joint(joint_name)
                                    lo, hi = joint.range
                                    if not lo <= radians <= hi:
                                        raise AssertionError((joint_name, radians,
                                                              lo, hi))
                                    room.data.qpos[joint.qposadr[0]] = radians*fraction
                                room.actual[:] = actual
                                room.actual[1] -= STEP_M*SLIP*fraction
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
                                gap, body, scene = room._current_obstacle_clearance()
                                self_gap, own, other = self_clearance(room, "right")
                                scene_min = min(scene_min, (gap, time_s, body, scene))
                                self_min = min(self_min, (self_gap, time_s,
                                                          own, other))
                                if first_below is None and gap < .05:
                                    first_below = time_s
                                path.append({"time_s": round(time_s, 4),
                                             "scene_gap_m": round(gap, 5),
                                             "self_gap_m": round(self_gap, 5)})
                                if writer is not None and tick % 10 == 0:
                                    writer.append_data(np.asarray(
                                        room.capture_external()))
                        finally:
                            if writer is not None:
                                writer.close()
                        rows.append({
                            "seed": seed, "case": case, "posture": name,
                            "target_joints_rad": target,
                            "minimum_scene_gap_m": round(scene_min[0], 5),
                            "closest_scene_time_s": round(scene_min[1], 4),
                            "closest_scene_geoms": list(scene_min[2:]),
                            "minimum_self_gap_m": round(self_min[0], 5),
                            "closest_self_time_s": round(self_min[1], 4),
                            "closest_self_geoms": list(self_min[2:]),
                            "first_scene_below_5cm_s": (round(first_below, 4)
                                                         if first_below else None),
                            "geometrically_feasible": (scene_min[0] >= .05 and
                                                       self_min[0] >= 0.),
                            "video": str(video_path.relative_to(ROOT))
                            if record else None,
                            "path": path,
                        })
                finally:
                    room.close()
    finally:
        structural.structural_model = old_model
    report = {
        "schema": "robot-learning-lab.ape-alarm-tuck-step-screen.v1",
        "source_sha256": sha(SOURCE),
        "v98_report_sha256": sha(V98 / "live-moving-owner-offer-audit.json"),
        "selected_visual_sha256": sha(selected_visual()),
        "body_status": "historical_rejected_direct_frame_no_selected_design",
        "scope": "Evaluator-only kinematic 20ms combined right step and right-arm posture screen from V98 alarm. No actor-observed free corridor, pose bound, low-level skill, dynamics or safety claim.",
        "alarm_time_s": ALARM_TIME_S, "end_time_s": END_TIME_S,
        "command_step_m": STEP_M, "imposed_actual_motion_factor": SLIP,
        "postures": POSTURES, "rows": rows,
    }
    report_path = OUT / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for case in CASES:
        subset = [r for r in rows if r["seed"] == 170 and r["case"] == case]
        print(case, [(r["posture"], r["minimum_scene_gap_m"],
                      r["minimum_self_gap_m"], r["geometrically_feasible"])
                     for r in subset])
    print(report_path)


if __name__ == "__main__":
    main()
