"""Visualize nominal versus native-error-shaped early owner meeting.

The stress branch uses one laterally mirrored first-12cm saved native gait
error. Both scenes remain prescribed kinematics on the historical body.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from articulated_person_scene import (FAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
import direct_structural_rooms_v270 as structural
from direct_structural_rooms_v270 import selected_visual
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
import run_blocked_owner_help_v317 as owner_panel
import run_live_articulated_owner_v343 as owner
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_owner_exit_height_v96 import CASES
import run_pre_show_reserve_screen_v100 as v100
import stress_early_meeting_native_pose_v142p as v142p
import validate_early_meeting_v128p as v128p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-native-pose-meeting"
VIDEO = OUT / "nominal-vs-native-error.mp4"
SEED = 170
CASE = "hand_close"


def configure(room: OfferDriftRoom, qpos: np.ndarray,
              estimate: np.ndarray, t: float,
              offset_xy: np.ndarray, yaw_error: float,
              label: str) -> tuple[np.ndarray, dict]:
    early = t < 10.
    fraction = (min(1., max(0., (t-8.0)/1.2))
                if early else 1.)
    room.actual[:] = (qpos[0]+fraction*offset_xy[0],
                      qpos[1]+fraction*offset_xy[1],
                      fraction*yaw_error)
    room.estimated[:] = estimate
    room.estimated[:2] += fraction*v142p.DESIRED_OFFSET
    shoulder = room.model.joint("left_shoulder_pitch").qposadr[0]
    room.data.qpos[shoulder] = fraction*v142p.SHOULDER_RAD
    if early:
        room.head = float(qpos[room.head_addr])
        offer = 0.
        room.model.body_pos[room.model.body("loop_box").id] = (10., 0., .1)
    else:
        position, size = CASES[CASE]
        room.model.body_pos[room.model.body("loop_box").id] = position
        room.model.geom("loop_box_geom").size[:] = size
        offer = min(1., max(0., (t-16.6)/.4))
        look = min(1., max(0., (t-17.)/.4))
        room.head = float(qpos[room.head_addr])*(1.-look)
    hand = START_HAND_XYZ_M+offer*(FAR_HAND_XYZ_M-START_HAND_XYZ_M)
    set_person_pose(room.model, hand, carried_item="loop_TomatoSauce")
    if not early and t >= 17.-1e-6:
        room.model.body_pos[room.model.body(
            "loop_TomatoSauce").id] = owner.OWNER_SHOW_XYZ_M
    y = v100.prop_y(t)
    room.obstacle_xy[3] = (0., y)
    room.model.body_pos[room.model.body("loop_stool").id] = (
        0., y, STOOL_SIZE[2])
    room.time_s = t
    room._forward()
    gap, robot, obstacle = room._current_obstacle_clearance()
    phase = ("early retreat" if early else
             "owner waits" if t < 16.6 else
             "owner shows" if t < 17. else
             "ape looks" if t < 17.4 else "held view")
    rgb, _ = room.capture("lower_rgb", depth=False)
    frame = owner_panel.panel(
        room, rgb, f"{label} | {phase} | t{t:.1f}s | "
        f"evaluator gap {100*gap:.1f}cm")
    return frame, {"time_s": t, "phase": phase,
                   "evaluator_gap_m": round(float(gap), 5),
                   "closest_pair": [robot, obstacle]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v142 = json.loads((v142p.OUT / "report.json").read_text())
    stress = next(row for row in v142["rows"]
                  if row["native_case"] == "planar200v3-forward"
                  and row["mirrored_lateral_error"]
                  and row["case"] == CASE)
    stress_offset = np.asarray(stress["actual_station_offset_xy_m"])
    stress_yaw = float(stress["actual_minus_estimated_yaw_rad"])
    v98 = json.loads((v128p.V98 /
                      "live-moving-owner-offer-audit.json").read_text())
    prior = next(row for row in v98["rows"]
                 if row["condition"] == "moving" and row["case"][0] == SEED)
    source = (v128p.V98 / "moving" / f"seed{SEED}-present" /
              f"seed{SEED}-present/side-probe-00.npz")
    with np.load(source) as packet:
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
    original_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        original_model(kind, side))
    rooms = []
    try:
        for _ in range(2):
            room = OfferDriftRoom(case=tuple(prior["case"]),
                                  pose_source="command", branch=OUT)
            room.layout(SEED, absent=False, furnished=True)
            room.data.qpos[:] = qpos
            rooms.append(room)
        timeline = [round(8.0+.1*i, 3) for i in range(13)] + [
            round(16.2+.1*i, 3) for i in range(21)]
        frames = []
        with imageio.get_writer(VIDEO, fps=10, codec="libx264",
                                quality=7) as writer:
            for t in timeline:
                nominal_frame, nominal = configure(
                    rooms[0], qpos, estimate, t,
                    v142p.DESIRED_OFFSET, 0., "Nominal")
                stress_frame, stressed = configure(
                    rooms[1], qpos, estimate, t,
                    stress_offset, stress_yaw, "Native error transfer")
                writer.append_data(np.concatenate(
                    (nominal_frame, stress_frame), axis=1))
                frames.append({"time_s": t,
                               "nominal": nominal,
                               "native_error_transfer": stressed})
    finally:
        for room in rooms:
            room.close()
        structural.structural_model = original_model
    report = {"schema": "robot-learning-lab.ape-native-pose-meeting-video.v1",
              "source_sha256": sha(SOURCE),
              "v142_report_sha256": sha(v142p.OUT / "report.json"),
              "selected_visual_sha256": sha(selected_visual()),
              "source_packet_path": str(source.relative_to(ROOT)),
              "source_packet_sha256": sha(source),
              "stress_native_case": stress["native_case"],
              "stress_mirrored_lateral_error": True,
              "stress_offset_xy_m": stress_offset.tolist(),
              "stress_yaw_error_rad": stress_yaw,
              "video_path": str(VIDEO.relative_to(ROOT)),
              "video_sha256": sha(VIDEO),
              "frames": frames,
              "scope": "Side-by-side visual replay of the prescribed "
              "nominal12cm retreat and one measured forward-native-gait "
              "error shape transferred to the historical owner scene. "
              "Static9.2-16.2s interval is omitted; evaluator gap labels "
              "and actual poses are not actor inputs. The native policy "
              "did not run backward in this room."}
    (OUT / "video-report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(VIDEO, len(frames),
          min(x["native_error_transfer"]["evaluator_gap_m"]
              for x in frames))


if __name__ == "__main__":
    main()
