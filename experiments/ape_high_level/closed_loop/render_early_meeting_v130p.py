"""Render the prescribed early-meeting candidate and late owner offer.

The video skips a static idle interval from 9.2s to16.2s. It visualizes an
evaluator geometry candidate, not an actor-authorized route or joint skill.
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
import validate_early_meeting_v128p as v128p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-early-meeting-video"
SEED = 170
CASES_TO_RENDER = ("clear", "hand_close")
OFFSET = v128p.STATIONS["back12"]
TARGET_SHOULDER_RAD = -.6


def render_case(case: str, qpos: np.ndarray,
                estimate: np.ndarray, source_case: tuple) -> dict:
    room = OfferDriftRoom(case=source_case, pose_source="command", branch=OUT)
    try:
        room.layout(SEED, absent=False, furnished=True)
        room.data.qpos[:] = qpos
        room.head = float(qpos[room.head_addr])
        shoulder = room.model.joint("left_shoulder_pitch").qposadr[0]
        yaw = 2*math.atan2(qpos[6], qpos[3])
        branch = OUT / case
        branch.mkdir(parents=True, exist_ok=True)
        video = branch / "early-meeting-replay.mp4"
        frames = []
        timeline = [round(8.0 + .1*i, 3) for i in range(13)] + [
            round(16.2 + .1*i, 3) for i in range(21)]
        with imageio.get_writer(video, fps=10, codec="libx264",
                                quality=7) as writer:
            for t in timeline:
                early = t < 10.
                travel = (min(1., max(0., (t-8.0)/1.2))
                          if early else 1.)
                room.actual[:] = (qpos[0]+travel*OFFSET[0],
                                  qpos[1]+travel*OFFSET[1], yaw)
                room.estimated[:] = estimate
                room.estimated[:2] += travel*np.asarray(OFFSET)
                room.data.qpos[shoulder] = TARGET_SHOULDER_RAD*travel
                if early:
                    room.head = float(qpos[room.head_addr])
                    offer = 0.
                    room.model.body_pos[room.model.body(
                        "loop_box").id] = (10., 0., .1)
                else:
                    obstacle = CASES[case]
                    if obstacle is not None:
                        position, size = obstacle
                        room.model.body_pos[room.model.body(
                            "loop_box").id] = position
                        room.model.geom("loop_box_geom").size[:] = size
                    else:
                        room.model.body_pos[room.model.body(
                            "loop_box").id] = (10., 0., .1)
                    offer = min(1., max(0., (t-16.6)/.4))
                    look = min(1., max(0., (t-17.)/.4))
                    room.head = float(qpos[room.head_addr])*(1.-look)
                hand = START_HAND_XYZ_M + offer*(
                    FAR_HAND_XYZ_M-START_HAND_XYZ_M)
                set_person_pose(room.model, hand,
                                carried_item="loop_TomatoSauce")
                if not early and t >= 17.-1e-6:
                    room.model.body_pos[room.model.body(
                        "loop_TomatoSauce").id] = owner.OWNER_SHOW_XYZ_M
                y = v100.prop_y(t)
                room.obstacle_xy[3] = (0., y)
                room.model.body_pos[room.model.body("loop_stool").id] = (
                    0., y, STOOL_SIZE[2])
                room.time_s = t
                room._forward()
                gap, robot, obstacle_name = room._current_obstacle_clearance()
                phase = ("early retreat" if early else
                         "owner waits" if t < 16.6 else
                         "owner shows" if t < 17. else
                         "ape looks" if t < 17.4 else "held view")
                rgb, _ = room.capture("lower_rgb", depth=False)
                writer.append_data(owner_panel.panel(
                    room, rgb,
                    f"Early meeting {case} | {phase} | "
                    f"t{t:.1f}s | gap {100*gap:.1f}cm"))
                frames.append({"time_s": t, "phase": phase,
                               "evaluator_gap_m": round(float(gap), 5),
                               "closest_pair": [robot, obstacle_name]})
        return {"case": case,
                "video_path": str(video.relative_to(ROOT)),
                "video_sha256": sha(video), "frames": frames,
                "minimum_sampled_gap_m": min(f["evaluator_gap_m"]
                                             for f in frames)}
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v98 = json.loads((v128p.V98 /
                     "live-moving-owner-offer-audit.json").read_text())
    prior = next(row for row in v98["rows"]
                 if row["condition"] == "moving" and
                 row["case"][0] == SEED)
    source = (v128p.V98 / "moving" / f"seed{SEED}-present" /
              f"seed{SEED}-present/side-probe-00.npz")
    with np.load(source) as packet:
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
    original_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        original_model(kind, side))
    try:
        rows = [render_case(case, qpos, estimate, tuple(prior["case"]))
                for case in CASES_TO_RENDER]
    finally:
        structural.structural_model = original_model
    report = {"schema": "robot-learning-lab.ape-early-meeting-video.v1",
              "source_sha256": sha(SOURCE),
              "v128_report_sha256": sha(v128p.OUT / "report.json"),
              "selected_visual_sha256": sha(selected_visual()),
              "target_left_shoulder_pitch_rad": TARGET_SHOULDER_RAD,
              "source_packet_path": str(source.relative_to(ROOT)),
              "source_packet_sha256": sha(source),
              "scope": "Visual replay of the prescribed12cm early retreat "
              "with left shoulder raised, followed by scripted owner show. "
              "The static idle interval9.2-16.2s is omitted from playback. "
              "Rendered frames are10Hz; safety gaps in V128 use20ms checks. "
              "Route remains unobserved in the actor's V129 map.",
              "rows": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    for row in rows:
        print(row["case"], row["video_path"],
              row["minimum_sampled_gap_m"], flush=True)


if __name__ == "__main__":
    main()
