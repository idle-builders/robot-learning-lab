"""Render the V120 changed-prop challenge as one synchronized comparison video."""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from articulated_person_scene import (FAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
import direct_structural_rooms_v270 as structural
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
import run_blocked_owner_help_v317 as owner_panel
import run_live_articulated_owner_v343 as owner
from run_moving_low_clutter_v302 import STOOL_SIZE
from run_pre_show_owner_request_v112 import (LOOK_END_S, PRESENT_END_S,
                                             PRESENT_START_S, PROP_START_Y_M)
from run_pre_show_reserve_screen_v100 import START_S, V98
from run_compact_shin_challenge_v120 import OUT, PROFILES, motion_y


SOURCE = Path(__file__)


def main() -> None:
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    prior = next(row for row in v98["rows"]
                 if row["condition"] == "moving" and row["case"][0] == 170)
    source = (V98 / "moving/seed170-present/seed170-present/owner-guard-00.npz")
    with np.load(source) as saved:
        qpos = saved["actor_qpos"].copy()
        estimate = saved["estimated_pose_xyyaw"].copy()
    original = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        original(kind, side))
    rooms = {}
    try:
        for profile in PROFILES:
            room = OfferDriftRoom(case=tuple(prior["case"]),
                                  pose_source="command",
                                  branch=OUT / f"render-{profile}")
            rooms[profile] = room
            room.layout(170, absent=False, furnished=True)
            room.data.qpos[:] = qpos
            room.actual[:] = (qpos[0], qpos[1],
                              2*math.atan2(qpos[6],qpos[3]))
            room.estimated[:] = estimate
            room.head = float(qpos[room.head_addr])
            room.time_s = START_S
            set_person_pose(room.model, START_HAND_XYZ_M,
                            carried_item="loop_TomatoSauce")
            room.obstacle_xy[3] = (0., PROP_START_Y_M)
            room.model.body_pos[room.model.body("loop_stool").id] = (
                0., PROP_START_Y_M, STOOL_SIZE[2])
            room._forward()
        path = OUT / "compact-shin-changed-props.mp4"
        with imageio.get_writer(path, fps=15, codec="libx264",
                                quality=7, macro_block_size=1) as video:
            for index in range(31):
                time_s = START_S+index/15
                panels = []
                for profile, room in rooms.items():
                    room.time_s = time_s
                    room.actual[:2] = qpos[:2]
                    room.estimated[:] = estimate
                    y = motion_y(profile, time_s)
                    room.obstacle_xy[3] = (0., y)
                    room.model.body_pos[room.model.body(
                        "loop_stool").id] = (0., y, STOOL_SIZE[2])
                    shown = min(1., max(0., (time_s-PRESENT_START_S)/
                                         (PRESENT_END_S-PRESENT_START_S)))
                    hand = START_HAND_XYZ_M + shown * (
                        FAR_HAND_XYZ_M-START_HAND_XYZ_M)
                    set_person_pose(room.model, hand,
                                    carried_item="loop_TomatoSauce")
                    if time_s >= PRESENT_END_S-1e-6:
                        room.model.body_pos[room.model.body(
                            "loop_TomatoSauce").id] = owner.OWNER_SHOW_XYZ_M
                    look = min(1., max(0., (time_s-PRESENT_END_S)/
                                          (LOOK_END_S-PRESENT_END_S)))
                    room.head = float(qpos[room.head_addr])*(1.-look)
                    room._forward()
                    row = next(item for item in report["rows"]
                               if item["seed"] == 170 and
                               item["branch"] == profile and
                               item["sensor_trial"] == 1)
                    alarm = row["audited_alarms"]["four"][
                        "first_supported"]
                    status = ("range alert" if alarm is not None and
                              time_s >= alarm["time_s"]-1e-4 else
                              "range monitoring")
                    gap = room._current_obstacle_clearance()[0]
                    rgb, _ = room.capture("lower_rgb", depth=False)
                    panels.append(owner_panel.panel(
                        room, rgb,
                        f"{profile} t={time_s:.2f}s | {status} | "
                        f"evaluator gap={100*gap:.1f}cm"))
                video.append_data(np.concatenate(panels, axis=0))
        media = {"source_sha256": sha(SOURCE),
                 "v120_report_sha256": sha(report_path),
                 "video_path": str(path.relative_to(ROOT)),
                 "video_sha256": sha(path),
                 "frames": 31, "fps": 15,
                 "scope": "Seed170, sensor trial1 comparison of V120 scripted prop motions. The overlaid range alert is derived from saved actor packets; gap is evaluator-only. Visual replay, not fresh RGB-D inference or body response."}
        metadata = OUT / "video-report.json"
        metadata.write_text(json.dumps(media, indent=2)+"\n")
        print(path)
    finally:
        for room in rooms.values():
            room.close()
        structural.structural_model = original


if __name__ == "__main__":
    main()
