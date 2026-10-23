"""Render later V121 alert with held and hypothetical translation responses."""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from articulated_person_scene import add_person
import direct_structural_rooms_v270 as structural
from run import ROOT, sha
from run_articulated_owner_low_drift_v92 import OfferDriftRoom
from run_owner_exit_height_v96 import CASES
from run_pre_show_reserve_screen_v100 import V98
from audit_alert_response_envelope_v122p import (DIRECTIONS, END_S, OUT,
                                                  scene_at)


SOURCE = Path(__file__)
REPORT = OUT / "report.json"
PANELS = (
    ("clear", "hold", 0., 1., .2),
    ("clear", "right", .08, .55, .2),
    ("hand_close", "hold", 0., 1., .2),
    ("hand_close", "back", .16, 1., .2),
)


def main() -> None:
    report = json.loads(REPORT.read_text())
    alarm_s = next(r["alert_time_s"] for r in report["rows"] if
                   r["seed"] == 172 and r["case"] == "clear")
    assert alarm_s == 17.4
    v98 = json.loads((V98 / "live-moving-owner-offer-audit.json").read_text())
    prior = next(r for r in v98["rows"] if
                 r["condition"] == "moving" and r["case"][0] == 172)
    source = (V98 / "moving/seed172-present/seed172-present/owner-guard-00.npz")
    with np.load(source) as saved:
        qpos = saved["actor_qpos"].copy()
        estimate = saved["estimated_pose_xyyaw"].copy()
    actual_start = np.array((qpos[0],qpos[1],
        2*math.atan2(qpos[6],qpos[3])))
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind,side))
    rooms = []
    try:
        for case,direction,command,factor,duration in PANELS:
            room = OfferDriftRoom(case=tuple(prior["case"]),
                                  pose_source="command",branch=OUT/"video")
            rooms.append(room)
            room.layout(172,absent=False,furnished=True)
            room.data.qpos[:] = qpos
            obstacle = CASES[case]
            if obstacle is not None:
                position,size = obstacle
                room.model.body_pos[room.model.body("loop_box").id] = position
                room.model.geom("loop_box_geom").size[:] = size
        video_path = OUT/"later-alert-hold-vs-optimistic-exit.mp4"
        with imageio.get_writer(video_path,fps=15,codec="libx264",
                                quality=7,macro_block_size=1) as writer:
            for tick in range(41):
                time_s = alarm_s+.02*tick
                tiles = []
                for (case,direction,command,factor,duration),room in zip(
                        PANELS,rooms):
                    dx,dy = DIRECTIONS[direction]
                    fraction = min(1.,max(0.,(time_s-alarm_s)/duration))
                    actual = actual_start.copy()
                    actual[:2] += command*factor*fraction*np.array((dx,dy))
                    gap = scene_at(room,qpos,estimate,actual,time_s)[0]
                    view = Image.fromarray(np.asarray(
                        room.capture_external()))
                    tile = view.crop((50,50,500,400)).resize((640,500))
                    draw = ImageDraw.Draw(tile)
                    draw.rectangle((0,0,640,40),fill=(0,0,0))
                    caption = (f"{case}: {'hold' if direction=='hold' else 'ideal '+direction} "
                               f"t={time_s:.2f}s  eval gap={100*gap:.1f}cm")
                    draw.text((8,12),caption,fill=(255,255,255))
                    tiles.append(np.asarray(tile))
                top = np.concatenate(tiles[:2],axis=1)
                bottom = np.concatenate(tiles[2:],axis=1)
                writer.append_data(np.concatenate((top,bottom),axis=0))
        output={"schema":"robot-learning-lab.ape-alert-response-video.v1",
                "source_sha256":sha(SOURCE),
                "response_report_sha256":sha(REPORT),
                "video_path":str(video_path.relative_to(ROOT)),
                "video_sha256":sha(video_path),
                "frames":41,"fps":15,
                "scope":"Seed172 later finite-zone alert at 17.4s. Side-by-side evaluator render of hold, idealized clear-scene right translation (8cm command x0.55 in0.2s), and optimistic hand-close back translation (16cm command x1.0 in0.2s). No actor route authorization, learned skill, measured dynamics, or fresh inference. Gap captions are evaluator-only."}
        (OUT/"video-report.json").write_text(json.dumps(
            output,indent=2)+"\n")
        print(video_path)
    finally:
        for room in rooms:
            room.close()
        structural.structural_model=old_model


if __name__=="__main__":
    main()
