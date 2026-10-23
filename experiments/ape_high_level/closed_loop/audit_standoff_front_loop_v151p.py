"""Source, packet, timing, geometry and video audit for V148p–V150p."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

from articulated_person_scene import (FAR_HAND_XYZ_M, PERSON_GEOMS,
                                      add_person, set_person_pose)
import direct_structural_rooms_v270 as structural
from run import ROOT, sha
import run_early_standoff_owner_v148p as v148p
import run_live_articulated_owner_v343 as owner
from run_moving_low_clutter_v302 import STOOL_SIZE
import run_pre_show_reserve_screen_v100 as v100
import run_standoff_front_response_v150p as v150p
import screen_standoff_front_cue_v149p as v149p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-standoff-audit"


def person_prop_gap() -> float:
    """Independently check the challenging scripted prop against the owner."""
    v98 = json.loads((v100.V98 /
                      "live-moving-owner-offer-audit.json").read_text())
    prior = next(row for row in v98["rows"]
                 if row["condition"] == "moving" and row["case"][0] == 170)
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    try:
        room = v148p.StandoffRoom(case=tuple(prior["case"]),
                                  pose_source="command", branch=OUT)
        try:
            room.layout(170, absent=False, furnished=True)
            prop_geom = room.model.geom("loop_stool_geom").id
            person_geoms = [room.model.geom(name).id
                            for name in PERSON_GEOMS]
            minimum = float("inf")
            for step in range(51):
                time_s = 17.2 + .02*step
                x = v148p.prop_x("prop_turn_into_standoff", time_s)
                y = v148p.prop_y("prop_turn_into_standoff", time_s)
                room.time_s = time_s
                room.obstacle_xy[3] = (x, y)
                room.model.body_pos[room.model.body("loop_stool").id] = (
                    x, y, STOOL_SIZE[2])
                set_person_pose(room.model, FAR_HAND_XYZ_M,
                                carried_item="loop_TomatoSauce")
                room.model.body_pos[room.model.body(
                    "loop_TomatoSauce").id] = owner.OWNER_SHOW_XYZ_M
                super(v148p.StandoffRoom, room)._forward()
                minimum = min(minimum, *(float(mujoco.mj_geomDistance(
                    room.model, room.data, prop_geom, person_geom, 10., None))
                    for person_geom in person_geoms))
            return round(minimum, 5)
        finally:
            room.close()
    finally:
        structural.structural_model = old_model


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = (v148p.OUT / "report.json", v149p.OUT / "report.json",
             v150p.OUT / "report.json")
    stage, cue, response = [json.loads(path.read_text()) for path in files]
    checks = 0
    decoded_videos = 0
    checked_packets = 0

    def check(ok: bool, reason: str) -> None:
        nonlocal checks
        if not ok:
            raise AssertionError(reason)
        checks += 1

    for report, source in ((stage, Path(v148p.__file__)),
                           (cue, Path(v149p.__file__)),
                           (response, Path(v150p.__file__))):
        check(sha(source) == report["source_sha256"],
              f"Experiment source changed: {source}")
    check(cue["v148_report_sha256"] == sha(files[0]),
          "Cue screen refers to another owner run")
    check(len(stage["rows"]) == 20 and len(cue["rows"]) == 20 and
          len(response["rows"]) == 12, "Episode matrix incomplete")
    check({(r["seed"], r["station_x_m"], r["profile"])
           for r in stage["rows"]} == {
              (seed, x, profile) for seed in (170, 172)
              for x in (-.30, -.22) for profile in v148p.PROFILES},
          "Owner profile matrix changed")
    cue_by_key = {(r["seed"], r["station_x_m"], r["profile"]): r
                  for r in cue["rows"]}
    for row in stage["rows"]:
        turn = row["owner_turn"]
        key = (row["seed"], row["station_x_m"], row["profile"])
        c = cue_by_key[key]
        monitor = ROOT / turn["monitor"]["path"]
        check(sha(monitor) == turn["monitor"]["sha256"],
              "Monitor trace changed")
        check(c["source_monitor_sha256"] == turn["monitor"]["sha256"],
              "Cue input differs from owner monitor")
        monitor_data = json.loads(monitor.read_text())
        check(len(monitor_data["frames"]) >= 20,
              "Range monitor stopped early")
        check(turn["monitor_packets_during_first_inference"] >= 1,
              "Range monitor did not run during image inference")
        for frame in monitor_data["frames"]:
            check(sha(ROOT / frame["packet_path"]) == frame["packet_sha256"],
                  "Actor range packet changed")
            checked_packets += 1
        for image in (turn["first_image"], turn["recheck_image"]):
            if image is not None:
                check(sha(ROOT / image["source_frame_path"]) ==
                      image["source_frame_sha256"],
                      "Owner RGB-D packet changed")
        video = ROOT / turn["video_path"]
        check(sha(video) == turn["video_sha256"],
              "Episode video changed")
        capture = cv2.VideoCapture(str(video))
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            count += 1
            check(frame.ndim == 3 and frame.shape[0] > 0,
                  "Video frame decode malformed")
        capture.release()
        check(count >= 18, "Episode video too short")
        decoded_videos += 1
        if row["profile"] in ("moving_prop", "hand_close"):
            check(c["first_front_left_cue_s"] is None,
                  "False front-left alarm in control")
            check(turn["task_result"] == "owner_assisted_visual_reacquisition"
                  and turn["evaluator_minimum_scene_gap_m"] > .05,
                  "Clear owner turn changed")
        else:
            check(c["first_front_left_cue_s"] == 17.3,
                  "Changed-path front cue timing changed")
            check(turn["task_result"] == "owner_turn_unresolved",
                  "Changed path incorrectly claimed owner result")
        if row["profile"] == "prop_turn_into_standoff":
            check(turn["evaluator_minimum_scene_gap_m"] < 0.,
                  "Ignored changed path no longer overlaps")
            check(c["front_cue_to_5cm_lead_s_evaluator_only"] > .3,
                  "Front cue lost pre-breach lead")
            if row["station_x_m"] == -.30:
                check(c["original_side_alarm_to_5cm_lead_s_evaluator_only"] < 0.,
                      "Old side alarm no longer late at farther standoff")
    check({(r["seed"], r["station_x_m"], r["response"])
           for r in response["rows"]} == {
              (seed, x, label) for seed in (170, 172)
              for x in (-.30, -.22)
              for label in ("stop-0.2s", "stop-0.5s", "ignored")},
          "Response matrix incomplete")
    for row in response["rows"]:
        turn = row["owner_turn"]
        monitor_path = ROOT / turn["monitor"]["path"]
        check(sha(monitor_path) ==
              turn["monitor"]["sha256"],
              "Response monitor changed")
        monitor_data = json.loads(monitor_path.read_text())
        for frame in monitor_data["frames"]:
            check(sha(ROOT / frame["packet_path"]) == frame["packet_sha256"],
                  "Response actor range packet changed")
            checked_packets += 1
        check(sum(frame.get("front_left_first_cue", False)
                  for frame in monitor_data["frames"]) == 1,
              "Response front cue did not fire exactly once")
        video_path = ROOT / turn["video_path"]
        check(sha(video_path) == turn["video_sha256"],
              "Response video changed")
        capture = cv2.VideoCapture(str(video_path))
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            count += 1
            check(frame.ndim == 3 and frame.shape[0] > 0,
                  "Response video frame malformed")
        capture.release()
        check(count >= 18, "Response video too short")
        decoded_videos += 1
        check(turn["monitor"]["alarm"]["capture_time_s"] == 17.3,
              "Online front cue timing changed")
        check(turn["monitor_packets_during_first_inference"] >= 1,
              "Response monitor stopped during inference")
        check(turn["task_result"] == "owner_turn_unresolved",
              "Alarm latch unexpectedly completed owner turn")
        gap = turn["evaluator_minimum_scene_gap_m"]
        if row["response"] == "stop-0.2s":
            check(gap > .05, "Fast cooperative stop lost clearance")
        elif row["response"] == "stop-0.5s" and row["station_x_m"] == -.30:
            check(gap > .05, "Far standoff lost half-second reserve")
        elif row["response"] == "stop-0.5s":
            check(0. < gap < .05, "Near standoff half-second miss changed")
        else:
            check(gap < 0., "Ignored prop no longer overlaps")
    person_gap = person_prop_gap()
    check(person_gap > .30, "Changed prop crosses the scripted owner")
    pair_report_path = v150p.OUT / "pair-video-report.json"
    pair = json.loads(pair_report_path.read_text())
    check(pair["v150_report_sha256"] == sha(files[2]),
          "Paired video refers to another response run")
    for source in pair["source_videos"].values():
        check(sha(ROOT / source["path"]) == source["sha256"],
              "Paired video source changed")
    pair_video = ROOT / pair["video_path"]
    check(sha(pair_video) == pair["video_sha256"],
          "Paired video changed")
    capture = cv2.VideoCapture(str(pair_video))
    pair_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        pair_frames += 1
        check(frame.ndim == 3 and frame.shape[0] > 0,
              "Paired video frame malformed")
    capture.release()
    check(pair_frames == pair["frames"] == 22,
          "Paired video decode count changed")
    decoded_videos += 1
    result = {
        "schema": "robot-learning-lab.ape-standoff-front-loop-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
        "checks_passed": checks,
        "actor_range_packets_checked": checked_packets,
        "episode_videos_decoded": decoded_videos,
        "minimum_person_prop_gap_m_evaluator_only": person_gap,
        "outcome": "pass",
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
