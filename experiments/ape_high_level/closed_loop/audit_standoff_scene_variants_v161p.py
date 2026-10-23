"""Audit source, scene geometry, packet timing and videos for V160p."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from articulated_person_scene import (FAR_HAND_XYZ_M, PERSON_GEOMS,
                                      add_person, set_person_pose)
import direct_structural_rooms_v270 as structural
from run import ROOT, sha
import run_live_articulated_owner_v343 as owner
import run_pre_show_reserve_screen_v100 as v100
import run_standoff_scene_variants_v160p as v160p


SOURCE = Path(__file__)
REPORT = v160p.OUT / "report.json"


def minimum_person_prop_gap(variant: dict) -> float:
    v98 = json.loads((v100.V98 /
                      "live-moving-owner-offer-audit.json").read_text())
    prior = next(row for row in v98["rows"]
                 if row["condition"] == "moving" and row["case"][0] == 170)
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    v160p.VariantRoom.variant = variant
    room = None
    try:
        room = v160p.VariantRoom(case=tuple(prior["case"]),
                                 pose_source="command", branch=v160p.OUT)
        room.layout(170, absent=False, furnished=True)
        prop = room.model.geom("loop_stool_geom").id
        person = [room.model.geom(name).id for name in PERSON_GEOMS]
        minimum = float("inf")
        start = variant["approach_start_s"]
        end = variant["approach_end_s"]
        for time_s in np.arange(start, end+.0101, .02):
            room.time_s = float(time_s)
            set_person_pose(room.model, FAR_HAND_XYZ_M,
                            carried_item="loop_TomatoSauce")
            room.model.body_pos[room.model.body(
                "loop_TomatoSauce").id] = owner.OWNER_SHOW_XYZ_M
            room._forward()
            minimum = min(minimum, *(float(mujoco.mj_geomDistance(
                room.model, room.data, prop, geom, 10., None))
                for geom in person))
        return round(minimum, 5)
    finally:
        if room is not None:
            room.close()
        structural.structural_model = old_model


def main() -> None:
    report = json.loads(REPORT.read_text())
    checks = packets = videos = confirm_packets = 0

    def check(ok: bool, reason: str) -> None:
        nonlocal checks
        if not ok:
            raise AssertionError(reason)
        checks += 1

    check(sha(Path(v160p.__file__)) == report["source_sha256"],
          "Variant source changed")
    check(report["variants"] == json.loads(json.dumps(v160p.VARIANTS)),
          "Variant definitions changed")
    check(len(report["rows"]) == 16, "Variant matrix incomplete")
    check({(r["variant"], r["seed"], r["station_x_m"],
            r["scripted_prop_stop_delay_s"]) for r in report["rows"]} ==
          {(variant, seed, station, delay)
           for variant in v160p.VARIANTS for seed in v160p.SEEDS
           for station in v160p.STATIONS_X_M
           for delay in v160p.RESPONSE_DELAYS_S},
          "Variant matrix changed")
    person_gaps = {name: minimum_person_prop_gap(spec)
                   for name, spec in v160p.VARIANTS.items()}
    check(all(gap > .05 for gap in person_gaps.values()),
          "A variant prop intersects the scripted owner")
    for row in report["rows"]:
        check(sha(ROOT / row["staging_packet"]) ==
              row["staging_packet_sha256"], "Staging packet changed")
        monitor_file = ROOT / row["monitor"]["path"]
        check(sha(monitor_file) == row["monitor"]["sha256"],
              "Variant monitor changed")
        monitor = json.loads(monitor_file.read_text())
        frames = monitor["frames"]
        check(len(frames) >= 50, "Variant monitor ended early")
        alarm = row["front_alarm"]
        check(alarm is not None and alarm["side"] == "front_left" and
              alarm["capture_time_s"] in (17.2, 17.3),
              "Changed scene lost the front-left warning")
        check([f["capture_time_s"] for f in frames
               if f.get("front_left_first_cue")] ==
              [alarm["capture_time_s"]],
              "Actor cue differs from monitor packet timeline")
        for frame in frames:
            check(sha(ROOT / frame["packet_path"]) ==
                  frame["packet_sha256"], "Variant range packet changed")
            packets += 1
        gap = row["evaluator_minimum_scene_gap_m"]
        sampled_gap = min(s["evaluator_gap_m"]
                          for s in row["samples_10hz"])
        check(0. <= sampled_gap-gap < .002,
              "Saved evaluator gap samples inconsistent")
        if row["task_result"] == "fresh_unattributed_item_view_after_yield":
            check(row["scripted_prop_stop_delay_s"] == .2,
                  "Ignored response made a task claim")
            check(row["actor_first_stable_range_s"] is not None and
                  row["first_image_valid"] and
                  row["confirm_image_valid"] and
                  not row["renewed_front_cue_after_first_capture"],
                  "Claim lacks stable packet and image evidence")
            first, confirm = row["first_image"], row["confirm_image"]
            check(first["capture_time_s"] >=
                  row["actor_first_stable_range_s"] and
                  confirm["capture_time_s"] > first["release_time_s"],
                  "Recovery images are stale or out of order")
            for image in (first, confirm):
                check(sha(ROOT / image["source_frame_path"]) ==
                      image["source_frame_sha256"],
                      "Recovery image source changed")
            check(row["range_packets_during_first_inference"] >= 1,
                  "Range monitor stopped during first image worker")
            during_confirm = sum(
                confirm["worker_start_wall_elapsed_s"] <
                f["wall_elapsed_s"] <
                confirm["worker_finish_wall_elapsed_s"] for f in frames)
            check(during_confirm >= 1,
                  "Range monitor stopped during confirmation worker")
            confirm_packets += during_confirm
            check(gap > .05, "Actor claim fell below modeled gap margin")
        else:
            check(row["task_result"] == "owner_turn_unresolved" and
                  row["first_image"] is None and
                  row["confirm_image"] is None,
                  "Unresolved branch made an unsupported image claim")
        video = ROOT / row["video_path"]
        check(sha(video) == row["video_sha256"],
              "Variant video changed")
        cap = cv2.VideoCapture(str(video))
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1
            check(frame.ndim == 3 and frame.shape[0] > 0,
                  "Variant video frame malformed")
        cap.release()
        check(decoded >= 25, "Variant video incomplete")
        videos += 1
    result = {
        "schema": "robot-learning-lab.ape-standoff-scene-variants-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "checks_passed": checks,
        "actor_range_packets_checked": packets,
        "episode_videos_decoded": videos,
        "packets_during_confirmation_inference": confirm_packets,
        "minimum_person_prop_gap_m_evaluator_only": person_gaps,
        "outcome": "pass",
    }
    (v160p.OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
