"""Independent preflight, live packet, outcome and media audit for V169p."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from run import ROOT, sha
import run_admitted_occluded_owner_v169p as v169p


SOURCE = Path(__file__)
REPORT = v169p.OUT / "report.json"


def main() -> None:
    report = json.loads(REPORT.read_text())
    checks = packets = videos = 0

    def check(ok: bool, reason: str) -> None:
        nonlocal checks
        if not ok:
            raise AssertionError(reason)
        checks += 1

    check(sha(Path(v169p.__file__)) == report["source_sha256"],
          "Joined runner source changed")
    preflight_file = v169p.v167p.OUT / "report.json"
    check(sha(preflight_file) == report["v167_report_sha256"],
          "Pre-show admission report changed")
    preflight = json.loads(preflight_file.read_text())
    admitted = [r for r in preflight["rows"]
                if r["layout"] == "four_front_side" and
                r["occlusion"] == "forward_pair_occluded" and
                r["pre_show_decision"]["provisional_admission"]]
    check(report["admission_decisions"] == {
        "candidate_scenes": 8,
        "admitted_scenes": 2,
        "deferred_without_owner_turn": 6,
    } and len(admitted) == 2, "Pre-show gate changed")
    check(len(report["rows"]) == 4 and
          {(r["seed"], r["scripted_prop_stop_delay_s"])
           for r in report["rows"]} ==
          {(seed, delay) for seed in (170, 172)
           for delay in (.2, None)},
          "Live continuation matrix incomplete")
    for row in report["rows"]:
        prior = next(r for r in admitted if r["seed"] == row["seed"])
        check(row["variant"] == prior["variant"] ==
              "fast_wide_turn" and
              row["station_x_m"] == prior["station_x_m"] == -.22 and
              row["pre_show_supported_packets"] ==
              prior["pre_show_decision"]["supported_packets"],
              "Continued scene differs from admitted scene")
        check(sha(ROOT / row["staging_packet"]) ==
              row["staging_packet_sha256"] and
              row["staging_packet_sha256"] ==
              prior["staging_packet_sha256"],
              "Staging joint/pose source changed")
        check(row["pre_show_packet_paths"] == prior["pre_show_packets"],
              "Pre-show packet provenance changed")
        for packet in row["pre_show_packet_paths"]:
            check(sha(ROOT / packet["packet_path"]) ==
                  packet["packet_sha256"],
                  "Pre-show range packet changed")
        monitor_file = ROOT / row["monitor"]["path"]
        check(sha(monitor_file) == row["monitor"]["sha256"],
              "Live range monitor changed")
        monitor = json.loads(monitor_file.read_text())
        frames = monitor["frames"]
        check(len(frames) >= 50 and
              len(row["removed_by_live_packet"]) == len(frames),
              "Live monitor or occlusion injection incomplete")
        check(row["front_alarm"] is not None and
              row["front_alarm"]["side"] == "front_left" and
              row["front_alarm"]["capture_time_s"] == 17.3,
              "Live front-left warning changed")
        check(sum(p["removed_forward_roi_returns"]
                  for p in row["removed_by_live_packet"]) > 0,
              "Live occlusion removed no returns")
        for frame in frames:
            packet_file = ROOT / frame["packet_path"]
            check(sha(packet_file) == frame["packet_sha256"],
                  "Live actor range packet changed")
            with np.load(packet_file, allow_pickle=False) as packet:
                front = packet["front_four_points_xyz_m"]
                pose = packet["estimated_pose_xyyaw"]
            forward_count = frame["front_counts"].get(
                "two_environment_points", 0)
            side_count = frame["front_counts"].get(
                "four_environment_points", 0)
            check(len(front) == forward_count+side_count,
                  "Live source counts differ from packet")
            check(not np.any(v169p.front_left_roi(
                front[:forward_count], pose)),
                "Occluded forward source still returns into target field")
            packets += 1
        check(row["task_result"] == "owner_turn_unresolved" and
              row["first_image"] is None and
              row["confirm_image"] is None,
              "Occluded turn made unsupported item claim")
        gap = row["evaluator_minimum_scene_gap_m"]
        if row["scripted_prop_stop_delay_s"] is None:
            check(gap < 0., "Ignored branch no longer overlaps")
        else:
            check(.05 < gap < .06,
                  "Scripted halt gap changed")
        video_file = ROOT / row["video_path"]
        check(sha(video_file) == row["video_sha256"],
              "Live episode video changed")
        capture = cv2.VideoCapture(str(video_file))
        decoded = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            decoded += 1
            check(image.ndim == 3 and image.shape[0] > 0,
                  "Live video frame malformed")
        capture.release()
        check(decoded >= 25, "Live episode video incomplete")
        videos += 1
    result = {
        "schema": "robot-learning-lab.ape-admitted-occluded-owner-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "checks_passed": checks,
        "live_actor_packets_checked": packets,
        "episode_videos_decoded": videos,
        "outcome": "pass",
    }
    (v169p.OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
