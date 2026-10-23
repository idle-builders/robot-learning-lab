"""Independent source, fault, task and video audit for V155p."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from run import ROOT, sha
from screen_standoff_front_cue_v149p import front_left_stat
import run_standoff_range_blind_v155p as v155p


SOURCE = Path(__file__)
REPORT = v155p.OUT / "report.json"


def main() -> None:
    report = json.loads(REPORT.read_text())
    checks = packet_count = video_count = 0

    def check(ok: bool, message: str) -> None:
        nonlocal checks
        if not ok:
            raise AssertionError(message)
        checks += 1

    check(sha(Path(v155p.__file__)) == report["source_sha256"],
          "Fault source changed")
    check(len(report["rows"]) == 4, "Fault matrix incomplete")
    check({(r["seed"], r["fault_start_s"]) for r in report["rows"]} ==
          {(seed, start) for seed in v155p.SEEDS
           for start in v155p.FAULT_STARTS_S}, "Fault matrix changed")
    for row in report["rows"]:
        check(sha(ROOT / row["staging_packet"]) ==
              row["staging_packet_sha256"], "Staging packet changed")
        monitor_file = ROOT / row["monitor"]["path"]
        check(sha(monitor_file) == row["monitor"]["sha256"],
              "Fault monitor changed")
        monitor = json.loads(monitor_file.read_text())
        frames = monitor["frames"]
        check(len(frames) >= 50, "Fault monitor ended early")
        fault_frames = [f for f in frames
                        if f["capture_time_s"]+1e-6 >=
                        row["fault_start_s"]]
        check(len(fault_frames) >= 40, "Too few blind packets")
        check(len(row["removed_by_packet"]) == len(fault_frames),
              "Injected packet count mismatches actor packet count")
        check(sum(x["removed_front_left_returns"] for x in
                  row["removed_by_packet"]) > 0,
              "Fault did not remove any returns")
        for frame in frames:
            packet_file = ROOT / frame["packet_path"]
            check(sha(packet_file) == frame["packet_sha256"],
                  "Fault packet changed")
            with np.load(packet_file, allow_pickle=False) as packet:
                support, p10 = front_left_stat(
                    packet["front_four_points_xyz_m"],
                    packet["estimated_pose_xyyaw"])
            check(support == frame["front_left_support"] and
                  p10 == frame["front_left_p10_x_m"],
                  "Actor front statistic differs from saved packet")
            if frame in fault_frames:
                check(support == 0 and p10 is None,
                      "Front-left view not fully blind")
            packet_count += 1
        first_cues = [f["capture_time_s"] for f in frames
                      if f.get("front_left_first_cue")]
        check(row["task_result"] == "owner_turn_unresolved" and
              row["first_image"] is None and
              row["confirm_image"] is None and
              row["actor_first_stable_range_s"] is None,
              "Blind run made an unsupported visual claim")
        if row["fault_start_s"] == 17.2:
            check(not first_cues and row["front_alarm"] is None,
                  "Pre-alert blindness still produced front warning")
            check(row["evaluator_minimum_scene_gap_m"] < 0.,
                  "Pre-alert blind outcome changed")
        else:
            check(first_cues == [17.3] and
                  row["front_alarm"]["side"] == "front_left",
                  "Post-alert blindness lost prior front warning")
            check(row["evaluator_minimum_scene_gap_m"] > .05,
                  "Post-alert scripted halt lost margin")
        video_file = ROOT / row["video_path"]
        check(sha(video_file) == row["video_sha256"],
              "Fault video changed")
        cap = cv2.VideoCapture(str(video_file))
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1
            check(frame.ndim == 3 and frame.shape[0] > 0,
                  "Fault video frame malformed")
        cap.release()
        check(decoded >= 25, "Fault video incomplete")
        video_count += 1
    result = {
        "schema": "robot-learning-lab.ape-standoff-range-blind-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "checks_passed": checks,
        "actor_range_packets_checked": packet_count,
        "episode_videos_decoded": video_count,
        "outcome": "pass",
    }
    (v155p.OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
