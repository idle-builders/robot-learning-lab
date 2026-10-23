"""Independent packet, event, geometry and media audit for V158p."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from run import ROOT, sha
from screen_standoff_front_cue_v149p import front_left_stat
import run_standoff_coverage_health_v158p as v158p


SOURCE = Path(__file__)
REPORT = v158p.OUT / "report.json"


def main() -> None:
    report = json.loads(REPORT.read_text())
    checks = packets = videos = 0

    def check(ok: bool, reason: str) -> None:
        nonlocal checks
        if not ok:
            raise AssertionError(reason)
        checks += 1

    check(sha(Path(v158p.__file__)) == report["source_sha256"],
          "Coverage-health source changed")
    check(len(report["rows"]) == 4, "Coverage-health matrix incomplete")
    check({(r["seed"], r["scripted_prop_stop_delay_s"])
           for r in report["rows"]} ==
          {(seed, delay) for seed in v158p.SEEDS
           for delay in v158p.RESPONSES_S},
          "Coverage-health matrix changed")
    for row in report["rows"]:
        check(sha(ROOT / row["staging_packet"]) ==
              row["staging_packet_sha256"], "Staging packet changed")
        monitor_file = ROOT / row["monitor"]["path"]
        check(sha(monitor_file) == row["monitor"]["sha256"],
              "Monitor changed")
        monitor = json.loads(monitor_file.read_text())
        frames = monitor["frames"]
        check(len(frames) >= 50, "Monitor ended early")
        health_events = [f for f in frames
                         if f.get("front_left_coverage_loss_first")]
        check(len(health_events) == 1 and
              health_events[0]["capture_time_s"] == 17.2,
              "Coverage event timing changed")
        check(row["front_alarm"]["side"] ==
              "front_left_coverage_lost" and
              row["front_alarm"]["capture_time_s"] == 17.2,
              "Coverage event did not reach response interface")
        check(not any(f.get("front_left_first_cue") for f in frames),
              "A motion cue rather than coverage event triggered")
        check(len(row["removed_by_packet"]) >= 40 and
              sum(p["removed_front_left_returns"] for p in
                  row["removed_by_packet"]) > 0,
              "Injected sensor loss missing")
        for frame in frames:
            path = ROOT / frame["packet_path"]
            check(sha(path) == frame["packet_sha256"],
                  "Actor packet changed")
            with np.load(path, allow_pickle=False) as packet:
                support, p10 = front_left_stat(
                    packet["front_four_points_xyz_m"],
                    packet["estimated_pose_xyyaw"])
            check(support == frame["front_left_support"] and
                  p10 == frame["front_left_p10_x_m"],
                  "Actor statistic differs from source packet")
            if frame["capture_time_s"]+1e-6 >= 17.2:
                check(support == 0, "Outage packet retained front returns")
            packets += 1
        check(row["task_result"] == "owner_turn_unresolved" and
              row["first_image"] is None and
              row["confirm_image"] is None,
              "Blind actor made a visual success claim")
        if row["scripted_prop_stop_delay_s"] is None:
            check(row["evaluator_minimum_scene_gap_m"] < 0.,
                  "Ignored coverage request no longer overlaps")
        else:
            check(row["evaluator_minimum_scene_gap_m"] > .05,
                  "Cooperative coverage response lost margin")
        video_file = ROOT / row["video_path"]
        check(sha(video_file) == row["video_sha256"],
              "Episode video changed")
        cap = cv2.VideoCapture(str(video_file))
        decoded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded += 1
            check(frame.ndim == 3 and frame.shape[0] > 0,
                  "Episode video frame malformed")
        cap.release()
        check(decoded >= 25, "Episode video incomplete")
        videos += 1
    result = {
        "schema": "robot-learning-lab.ape-standoff-coverage-health-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "checks_passed": checks,
        "actor_range_packets_checked": packets,
        "episode_videos_decoded": videos,
        "outcome": "pass",
    }
    (v158p.OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
