"""Independent packet, task-state and media audit for V153p recovery."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from run import ROOT, sha
import run_standoff_yield_recovery_v153p as v153p


SOURCE = Path(__file__)
OUT = v153p.OUT
REPORT = OUT / "report.json"


def main() -> None:
    report = json.loads(REPORT.read_text())
    checks = 0
    packet_count = 0
    video_count = 0
    confirm_worker_packets = 0

    def check(ok: bool, reason: str) -> None:
        nonlocal checks
        if not ok:
            raise AssertionError(reason)
        checks += 1

    check(sha(Path(v153p.__file__)) == report["source_sha256"],
          "Recovery source changed")
    check(len(report["rows"]) == 12, "Recovery matrix incomplete")
    check({(r["seed"], r["station_x_m"],
            r["scripted_prop_stop_delay_s"]) for r in report["rows"]} == {
              (seed, x, delay) for seed in (170, 172)
              for x in (-.30, -.22) for delay in (.2, .5, None)},
          "Recovery cases changed")
    for row in report["rows"]:
        check(sha(ROOT / row["staging_packet"]) ==
              row["staging_packet_sha256"], "Staging packet changed")
        check(row["front_alarm"]["capture_time_s"] == 17.3,
              "Front alert timing changed")
        monitor_path = ROOT / row["monitor"]["path"]
        check(sha(monitor_path) == row["monitor"]["sha256"],
              "Recovery monitor changed")
        monitor = json.loads(monitor_path.read_text())
        frames = monitor["frames"]
        check(len(frames) >= 50, "Monitor stopped before recovery end")
        for frame in frames:
            check(sha(ROOT / frame["packet_path"]) == frame["packet_sha256"],
                  "Recovery range packet changed")
            packet_count += 1
        for image in (row["first_image"], row["confirm_image"]):
            if image is not None:
                check(sha(ROOT / image["source_frame_path"]) ==
                      image["source_frame_sha256"],
                      "Recovery image changed")
                check(image["capture_time_s"] > row["front_alarm"][
                    "capture_time_s"], "Stale pre-alarm image used")
        video = ROOT / row["video_path"]
        check(sha(video) == row["video_sha256"],
              "Recovery video changed")
        capture = cv2.VideoCapture(str(video))
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            check(frame.ndim == 3 and frame.shape[0] > 0,
                  "Recovery video frame malformed")
        capture.release()
        check(decoded >= 25, "Recovery video incomplete")
        video_count += 1
        gap = row["evaluator_minimum_scene_gap_m"]
        should_claim = (row["scripted_prop_stop_delay_s"] == .2 or
                        (row["station_x_m"] == -.30 and
                         row["scripted_prop_stop_delay_s"] == .5))
        if should_claim:
            check(row["task_result"] ==
                  "fresh_unattributed_item_view_after_yield",
                  "Supported yield did not produce fresh item view")
            check(row["actor_first_stable_range_s"] is not None and
                  row["actor_first_stable_range_s"] > 17.3,
                  "Fresh range gate missing")
            check(row["first_image_valid"] and row["confirm_image_valid"]
                  and not row["renewed_front_cue_after_first_capture"],
                  "Item confirmation not grounded in healthy packets")
            check(row["first_image"]["capture_time_s"] >=
                  row["actor_first_stable_range_s"] and
                  row["confirm_image"]["capture_time_s"] >
                  row["first_image"]["release_time_s"],
                  "Recovery image order wrong")
            check(row["range_packets_during_first_inference"] >= 1,
                  "Monitor absent during first inference")
            confirm = row["confirm_image"]
            during_confirm = sum(
                confirm["worker_start_wall_elapsed_s"] <
                frame["wall_elapsed_s"] <
                confirm["worker_finish_wall_elapsed_s"]
                for frame in frames)
            check(during_confirm >= 1,
                  "Monitor absent during confirmation inference")
            confirm_worker_packets += during_confirm
            check(gap > .05, "Claim produced below-margin body gap")
        else:
            check(row["task_result"] == "owner_turn_unresolved" and
                  row["first_image"] is None and
                  row["confirm_image"] is None,
                  "Unsupported branch made an item claim")
            check(row["actor_first_stable_range_s"] is None,
                  "Unsupported branch falsely saw clear range")
            if row["scripted_prop_stop_delay_s"] is None:
                check(gap < 0., "Ignored prop no longer overlaps")
            else:
                check(0. < gap < .05,
                      "Near half-second response outcome changed")
        check(min(s["evaluator_gap_m"] for s in row["samples_10hz"])
              <= gap+.002, "Saved gap samples inconsistent")
    result = {
        "schema": "robot-learning-lab.ape-yield-recovery-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "checks_passed": checks,
        "actor_range_packets_checked": packet_count,
        "episode_videos_decoded": video_count,
        "packets_during_confirmation_inference": confirm_worker_packets,
        "outcome": "pass",
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
