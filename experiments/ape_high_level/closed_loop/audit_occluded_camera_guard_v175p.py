"""Audit actor RGB-D cue, range occlusion, timing and video for V174p."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from run import ROOT, sha
import audit_standoff_camera_depth_v172p as v172p
import run_occluded_camera_guard_v174p as v174p


SOURCE = Path(__file__)
REPORT = v174p.OUT / "report.json"


def main() -> None:
    report = json.loads(REPORT.read_text())
    checks = range_packets = depth_packets = videos = 0
    response_wall_deltas = []

    def check(ok: bool, reason: str) -> None:
        nonlocal checks
        if not ok:
            raise AssertionError(reason)
        checks += 1

    check(sha(Path(v174p.__file__)) == report["source_sha256"],
          "Camera-guard runner changed")
    check(sha(Path(v174p.v171p.__file__)) ==
          report["v171_source_sha256"],
          "Camera projection source changed")
    check(len(report["rows"]) == 8 and
          {(r["variant"], r["seed"],
            r["scripted_prop_stop_delay_s"]) for r in report["rows"]} ==
          {(variant, seed, delay)
           for variant in v174p.v160p.VARIANTS
           for seed in v174p.SEEDS
           for delay in v174p.RESPONSES_S},
          "Camera-guard episode matrix incomplete")
    for row in report["rows"]:
        check(row["station_x_m"] == v174p.STATION_X_M and
              sha(ROOT / row["staging_packet"]) ==
              row["staging_packet_sha256"],
              "Staged far-pose source changed")
        monitor_file = ROOT / row["monitor"]["path"]
        check(sha(monitor_file) == row["monitor"]["sha256"],
              "Concurrent monitor changed")
        monitor = json.loads(monitor_file.read_text())
        frames = monitor["frames"]
        check(len(frames) >= 50 and
              len(row["removed_by_live_packet"]) == len(frames),
              "Monitor/occlusion stream incomplete")
        range_first = [f["capture_time_s"] for f in frames
                       if f.get("front_left_first_cue")]
        camera_first = [f["capture_time_s"] for f in frames
                        if f.get("owner_depth_first_cue")]
        alarm = row["front_alarm"]
        check(not range_first and camera_first == [
            alarm["capture_time_s"]] and
              alarm["side"] == "owner_rgbd_low_region" and
              alarm["capture_time_s"] in (17.2, 17.3),
              "Actor alarm did not come from owner RGB-D")
        wall_by_time = {f["capture_time_s"]: f["wall_elapsed_s"]
                        for f in frames}
        after = round(alarm["capture_time_s"]+.2, 1)
        wall_delta = round(wall_by_time[after]-wall_by_time[
            alarm["capture_time_s"]], 5)
        check(.15 <= wall_delta <= .25,
              "Local camera-watch wall cadence changed near response")
        response_wall_deltas.append(wall_delta)
        for index, frame in enumerate(frames):
            range_file = ROOT / frame["packet_path"]
            depth_file = ROOT / frame["owner_depth_packet_path"]
            check(sha(range_file) == frame["packet_sha256"] and
                  sha(depth_file) == frame[
                      "owner_depth_packet_sha256"],
                  "Actor range or depth packet changed")
            with np.load(range_file, allow_pickle=False) as packet:
                points = packet["front_four_points_xyz_m"]
                pose = packet["estimated_pose_xyyaw"]
            n_forward = frame["front_counts"].get(
                "two_environment_points", 0)
            side = frame["front_counts"].get(
                "four_environment_points", 0)
            check(len(points) == n_forward+side and
                  not np.any(v174p.front_left_roi(
                      points[:n_forward], pose)),
                  "Occluded forward sensor returned into target field")
            with np.load(depth_file, allow_pickle=False) as depth:
                support, p10, valid = v172p.low_region_from_saved(depth)
                time_s = float(depth["capture_time_s"])
            check(abs(time_s-frame["capture_time_s"]) < 1e-6 and
                  support == frame["owner_depth_low_support"] and
                  p10 == frame["owner_depth_low_p10_x_m"] and
                  valid == frame["owner_depth_valid_samples"],
                  "Live camera statistic differs from saved depth")
            if index >= 2 and p10 is not None:
                older = frames[index-2]["owner_depth_low_p10_x_m"]
                if older is not None:
                    closing = round(older-p10, 5)
                    check(frame["owner_depth_closing_0p2s_m"] == closing and
                          frame["owner_depth_cue_now"] == (
                              closing >=
                              v174p.v171p.MIN_CLOSING_M_PER_0P2S),
                          "Live camera closing cue changed")
            range_packets += 1
            depth_packets += 1
        check(row["task_result"] == "owner_turn_unresolved" and
              row["first_image"] is None and
              row["confirm_image"] is None and
              row["actor_first_stable_range_s"] is None,
              "Camera warning incorrectly completed item task")
        gap = row["evaluator_minimum_scene_gap_m"]
        if row["scripted_prop_stop_delay_s"] is None:
            check(gap < .05, "Ignored prop no longer crosses margin")
        else:
            check(gap > .05, "Prompt scripted prop halt lost margin")
        video_file = ROOT / row["video_path"]
        check(sha(video_file) == row["video_sha256"],
              "Complete episode video changed")
        capture = cv2.VideoCapture(str(video_file))
        decoded = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            decoded += 1
            check(image.ndim == 3 and image.shape[0] > 0,
                  "Episode video frame malformed")
        capture.release()
        check(decoded >= 25, "Episode video incomplete")
        videos += 1
    result = {
        "schema": "robot-learning-lab.ape-occluded-camera-guard-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "checks_passed": checks,
        "range_packets_checked": range_packets,
        "owner_depth_packets_recomputed": depth_packets,
        "episode_videos_decoded": videos,
        "camera_warning_to_0p2s_wall_delta_min_s": min(
            response_wall_deltas),
        "camera_warning_to_0p2s_wall_delta_max_s": max(
            response_wall_deltas),
        "outcome": "pass",
    }
    (v174p.OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
