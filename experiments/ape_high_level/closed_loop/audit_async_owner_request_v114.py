"""Independent packet/media and concurrent timing audit for V114."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from run import ROOT, iou, sha
from run_continuous_owner_guard_v95 import CLOSING_THRESHOLD_M_PER_0P2S
from run_shin_intrusion_posture_v324 import actor_alarm


SOURCE = Path(__file__)
REPORT = ROOT / "artifacts/ape-closed-loop-cove50-v114/report.json"
OUT = REPORT.parent / "independent-audit.json"


def main() -> None:
    report = json.loads(REPORT.read_text())
    if len(report["rows"]) != 6:
        raise AssertionError("expected six paired owner turns")
    checks = 0
    rows = []
    for row in report["rows"]:
        source = ROOT / row["source_packet_path"]
        if sha(source) != row["source_packet_sha256"]:
            raise AssertionError("changed V98 start packet")
        checks += 1
        video = ROOT / row["video_path"]
        if sha(video) != row["video_sha256"]:
            raise AssertionError("changed video")
        reader = imageio.get_reader(video)
        try:
            count = reader.count_frames()
            shape = reader.get_data(0).shape
            if count < 20 or reader.get_data(count-1).shape != shape:
                raise AssertionError("incomplete video")
        finally:
            reader.close()
        checks += 2
        monitor_path = ROOT / row["monitor"]["path"]
        if sha(monitor_path) != row["monitor"]["sha256"]:
            raise AssertionError("changed monitor trace")
        monitor = json.loads(monitor_path.read_text())
        frames = monitor["frames"]
        checks += 1
        stamps = [frame["capture_time_s"] for frame in frames]
        wall = [frame["wall_elapsed_s"] for frame in frames]
        if not all(abs(b-a-.1) < 1e-5 for a, b in zip(stamps, stamps[1:])):
            raise AssertionError("simulation monitor skipped a packet")
        if not all(b > a for a, b in zip(wall, wall[1:])):
            raise AssertionError("wall packet times not monotonic")
        checks += 2
        packets = []
        for frame in frames:
            packet_path = ROOT / frame["packet_path"]
            if sha(packet_path) != frame["packet_sha256"]:
                raise AssertionError("changed range packet")
            with np.load(packet_path) as packet:
                points = packet["front_four_points_xyz_m"].copy()
                pose = packet["estimated_pose_xyyaw"].copy()
                stamp = float(packet["capture_time_s"])
            if abs(stamp-frame["capture_time_s"]) > 1e-5:
                raise AssertionError("packet timestamp mismatch")
            packets.append((points, pose, stamp))
            checks += 2
        first_alarm = None
        for index in range(2, len(packets)):
            motion = actor_alarm(packets[index-2][0], packets[index][0],
                                 packets[index][1])
            closing = motion["side_returns"]["left"]["closing_0p2s_m"]
            if closing >= CLOSING_THRESHOLD_M_PER_0P2S:
                first_alarm = round(packets[index][2], 4)
                break
        reported_alarm = (None if monitor["alarm"] is None else
                          monitor["alarm"]["capture_time_s"])
        if first_alarm != reported_alarm:
            raise AssertionError("alarm does not follow retained range")
        checks += 1
        cue = row["actor_show_clearance_cue"]
        before = [frame for frame in frames
                  if frame["capture_time_s"] <= cue["capture_time_s"]]
        recent = [frame["left_closing_0p2s_m"]
                  for frame in before[-4:]
                  if frame["left_closing_0p2s_m"] is not None]
        retreat = sum(value <= -.012 for value in recent)
        if (cue["recent_left_closing_0p2s_m"] != recent or
                cue["retreat_packet_count"] != retreat or
                cue["retreat_observed"] != (retreat >= 2)):
            raise AssertionError("cue not derived from packets")
        if row["branch"] == "parked_control":
            if not row["actor_show_allowed"]:
                raise AssertionError("parked control declined")
        elif row["actor_show_allowed"] != (retreat >= 2):
            raise AssertionError("receding cue decision mismatch")
        checks += 2
        first = row["first_image"]
        for image in (first, row["recheck_image"]):
            if image is None:
                continue
            packet_path = ROOT / image["source_frame_path"]
            if sha(packet_path) != image["source_frame_sha256"]:
                raise AssertionError("changed image packet")
            with np.load(packet_path) as packet:
                stamp = float(packet["capture_time_s"])
            if abs(stamp-image["capture_time_s"]) > 1e-5:
                raise AssertionError("image timestamp mismatch")
            if not (image["submitted_wall_elapsed_s"] <=
                    image["worker_start_wall_elapsed_s"] <=
                    image["model_done_wall_elapsed_s"] <=
                    image["worker_finish_wall_elapsed_s"] <=
                    image["observed_finish_wall_elapsed_s"]):
                raise AssertionError("noncausal worker timeline")
            if image["capture_time_s"] > image["release_time_s"]:
                raise AssertionError("image released before capture")
            checks += 4
        start = first["worker_start_wall_elapsed_s"]
        finish = first["worker_finish_wall_elapsed_s"]
        overlap = sum(start < stamp < finish for stamp in wall)
        if overlap != row["monitor_packets_during_first_inference"] or overlap < 2:
            raise AssertionError("range monitor did not overlap image worker")
        during = [b-a for a,b in zip(wall,wall[1:])
                  if start < a and b < finish]
        if round(max(during),5) != row[
                "monitor_max_wall_interval_during_first_inference_s"]:
            raise AssertionError("worker-time range interval mismatch")
        if round(max(np.diff(wall)),5) != row[
                "monitor_max_wall_interval_s"]:
            raise AssertionError("overall range interval mismatch")
        checks += 3
        second = row["recheck_image"]
        if second is not None:
            if (row["owner_yes_time_s"] < first["release_time_s"] or
                    second["capture_time_s"] <= row["owner_yes_time_s"]):
                raise AssertionError("owner yes/recheck ordering failed")
            box1 = first["actor_accepted_box_before_validity"]
            box2 = second["actor_accepted_box_before_validity"]
            valid = (box1 is not None and box2 is not None and
                     iou(box1,box2) >= .25 and monitor["alarm"] is None)
            if row["recheck_valid"] != valid:
                raise AssertionError("recheck outcome mismatch")
            checks += 2
        if row["branch"] == "request_ignored":
            if (row["task_result"] != "owner_turn_unresolved" or
                    reported_alarm is None or
                    row["evaluator_minimum_scene_gap_m"] >= .05):
                raise AssertionError("ignored-request failure disappeared")
        else:
            if (row["task_result"] != "owner_assisted_visual_reacquisition"
                    or reported_alarm is not None or
                    row["evaluator_minimum_scene_gap_m"] < .05):
                raise AssertionError("clear branch failed")
        checks += 1
        rows.append({"seed": row["seed"], "branch": row["branch"],
                     "video_frames": count, "range_packets": len(frames),
                     "first_worker_overlap_packets": overlap,
                     "first_worker_wall_s": round(finish-start, 5),
                     "max_wall_range_interval_during_worker_s":
                         row["monitor_max_wall_interval_during_first_inference_s"],
                     "alarm_s": reported_alarm,
                     "task_result": row["task_result"]})
    audit = {"schema": "robot-learning-lab.ape-async-owner-request-audit.v1",
             "source_sha256": sha(SOURCE), "report_sha256": sha(REPORT),
             "checks_passed": checks, "rows": rows}
    OUT.write_text(json.dumps(audit, indent=2)+"\n")
    print("passed", checks, "checks", OUT)


if __name__ == "__main__":
    main()
