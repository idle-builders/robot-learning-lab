"""Independent retained-packet/media check for V112's owner request turn."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from run import ROOT, iou, sha
from run_continuous_owner_guard_v95 import CLOSING_THRESHOLD_M_PER_0P2S
from run_shin_intrusion_posture_v324 import actor_alarm


SOURCE = Path(__file__)
REPORT = ROOT / "artifacts/ape-closed-loop-cove50-v112/report.json"
OUT = REPORT.parent / "independent-audit.json"


def main() -> None:
    report = json.loads(REPORT.read_text())
    checks = 0
    rows = []
    if len(report["rows"]) != 6:
        raise AssertionError("not six owner-turn branches")
    for row in report["rows"]:
        source = ROOT / row["source_packet_path"]
        if sha(source) != row["source_packet_sha256"]:
            raise AssertionError("changed start state")
        checks += 1
        video = ROOT / row["video_path"]
        if sha(video) != row["video_sha256"]:
            raise AssertionError("changed video")
        reader = imageio.get_reader(video)
        try:
            frame_count = reader.count_frames()
            first_frame = reader.get_data(0)
            last_frame = reader.get_data(frame_count-1)
        finally:
            reader.close()
        if frame_count < 20 or first_frame.shape != last_frame.shape:
            raise AssertionError("missing or broken video frames")
        checks += 2
        monitor_path = ROOT / row["monitor"]["path"]
        if sha(monitor_path) != row["monitor"]["sha256"]:
            raise AssertionError("changed monitor trace")
        monitor = json.loads(monitor_path.read_text())
        checks += 1
        packets = []
        for frame in monitor["frames"]:
            packet_path = ROOT / frame["packet_path"]
            if sha(packet_path) != frame["packet_sha256"]:
                raise AssertionError("changed range packet")
            with np.load(packet_path) as packet:
                packets.append((packet["front_four_points_xyz_m"].copy(),
                                packet["estimated_pose_xyyaw"].copy(),
                                float(packet["capture_time_s"])))
            if abs(packets[-1][2]-frame["capture_time_s"]) > 1e-6:
                raise AssertionError("range packet time mismatch")
            checks += 2
        first_alarm = None
        for index in range(2, len(packets)):
            old, _, _ = packets[index-2]
            current, pose, stamp = packets[index]
            motion = actor_alarm(old, current, pose)
            closing = motion["side_returns"]["left"]["closing_0p2s_m"]
            if closing >= CLOSING_THRESHOLD_M_PER_0P2S:
                first_alarm = round(stamp, 4)
                break
        expected = (None if monitor["alarm"] is None else
                    monitor["alarm"]["capture_time_s"])
        if first_alarm != expected:
            raise AssertionError("alarm does not follow saved range packets")
        checks += 1
        cue = row["actor_show_clearance_cue"]
        before_show = [frame for frame in monitor["frames"]
                       if frame["capture_time_s"] <= cue["capture_time_s"]]
        recent = [frame["left_closing_0p2s_m"]
                  for frame in before_show[-4:]
                  if frame["left_closing_0p2s_m"] is not None]
        retreat_count = sum(value <= -.012 for value in recent)
        if (cue["recent_left_closing_0p2s_m"] != recent or
                cue["retreat_packet_count"] != retreat_count or
                cue["retreat_observed"] != (retreat_count >= 2)):
            raise AssertionError("show cue does not follow saved packets")
        if row["branch"] == "parked_control":
            if not row["actor_show_allowed"]:
                raise AssertionError("parked control was not offered")
        elif row["actor_show_allowed"] != (retreat_count >= 2):
            raise AssertionError("clearance request gate mismatch")
        checks += 2
        images = [row["first_image"]]
        if row["recheck_image"] is not None:
            images.append(row["recheck_image"])
        for image in images:
            frame_path = ROOT / image["source_frame_path"]
            if sha(frame_path) != image["source_frame_sha256"]:
                raise AssertionError("changed image frame")
            with np.load(frame_path) as packet:
                if abs(float(packet["capture_time_s"])-
                       image["capture_time_s"]) > 1e-6:
                    raise AssertionError("image time mismatch")
            if image["logical_release_s"]+.0001 < (
                    image["capture_time_s"] +
                    max(.9, image["local_wall_inference_s"])):
                raise AssertionError("inference released too early")
            checks += 3
        if row["recheck_image"] is not None:
            if row["owner_yes_time_s"] < row["first_image"][
                    "logical_release_s"]:
                raise AssertionError("owner confirmation preceded recognition")
            if row["recheck_image"]["capture_time_s"] <= row[
                    "owner_yes_time_s"]:
                raise AssertionError("recheck was not fresh after owner yes")
            box1 = row["first_image"]["actor_accepted_box_before_validity"]
            box2 = row["recheck_image"]["actor_accepted_box_before_validity"]
            if row["recheck_valid"] != (box1 is not None and box2 is not None
                                        and iou(box1, box2) >= .25
                                        and monitor["alarm"] is None):
                raise AssertionError("recheck validity mismatch")
            checks += 3
        if row["branch"] == "request_ignored":
            if (row["task_result"] != "owner_turn_unresolved" or
                    row["evaluator_minimum_scene_gap_m"] >= .05 or
                    expected is None):
                raise AssertionError("ignored-request failure disappeared")
        else:
            if (row["task_result"] != "owner_assisted_visual_reacquisition"
                    or row["evaluator_minimum_scene_gap_m"] < .05 or
                    expected is not None):
                raise AssertionError("clear owner control failed")
        checks += 1
        rows.append({"seed": row["seed"], "branch": row["branch"],
                     "video_frames": frame_count,
                     "range_packets": len(packets),
                     "first_alarm_s": expected,
                     "owner_yes_s": row["owner_yes_time_s"],
                     "first_release_s": row["first_image"][
                         "logical_release_s"],
                    "show_allowed": row["actor_show_allowed"],
                    "task_result": row["task_result"]})
    audit = {"schema": "robot-learning-lab.ape-pre-show-owner-request-audit.v1",
             "source_sha256": sha(SOURCE), "report_sha256": sha(REPORT),
             "checks_passed": checks, "rows": rows}
    OUT.write_text(json.dumps(audit, indent=2) + "\n")
    print("passed", checks, "checks", OUT)


if __name__ == "__main__":
    main()
