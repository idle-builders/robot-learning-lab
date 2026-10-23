"""Independent source, packet, decision and image audit for V145p."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-closed-loop-cove50-reverse-progress-gate"
REPORT = OUT / "report.json"


def sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_xy(qpos: np.ndarray, frame: int) -> np.ndarray:
    angle = -2 * math.atan2(float(qpos[0, 6]), float(qpos[0, 3]))
    c, s = math.cos(angle), math.sin(angle)
    x, y = qpos[frame, :2] - qpos[0, :2]
    return np.array((c * x - s * y, s * x + c * y))


def main() -> None:
    report = json.loads(REPORT.read_text())
    checks = 0

    def check(predicate: bool, explanation: str) -> None:
        nonlocal checks
        if not predicate:
            raise AssertionError(explanation)
        checks += 1

    for path, digest in (
        (ROOT / "experiments/ape_high_level/closed_loop/"
         "check_reverse_meeting_progress_v145p.py", report["source_sha256"]),
        (ROOT / report["historical_body"], report["historical_body_sha256"]),
        (ROOT / report["staging_packet"], report["staging_packet_sha256"]),
        (ROOT / "artifacts/ape-closed-loop-cove50-v98/"
         "live-moving-owner-offer-audit.json", report["v98_report_sha256"]),
    ):
        check(sha(path) == digest, f"Source changed: {path}")
    check(len(report["rows"]) == 2, "Expected two unique native traces")
    check(math.isclose(report["progress_arrival_threshold_m"], .10,
                       abs_tol=1e-10),
          "Arrival threshold changed")
    for row in report["rows"]:
        native = ROOT / row["native_trace"]
        check(sha(native) == row["native_trace_sha256"],
              "Native trace hash changed")
        check(sha(ROOT / row["native_report"]) ==
              row["native_report_sha256"], "Native report hash changed")
        with np.load(native, allow_pickle=False) as saved:
            qpos = saved["qpos"]
            dt = float(saved["dt"])
        frame = round(row["packet_time_s"] / dt)
        check(frame == 60, "Native decision frame changed")
        check(np.allclose(local_xy(qpos, frame),
                          row["actual_local_xy_m_evaluator_only"], atol=1e-5),
              "Native decision displacement changed")
        check(np.allclose(local_xy(qpos, len(qpos)-1),
                          row["actual_final_local_xy_m_evaluator_only"],
                          atol=1e-5), "Native end displacement changed")
        check(row["command_clock_assumed_arrival"],
              "Command-clock comparison changed")
        check(row["actual_backward_progress_m_evaluator_only"] < .10,
              "Native actor unexpectedly reached station")
        for packet in row["packet_progress"].values():
            check(not packet["progress_gate_reached"],
                  "Packet unexpectedly calls station reached")
            check(packet["estimated_backward_progress_m"] < .10,
                  "Packet progress exceeds gate")
        for case, scene in row["owner_show_if_command_clock_assumes_arrival"].items():
            image_path = ROOT / scene["owner_image"]["actor_frame_path"]
            check(sha(image_path) == scene["owner_image"]["actor_frame_sha256"],
                  "Owner image hash changed")
            with np.load(image_path, allow_pickle=False) as image:
                check(abs(float(image["capture_time_s"]) - 17.4) < 1e-6,
                      "Owner image time changed")
                check(image["rgb"].ndim == 3 and image["depth_m"].ndim == 2,
                      "Owner RGB-D packet malformed")
            check(scene["generic_taught_item_accepted"] and
                  scene["evaluator_item_box_iou"] > .95,
                  "Owner item result changed")
            check(abs(min(sample["evaluator_gap_m"]
                          for sample in scene["sampled_path"]) -
                      scene["minimum_scene_gap_m"]) < .002,
                  "Saved show gap differs from trajectory samples")
            check((scene["minimum_scene_gap_m"] > .05) == (case == "clear"),
                  "Clear/blocked show gap ordering changed")
    video_report_path = OUT / "video-report.json"
    video_report = json.loads(video_report_path.read_text())
    check(video_report["v145_report_sha256"] == sha(REPORT),
          "Video references a different experiment")
    check(sha(ROOT / video_report["native_trace"]) ==
          video_report["native_trace_sha256"],
          "Video native source changed")
    video_path = ROOT / video_report["video_path"]
    check(sha(video_path) == video_report["video_sha256"],
          "Video hash changed")
    check(len(video_report["frames"]) == 34,
          "Video frame metadata changed")
    check(min(row["saved_reverse_outcome_transfer"]["evaluator_gap_m"]
              for row in video_report["frames"]) < .05,
          "Transferred failure no longer shows close gap")
    check(min(row["intended"]["evaluator_gap_m"]
              for row in video_report["frames"]) > .05,
          "Intended trajectory no longer clears scene")
    capture = cv2.VideoCapture(str(video_path))
    check(capture.isOpened(), "Video cannot be opened")
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded += 1
        check(frame.shape[0] > 0 and frame.shape[1] > 0,
              "Decoded video frame malformed")
    capture.release()
    check(decoded == len(video_report["frames"]),
          "Video decode count differs from trace")
    result = {
        "schema": "robot-learning-lab.ape-reverse-meeting-progress-audit.v1",
        "report_sha256": sha(REPORT),
        "checks_passed": checks,
        "outcome": "pass",
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
