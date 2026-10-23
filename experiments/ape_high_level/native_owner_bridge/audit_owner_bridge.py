"""Verify saved owner episodes, native packet provenance and comparison videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/sensors"))
from cove_rgb_transfer import source_checks  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode video {path}")
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.size == 0:
            raise RuntimeError(f"Empty frame {count}: {path}")
        count += 1
    capture.release()
    if not count:
        raise RuntimeError(f"No frames: {path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text())
    for key, path in (("owner_actor_source_sha256",
                       ROOT / "experiments/ape_high_level/closed_loop/run.py"),
                      ("adapter_source_sha256",
                       ROOT / "experiments/ape_high_level/native_owner_bridge/replay_owner_packets.py"),
                      ("native_qpos_source_sha256", ROOT / report["native_qpos_source"]),
                      ("native_odometry_source_sha256", ROOT / report["native_odometry_source"])):
        if sha(path) != report[key]:
            raise ValueError(f"Source hash mismatch: {path}")
    if len(report["rows"]) != 12:
        raise ValueError("Expected 2 seeds × 2 presence cases × 3 motion arms")
    checked = []
    for row in report["rows"]:
        trace_path = ROOT / row["episode_trace"]
        video_path = ROOT / row["episode_video"]
        if sha(trace_path) != row["episode_trace_sha256"]:
            raise ValueError(f"Episode trace hash mismatch: {trace_path}")
        if sha(video_path) != row["episode_video_sha256"]:
            raise ValueError(f"Episode video hash mismatch: {video_path}")
        trace = json.loads(trace_path.read_text())
        if trace["outcome"] != row["outcome"]:
            raise ValueError("Outcome differs between report and episode trace")
        frames = decode(video_path)
        if frames != len(trace["events"]):
            raise ValueError(f"Video and event count differ: {video_path}")
        packet_count = None
        if row.get("native_packet_trace"):
            packets_path = ROOT / row["native_packet_trace"]
            if sha(packets_path) != row["native_packet_trace_sha256"]:
                raise ValueError(f"Packet hash mismatch: {packets_path}")
            packets = json.loads(packets_path.read_text())
            packet_count = len(packets)
            if bool(packet_count) == bool(row["absent"]):
                raise ValueError("Unexpected native packet count for present/absent case")
            if packets and packets[-1]["source_end_frame"] != row["native_source_end_frame"]:
                raise ValueError("Native source cursor mismatch")
        checked.append({"mode": row["mode"], "seed": row["seed"],
                        "absent": row["absent"], "frames": frames,
                        "events": len(trace["events"]),
                        "native_action_packets": packet_count,
                        "unsafe_obstacle_overlap": row["unsafe_obstacle_overlap"]})
    composites = {}
    for seed in (85, 86):
        path = report_path.parent / f"seed{seed}-present-pose-source-comparison.mp4"
        composites[str(seed)] = {"path": str(path.relative_to(ROOT)),
                                 "sha256": sha(path), "decoded_frames": decode(path)}
    audit = {"schema": "robot-learning-lab.native-owner-bridge-audit.v1",
             "report": str(report_path.relative_to(ROOT)),
             "report_sha256": sha(report_path),
             "audit_code_sha256": sha(Path(__file__)),
             "historical_sensor_and_model_sources": source_checks(),
             "episodes": checked, "composite_videos": composites,
             "all_episode_hashes_and_video_event_counts_match": True}
    output = report_path.parent / "audit.json"
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"episodes": len(checked),
                      "video_frames": sum(row["frames"] for row in checked),
                      "composites": composites}, indent=2))


if __name__ == "__main__":
    main()
