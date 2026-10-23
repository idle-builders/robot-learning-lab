"""Recheck RGB/depth owner-guard evidence, including the failed optical fault."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))
import audit_robocasa_first_return_person_reacquire_v578 as geometry_audit

OUT = ROOT / "artifacts/ape-rgb-depth-consistency/v3"
BASELINE = ROOT / "artifacts/ape-depth-guard-faults/v1/report.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packet_sha(packet):
    return hashlib.sha256(json.dumps(packet, separators=(",", ":")).encode()).hexdigest()


def main():
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    baseline = json.loads(BASELINE.read_text())
    sources = {
        "runner": ROOT / "experiments/ape_high_level/rgb_depth_consistency/run_rgb_consistency.py",
        "geometry_audit": Path(geometry_audit.__file__),
        "owner_turn": ROOT / "experiments/ape_high_level/closed_loop/robocasa_first_return_person_reacquire_turn_v597.py",
    }
    source_checks = {name: report["source_sha256"][name] == sha(path)
                     for name, path in sources.items()}
    rows = []
    for item in report["rows"]:
        name, ep = item["name"], item["episode"]
        sensor = ep["rgb_depth_consistency_sensor"]
        packet_path = ROOT / sensor["packet_trace"]
        packets = json.loads(packet_path.read_text())
        mid_trace = ep["raw_and_delivered_mid_range_packet_trace"]
        mid_path = ROOT / mid_trace["path"]
        mid_packets = json.loads(mid_path.read_text())
        geometry = geometry_audit.audit_episode(ep)
        geometry.pop("geometry_samples", None)
        frozen_depth_pass = True
        first_depth = packets[0]["actor_delivered_depth_packet_72_world_bearings_m"]
        for packet in packets:
            origin = np.asarray(packet["camera_xyz_m"][:2])
            target = np.asarray(packet["actor_track_center_xy_m"])
            bearing = math.atan2(target[1] - origin[1],
                                 target[0] - origin[0])
            angles = np.linspace(-math.pi, math.pi, 72, endpoint=False)
            delta = (angles - bearing + math.pi) % (2 * math.pi) - math.pi
            mask = np.abs(delta) <= math.radians(40.)
            live = packet["packet_72_world_bearings_m"]
            delivered = packet["actor_delivered_depth_packet_72_world_bearings_m"]
            frozen_depth_pass &= all(
                delivered[i] == (first_depth[i] if mask[i] else live[i])
                for i in range(72))
            frozen_depth_pass &= (
                packet_sha(live) == packet["raw_live_depth_packet_sha256"] and
                packet_sha(delivered) == packet[
                    "actor_delivered_depth_packet_sha256"])
        sample_checks = []
        for sample in sensor["rgb_frame_samples"]:
            path = ROOT / sample["path"]
            sample_checks.append(sha(path) == sample["sha256"] and
                                 cv2.imread(str(path)) is not None)
        movie = Path(ep["video"]["path"])
        cap = cv2.VideoCapture(str(movie))
        frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (528, 1040, 3):
                raise AssertionError(f"Bad frame shape in {movie}")
            frames += 1
        cap.release()
        cancels = [r for r in ep["trace"] if r["stage"] ==
                   "owner_reacquire_image_cancelled"]
        rows.append({
            "name": name,
            "scene_matches_baseline": ep["scene_fingerprint"] ==
                baseline["episode"]["scene_fingerprint"],
            "packet_hash_pass": sha(packet_path) == sensor["packet_trace_sha256"],
            "packet_count_pass": len(packets) == sensor["packet_count"],
            "mid_packet_trace_pass": sha(mid_path) == mid_trace["sha256"] and
                len(mid_packets) == mid_trace["packet_count"] and
                all(packet_sha(row["raw_mid_range_packet"]) == row[
                        "raw_mid_range_packet_sha256"] and
                    packet_sha(row["actor_delivered_mid_range_packet"]) == row[
                        "actor_delivered_mid_range_packet_sha256"]
                    for row in mid_packets),
            "rgb_samples_pass": all(sample_checks),
            "depth_owner_sector_fault_pass": frozen_depth_pass,
            "raw_rgb_unique": len({p["raw_rgb_sha256"] for p in packets}),
            "delivered_rgb_unique": len({p["actor_delivered_rgb_sha256"]
                                         for p in packets}),
            "camera_pose_reference_resets": sum(
                p["rgb_reference_reset_for_camera_pose"] for p in packets),
            "rgb_alarm_packet": sensor["first_rgb_consistency_alarm_packet"],
            "fused_invalidations": ep["fused_owner_sector_invalidations"],
            "image_cancels": len(cancels),
            "person_gap_m": geometry["person_exact_nearest"]["gap_m"],
            "geometry_matches_run": abs(
                geometry["person_exact_nearest"]["gap_m"] -
                item["geometry_audit"]["person_exact_nearest"]["gap_m"]) < 1e-8,
            "video_pass": sha(movie) == ep["video"]["sha256"] and
                frames == ep["video"]["frames"] and
                geometry["video_sha_pass"] and geometry["video_frames_pass"],
            "decoded_frames": frames,
            "outcome": ep["outcome"],
            "maximum_packet_start_gap_wall_s":
                sensor["maximum_packet_start_gap_wall_s"],
        })
    by_name = {row["name"]: row for row in rows}
    moving = by_name["moving_live_rgb"]
    stationary = by_name["stationary_live_rgb"]
    frozen = by_name["moving_frozen_rgb"]
    behavior_checks = {
        "live_rgb_rescues_shared_fault": moving["rgb_alarm_packet"] is not None and
            moving["image_cancels"] > 0 and moving["person_gap_m"] >= .05,
        "stationary_owner_no_false_interrupt": stationary["rgb_alarm_packet"] is None and
            stationary["image_cancels"] == 0 and
            stationary["outcome"] == "owner_confirmed_metric_find",
        "shared_rgb_fault_remains_unsafe": frozen["rgb_alarm_packet"] is None and
            frozen["image_cancels"] == 0 and frozen["person_gap_m"] < .05,
        "frozen_rgb_raw_still_changes": frozen["raw_rgb_unique"] > 10 and
            frozen["delivered_rgb_unique"] == 1,
        "stationary_reference_resets_for_gaze":
            stationary["camera_pose_reference_resets"] >= 2,
    }
    evidence_pass = all(source_checks.values()) and all(behavior_checks.values()) and all(
        row[key] for row in rows for key in (
            "scene_matches_baseline", "packet_hash_pass", "packet_count_pass",
            "mid_packet_trace_pass", "rgb_samples_pass", "depth_owner_sector_fault_pass",
            "geometry_matches_run", "video_pass"))
    audit = {"schema": "robot-learning-lab.rgb-depth-consistency-audit.v1",
             "report_sha256": sha(report_path), "baseline_sha256": sha(BASELINE),
             "source_checks": source_checks, "behavior_checks": behavior_checks,
             "rows": rows, "evidence_pass": evidence_pass}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"evidence_pass": evidence_pass,
                      "behavior_checks": behavior_checks,
                      "rows": [{k: row[k] for k in (
                          "name", "person_gap_m", "decoded_frames",
                          "maximum_packet_start_gap_wall_s")}
                               for row in rows]}, indent=2))
    if not evidence_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
