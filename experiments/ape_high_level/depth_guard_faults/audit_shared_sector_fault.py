"""Audit the shared owner-sector depth/range fault against live controls."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-depth-guard-faults/v1"
WARM = ROOT / "artifacts/ape-depth-guard-choice/warm-v2/report.json"
IDEAL_RANGE = ROOT / "artifacts/ape-robocasa-reacquire-frozen-sector-dual-v597/report.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report = json.loads((OUT / "report.json").read_text())
    warm = json.loads(WARM.read_text())
    ideal = json.loads(IDEAL_RANGE.read_text())
    assert report["schema"] == "robot-learning-lab.shared-owner-sector-fault.v1"
    files = {
        "runner": ROOT / "experiments/ape_high_level/depth_guard_faults/run_shared_sector_fault.py",
        "cold_depth_adapter": ROOT / "experiments/ape_high_level/depth_guard_choice/run_depth_guard.py",
        "warm_depth_adapter": ROOT / "experiments/ape_high_level/depth_guard_choice/run_warm_depth_guard.py",
        "owner_turn": ROOT / "experiments/ape_high_level/closed_loop/robocasa_first_return_person_reacquire_turn_v597.py",
        "joined_route": ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_person_joined_v497.py",
        "geometry_audit": ROOT / "experiments/ape_high_level/closed_loop/audit_robocasa_first_return_person_reacquire_v578.py",
        "config": ROOT / "experiments/ape_high_level/closed_loop/robocasa_layout4_safe_side_v473.json",
        "prior_map": ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz",
    }
    for key, path in files.items():
        assert sha(path) == report["source_sha256"][key], key
    assert report["source_sha256"]["owner_turn"] == warm["source_sha256"]["owner_turn"]
    episode = report["episode"]
    geometry = report["geometry_audit"]
    assert episode["scene_fingerprint"] == warm["episode"]["scene_fingerprint"]
    assert episode["scene_fingerprint"] == ideal["episodes"][0]["scene_fingerprint"]
    sensor = episode["depth_guard_sensor"]
    packet_file = ROOT / sensor["packet_trace"]
    assert sha(packet_file) == sensor["packet_trace_sha256"]
    packets = json.loads(packet_file.read_text())
    assert len(packets) == sensor["packet_count"] == sensor["owner_sector_fault_packets"]
    assert len(packets) > 50
    frozen = packets[0]["actor_delivered_packet_72_world_bearings_m"]
    assert len(frozen) == 72
    assert len({row["depth_sha256"] for row in packets}) > 10
    assert sum(row["live_to_delivered_changed_beams"] > 0 for row in packets) > 50
    for row in packets:
        live = row["packet_72_world_bearings_m"]
        delivered = row["actor_delivered_packet_72_world_bearings_m"]
        origin = np.asarray(row["camera_xyz_m"][:2])
        target = np.asarray(row["actor_track_center_xy_m"])
        bearing = math.atan2(target[1] - origin[1], target[0] - origin[0])
        angles = np.linspace(-math.pi, math.pi, 72, endpoint=False)
        delta = (angles - bearing + math.pi) % (2 * math.pi) - math.pi
        mask = np.abs(delta) <= math.radians(40.)
        assert int(mask.sum()) == row["frozen_owner_beams"]
        assert all(delivered[i] == frozen[i] if mask[i] else delivered[i] == live[i]
                   for i in range(72))
        assert row["delivered_owner_sector_return_count"] > 0
        assert row["packet_supported_beams"] >= 20
    for sample in sensor["sample_depth_frames"]:
        path = ROOT / sample["path"]
        assert sha(path) == sample["sha256"]
        depth = np.load(path, allow_pickle=False)
        assert depth.shape == (480, 640)
        index = sample["index"] if sample["index"] >= 0 else len(packets) - 1
        assert hashlib.sha256(depth.tobytes()).hexdigest() == packets[index]["depth_sha256"]
    assert sensor["prewarm"] is not None
    assert episode["frozen_mid_guard_packets"] > 0
    assert episode["outcome"] == "owner_track_uncertain_before_show"
    assert not any(row.get("stage") == "owner_reacquire_image_cancelled"
                   for row in episode["trace"])
    upper_ticks = [row for row in episode["trace"]
                   if row.get("stage") == "owner_reacquire_guard_tick"
                   and "upper_owner_sector_changed_beams" in row]
    assert upper_ticks and all(row["upper_owner_sector_motion_streak"] == 0
                               for row in upper_ticks)
    assert geometry["scene_fingerprint_pass"] and not geometry["ape_5cm_margin_pass"]
    assert geometry["video_sha_pass"] and geometry["video_frames_pass"]
    assert geometry["depth_packet_sha_pass"] and geometry["reacquire_packet_sha_pass"]
    gap = geometry["person_exact_nearest"]["gap_m"]
    warm_gap = warm["geometry_audit"]["person_exact_nearest"]["gap_m"]
    ideal_gap = ideal["independent_geometry_audits"][0]["person_exact_nearest"]["gap_m"]
    assert gap < .05 and warm_gap > .05 and ideal_gap > .05
    movie = Path(episode["video"]["path"])
    assert sha(movie) == episode["video"]["sha256"]
    cap = cv2.VideoCapture(str(movie))
    assert cap.isOpened()
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (528, 1040, 3)
        frames += 1
    cap.release()
    assert frames == episode["video"]["frames"]
    audit = {"schema": "robot-learning-lab.shared-owner-sector-fault-audit.v1",
             "passed": True, "report_sha256": sha(OUT / "report.json"),
             "raw_depth_unique_hashes": len({row["depth_sha256"] for row in packets}),
             "packets_with_live_to_delivered_change": sum(
                 row["live_to_delivered_changed_beams"] > 0 for row in packets),
             "delivered_owner_sector_support": packets[0]["delivered_owner_sector_return_count"],
             "cancelled_stale_image": False,
             "shared_fault_person_gap_m": gap,
             "warm_live_depth_person_gap_m": warm_gap,
             "independent_upper_range_person_gap_m": ideal_gap,
             "decoded_video_frames": frames}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("PASS shared-sector audit", gap, warm_gap, ideal_gap, frames)


if __name__ == "__main__":
    main()
