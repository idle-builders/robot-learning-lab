"""Audit the depth-versus-range guard comparison and complete episodes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
COLD = ROOT / "artifacts/ape-depth-guard-choice/v1"
WARM = ROOT / "artifacts/ape-depth-guard-choice/warm-v2"
BASELINES = [
    ROOT / "artifacts/ape-robocasa-reacquire-frozen-sector-v596/report.json",
    ROOT / "artifacts/ape-robocasa-reacquire-frozen-sector-dual-v597/report.json",
    ROOT / "artifacts/ape-robocasa-reacquire-dual-healthy-v598/report.json",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_episode(episode: dict, geometry: dict) -> dict:
    sensor = episode["depth_guard_sensor"]
    path = ROOT / sensor["packet_trace"]
    assert sha(path) == sensor["packet_trace_sha256"]
    packets = json.loads(path.read_text())
    assert len(packets) == sensor["packet_count"] and len(packets) > 10
    assert all(len(row["packet_72_world_bearings_m"]) == 72 for row in packets)
    assert all(row["valid_depth_sample_count"] > 0 for row in packets)
    assert all(row["packet_supported_beams"] >= 20 for row in packets)
    for sample in sensor["sample_depth_frames"]:
        frame = ROOT / sample["path"]
        assert sha(frame) == sample["sha256"]
        depth = np.load(frame, allow_pickle=False)
        assert depth.shape == (480, 640)
        log = packets[sample["index"]] if sample["index"] >= 0 else packets[-1]
        assert hashlib.sha256(depth.tobytes()).hexdigest() == log["depth_sha256"]
    gaps = np.diff([row["wall_timestamp_s"] for row in packets])
    assert abs(float(gaps.max()) - sensor["maximum_packet_start_gap_wall_s"]) < 1e-4
    assert abs(float(np.median([row["render_and_project_wall_s"]
                                for row in packets])) -
               sensor["median_render_and_project_wall_s"]) < 1e-4
    video = Path(episode["video"]["path"])
    assert sha(video) == episode["video"]["sha256"]
    cap = cv2.VideoCapture(str(video))
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
    assert geometry["scene_fingerprint_pass"]
    assert geometry["ape_5cm_margin_pass"]
    assert geometry["person_room_5cm_pass"]
    assert geometry["can_room_nonoverlap_pass"]
    assert geometry["video_sha_pass"] and geometry["video_frames_pass"]
    assert geometry["depth_packet_sha_pass"] and geometry["reacquire_packet_sha_pass"]
    return {"outcome": episode["outcome"], "video_frames": frames,
            "person_gap_m": geometry["person_exact_nearest"]["gap_m"],
            "guard_packets": len(packets),
            "maximum_packet_start_gap_wall_s": sensor["maximum_packet_start_gap_wall_s"],
            "median_depth_render_and_projection_wall_s":
                sensor["median_render_and_project_wall_s"]}


def main() -> None:
    cold = json.loads((COLD / "report.json").read_text())
    warm = json.loads((WARM / "report.json").read_text())
    assert sha(ROOT / "experiments/ape_high_level/depth_guard_choice/run_depth_guard.py") == cold["source_sha256"]["runner"]
    assert sha(ROOT / "experiments/ape_high_level/depth_guard_choice/run_warm_depth_guard.py") == warm["source_sha256"]["wrapper"]
    assert warm["source_sha256"]["cold_runner"] == cold["source_sha256"]["runner"]
    for name, rel in {
        "owner_turn": "experiments/ape_high_level/closed_loop/robocasa_first_return_person_reacquire_turn_v597.py",
        "joined_route": "experiments/ape_high_level/closed_loop/run_robocasa_person_joined_v497.py",
        "geometry_audit": "experiments/ape_high_level/closed_loop/audit_robocasa_first_return_person_reacquire_v578.py",
    }.items():
        digest = sha(ROOT / rel)
        assert cold["source_sha256"][name] == warm["source_sha256"][name] == digest
    baseline = [json.loads(path.read_text()) for path in BASELINES]
    fingerprint = baseline[0]["episodes"][0]["scene_fingerprint"]
    assert all(e["scene_fingerprint"] == fingerprint for report in baseline
               for e in report["episodes"])
    results = {}
    for row in cold["rows"]:
        episode = row["episode"]
        assert episode["scene_fingerprint"] == fingerprint
        results[row["condition"]] = audit_episode(episode, row["geometry_audit"])
    warm_episode = warm["episode"]
    assert warm_episode["scene_fingerprint"] == fingerprint
    results["moving_frozen_mid_warm"] = audit_episode(warm_episode,
                                                       warm["geometry_audit"])
    moving = cold["rows"][0]["episode"]
    stationary = cold["rows"][1]["episode"]
    for episode in (moving, warm_episode):
        alarms = [row for row in episode["trace"]
                  if row.get("stage") == "owner_reacquire_image_cancelled"]
        assert len(alarms) == 1 and alarms[0]["alarm_reason"] == "upper_owner_sector_motion"
        assert episode["outcome"] == "owner_interaction_reset_after_reacquire_motion"
        assert episode["frozen_mid_guard_packets"] > 0
    assert stationary["outcome"] == "owner_confirmed_metric_find"
    assert not any(row.get("stage") == "owner_reacquire_image_cancelled"
                   for row in stationary["trace"])
    old_gap = baseline[0]["independent_geometry_audits"][0]["person_exact_nearest"]["gap_m"]
    ideal_gap = baseline[1]["independent_geometry_audits"][0]["person_exact_nearest"]["gap_m"]
    assert old_gap < .05 and ideal_gap > .05
    assert results["moving_frozen_mid"]["person_gap_m"] > .05
    assert results["moving_frozen_mid_warm"]["person_gap_m"] > .05
    assert (results["moving_frozen_mid_warm"]["maximum_packet_start_gap_wall_s"] <
            results["moving_frozen_mid"]["maximum_packet_start_gap_wall_s"])
    summary = {"schema": "robot-learning-lab.depth-guard-choice-audit.v1",
               "passed": True,
               "cold_report_sha256": sha(COLD / "report.json"),
               "warm_report_sha256": sha(WARM / "report.json"),
               "baseline_unhandled_person_gap_m": old_gap,
               "baseline_ideal_upper_range_person_gap_m": ideal_gap,
               "results": results}
    (COLD / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PASS depth guard audit", {key: value["person_gap_m"]
                                     for key, value in results.items()})


if __name__ == "__main__":
    main()
