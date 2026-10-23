"""Recheck complete camera-gain owner episodes and retained evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))
import audit_robocasa_first_return_person_reacquire_v578 as geometry_audit

OUT = ROOT / "artifacts/ape-rgb-photometric-guard/v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def packet_sha(packet):
    return hashlib.sha256(json.dumps(packet, separators=(",", ":")).encode()).hexdigest()


def main():
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    sources = {
        "runner": ROOT / "experiments/ape_high_level/rgb_photometric_guard/run_photometric_guard.py",
        "rgb_guard": ROOT / "experiments/ape_high_level/rgb_depth_consistency/run_rgb_consistency.py",
        "owner_turn": ROOT / "experiments/ape_high_level/closed_loop/robocasa_first_return_person_reacquire_turn_v597.py",
        "joined_episode": ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_person_joined_v497.py",
        "geometry_audit": Path(geometry_audit.__file__),
    }
    source_checks = {key: sha(path) == report["source_sha256"][key]
                     for key, path in sources.items()}
    rows = []
    episodes = [item["episode"] for item in report["rows"]]
    pre_show = [tuple(json.dumps(row, sort_keys=True) for row in ep["trace"]
                      if row["stage"] in ("choose_pre_show_standoff",
                                          "pre_show_backtrack",
                                          "pre_show_motion_assessment"))
                for ep in episodes]
    for item in report["rows"]:
        name, ep = item["name"], item["episode"]
        sensor = ep["rgb_depth_consistency_sensor"]
        packet_path = ROOT / sensor["packet_trace"]
        packets = json.loads(packet_path.read_text())
        mid_ref = ep["raw_and_delivered_mid_range_packet_trace"]
        mid_path = ROOT / mid_ref["path"]
        mid_packets = json.loads(mid_path.read_text())
        geometry = geometry_audit.audit_episode(ep)
        geometry.pop("geometry_samples", None)
        sample_passes = []
        for sample in sensor["rgb_frame_samples"]:
            path = ROOT / sample["path"]
            frame = np.asarray(Image.open(path).convert("RGB"))
            index = sample["index"] if sample["index"] >= 0 else len(packets) - 1
            key = ("raw_rgb_sha256" if sample["kind"] == "raw" else
                   "actor_delivered_rgb_sha256")
            sample_passes.append(
                sha(path) == sample["sha256"] and
                hashlib.sha256(frame.tobytes()).hexdigest() == packets[index][key])
        movie = Path(ep["video"]["path"])
        cap = cv2.VideoCapture(str(movie))
        frames = 0
        while True:
            okay, frame = cap.read()
            if not okay:
                break
            if frame.shape != (528, 1040, 3):
                raise AssertionError(f"Bad video frame: {movie}")
            frames += 1
        cap.release()
        cancels = [r for r in ep["trace"] if r["stage"] ==
                   "owner_reacquire_image_cancelled"]
        rows.append({
            "name": name,
            "packet_hash_pass": sha(packet_path) == sensor["packet_trace_sha256"],
            "depth_packet_content_hash_pass": all(
                packet_sha(row["packet_72_world_bearings_m"]) == row[
                    "raw_live_depth_packet_sha256"] and
                packet_sha(row["actor_delivered_depth_packet_72_world_bearings_m"]) ==
                    row["actor_delivered_depth_packet_sha256"]
                for row in packets),
            "mid_range_packet_hash_pass": sha(mid_path) == mid_ref["sha256"] and
                len(mid_packets) == mid_ref["packet_count"] == len(packets) and
                all(packet_sha(row["raw_mid_range_packet"]) == row[
                        "raw_mid_range_packet_sha256"] and
                    packet_sha(row["actor_delivered_mid_range_packet"]) == row[
                        "actor_delivered_mid_range_packet_sha256"]
                    for row in mid_packets),
            "rgb_sample_hash_pass": all(sample_passes),
            "gain_schedule_pass": all(
                row["injected_digital_rgb_gain"] == (
                    1.25 if row["index"] in (2, 3) else 1.)
                for row in packets),
            "geometry_matches_saved": abs(
                geometry["person_exact_nearest"]["gap_m"] -
                item["geometry_audit"]["person_exact_nearest"]["gap_m"]) < 1e-8,
            "video_pass": sha(movie) == ep["video"]["sha256"] and
                frames == ep["video"]["frames"] and
                geometry["video_sha_pass"] and geometry["video_frames_pass"],
            "decoded_frames": frames,
            "outcome": ep["outcome"],
            "alarm_packet": sensor["first_rgb_consistency_alarm_packet"],
            "image_cancels": len(cancels),
            "person_gap_m": geometry["person_exact_nearest"]["gap_m"],
            "pulse_raw_change_fraction": packets[2][
                "uncompensated_changed_fraction"],
            "pulse_compensated_change_fraction": packets[2][
                "compensated_changed_fraction"],
            "raw_rgb_unique": len({r["raw_rgb_sha256"] for r in packets}),
            "delivered_rgb_unique": len({r["actor_delivered_rgb_sha256"]
                                         for r in packets}),
            "max_packet_start_gap_s": sensor["maximum_packet_start_gap_wall_s"],
        })
    by_name = {row["name"]: row for row in rows}
    raw = by_name["stationary_raw_gain"]
    corrected = by_name["stationary_compensated_gain"]
    moving = by_name["moving_compensated_gain"]
    frozen = by_name["moving_compensated_frozen_rgb"]
    behavior_checks = {
        "paired_stationary_pre_show": pre_show[0] == pre_show[1],
        "paired_moving_pre_show": pre_show[2] == pre_show[3],
        "same_scene": len({ep["scene_fingerprint"] for ep in episodes}) == 1,
        "raw_gain_false_interrupt": raw["pulse_raw_change_fraction"] > .005 and
            raw["alarm_packet"] is not None and raw["image_cancels"] > 0 and
            raw["outcome"] != "owner_confirmed_metric_find",
        "compensation_preserves_stationary_find":
            corrected["pulse_raw_change_fraction"] > .005 and
            corrected["pulse_compensated_change_fraction"] < .005 and
            corrected["alarm_packet"] is None and
            corrected["outcome"] == "owner_confirmed_metric_find",
        "compensation_detects_moving_owner": moving["alarm_packet"] is not None and
            moving["image_cancels"] > 0 and moving["person_gap_m"] >= .05,
        "shared_rgb_fault_still_fails": frozen["alarm_packet"] is None and
            frozen["image_cancels"] == 0 and frozen["person_gap_m"] < .05 and
            frozen["raw_rgb_unique"] > 10 and frozen["delivered_rgb_unique"] == 1,
    }
    evidence_pass = all(source_checks.values()) and all(behavior_checks.values()) and all(
        row[key] for row in rows for key in (
            "packet_hash_pass", "depth_packet_content_hash_pass",
            "mid_range_packet_hash_pass", "rgb_sample_hash_pass",
            "gain_schedule_pass", "geometry_matches_saved", "video_pass"))
    audit = {"schema": "robot-learning-lab.rgb-photometric-guard-audit.v1",
             "report_sha256": sha(report_path), "source_checks": source_checks,
             "behavior_checks": behavior_checks, "rows": rows,
             "evidence_pass": evidence_pass}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"evidence_pass": evidence_pass,
                      "behavior_checks": behavior_checks,
                      "decoded_frames": sum(row["decoded_frames"] for row in rows)},
                     indent=2))
    if not evidence_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
