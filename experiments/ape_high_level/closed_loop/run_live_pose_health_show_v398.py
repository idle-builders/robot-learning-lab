"""Live V393 owner show with a time-varying head-pose error and health gate.

The fault begins after the actor's frame-8 moving-item bootstrap. In the
flagged arm, a supplied sensor-health status prevents candidate updates
while the pose transform is unreliable. This tests the gate wiring, not
the ability of a physical IMU/encoder to detect such an error.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from audit_cued_retry_show_v395 import REPLICA, expected_held
import run_live_cued_retry_show_v393 as cued
import run_live_multi_candidate_show_v384 as live
from run import ROOT, Room, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v398"
FAULT_FIRST_INDEX = 9
FAULT_AMPLITUDE_DEG = 5.
SHOW_START_S = 16.2


def empty_unhealthy_choice() -> dict:
    return {
        "proposals": 0,
        "accepted_box": None,
        "accepted_score": None,
        "accepted_hue_intersection": None,
        "accepted_world_xyz_m": None,
        "accepted_correct_evaluator_only": False,
        "accepted_location_error_evaluator_only_m": None,
        "actor_all_candidates": [],
        "actor_candidate_count": 0,
        "actor_moving_track_count": 0,
        "actor_selected_track_id": None,
        "actor_selected_motion_span_m": None,
        "actor_pose_health": "unreliable_encoder_time_alignment",
    }


def run_arm(arm: str) -> dict:
    original_pose = Room.camera_pose_estimate
    original_infer = live.multi_candidate_infer
    original_out = cued.OUT

    def faulty_pose(room: Room, camera: str):
        origin, rotation = original_pose(room, camera)
        if camera != "lower_rgb":
            return origin, rotation
        index = int(round((room.time_s - SHOW_START_S) / .1))
        if index < FAULT_FIRST_INDEX or index > 19:
            return origin, rotation
        error = math.radians(FAULT_AMPLITUDE_DEG) * math.sin(
            2. * math.pi * index / 4.)
        c, s = math.cos(error), math.sin(error)
        error_rotation = np.asarray(((c, -s, 0.),
                                     (s, c, 0.),
                                     (0., 0., 1.)))
        return origin, error_rotation @ rotation

    def health_gated_infer(vision, exemplar, owner_hist, rgb, depth,
                           camera_pose, truth, true_target):
        if live._sample_index >= FAULT_FIRST_INDEX:
            return empty_unhealthy_choice()
        choice = original_infer(vision, exemplar, owner_hist, rgb, depth,
                                camera_pose, truth, true_target)
        choice["actor_pose_health"] = "healthy"
        return choice

    try:
        Room.camera_pose_estimate = faulty_pose
        live.multi_candidate_infer = (health_gated_infer
                                      if arm == "flagged" else original_infer)
        cued.OUT = OUT / arm
        report, rows = cued.run_mode("compliant")
    finally:
        Room.camera_pose_estimate = original_pose
        live.multi_candidate_infer = original_infer
        cued.OUT = original_out
    scored = []
    by_seed = {row["seed"]: row for row in report["owner_show_rows"]}
    for episode in report["rows"]:
        seed = int(episode["case"][0])
        response = by_seed[seed]["response"]
        samples = response["actor_samples"]
        start_s = float(samples[0]["capture_time_s"])
        held_count = replica_count = 0
        final_class = None
        for sample in samples:
            path = ROOT / sample["actor_packet_path"]
            assert sha(path) == sample["actor_packet_sha256"]
            point = sample["actor_choice"]["accepted_world_xyz_m"]
            if point is None:
                continue
            elapsed = float(sample["capture_time_s"]) - start_s
            position = np.asarray(point)
            held_error = float(np.linalg.norm(
                position - expected_held("compliant", elapsed)))
            replica_error = float(np.linalg.norm(position - REPLICA))
            final_class = ("held_item" if held_error < replica_error else
                           "raised_replica")
            held_count += final_class == "held_item"
            replica_count += final_class == "raised_replica"
        trace_path = ROOT / episode["trace_path"]
        video_path = ROOT / episode["video_path"]
        assert sha(trace_path) == episode["trace_sha256"]
        assert sha(video_path) == episode["video_sha256"]
        video = cv2.VideoCapture(str(video_path))
        frames = 0
        while True:
            ok, _ = video.read()
            if not ok:
                break
            frames += 1
        video.release()
        assert frames == 61, (video_path, frames)
        scored.append({
            "arm": arm, "seed": seed,
            "task_status": response["status"],
            "actor_track_fresh_at_answer": response[
                "actor_track_fresh_at_answer"],
            "evaluator_accepted_held_frames": held_count,
            "evaluator_accepted_replica_frames": replica_count,
            "evaluator_final_accepted_class": final_class,
            "evaluator_min_person_gap_m": response[
                "evaluator_min_person_gap_m"],
            "evaluator_min_scene_gap_m": response[
                "evaluator_min_scene_gap_m"],
            "video_decoded_frames": frames,
        })
    print(arm, [(row["seed"], row["task_status"],
                 row["evaluator_accepted_held_frames"],
                 row["evaluator_accepted_replica_frames"])
                for row in scored], flush=True)
    report_path = OUT / arm / "compliant" / "blocked-owner-help-audit.json"
    return {"arm": arm, "report_path": str(report_path.relative_to(ROOT)),
            "report_sha256": sha(report_path), "rows": scored}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_arm(arm) for arm in ("naive", "flagged")]
    report = {
        "scope": "V393 compliant-owner live guarded room, seeds170/172, with a constructed camera-to-world yaw error of +/-5deg at10Hz from show index9 onward. The naive arm uses corrupted camera pose for candidate metric association and final robust point. The flagged arm receives a supplied encoder-time-health-unreliable status and skips candidate update/claim during the fault. RGB/depth capture and scripted owner motion are identical. This is a fault-injection and gate-wiring test; the flag is supplied by the harness, not inferred from physical IMU/joint telemetry, and the Direct Frame is a rejected mechanical reference.",
        "source_sha256": sha(SOURCE),
        "v393_source_sha256": sha(Path(cued.__file__)),
        "fault_first_index": FAULT_FIRST_INDEX,
        "fault_amplitude_deg": FAULT_AMPLITUDE_DEG,
        "rows": rows,
    }
    (OUT / "live-pose-health-show-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
