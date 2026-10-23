"""Audit the frozen apartment native-shaped pose comparison and media.

Capture-aligned pose error is scored before the current command executes. The
older dynamic-owner trace's `evaluator_pose_error_m` scores after that command
and includes a one-step sensor-latency offset, so both remain visible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/cozy_apartment_owner_screen"
PRECOMMIT = ROOT / "artifacts/ape-real-room-next-captures-20261014/native_pose_fallback_precommit.json"
OUTPUT = ROOT / "artifacts/ape-real-room-next-captures-20261014/native_pose_fallback_report.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decoded_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode {path}")
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    return count


def audit(batch_path: Path) -> dict:
    batch = json.loads(batch_path.read_text())
    entry = batch["scenes"][0]
    trace_path = ROOT / entry["trace"]
    trace = json.loads(trace_path.read_text())
    frames = trace["trace"]
    source = json.loads((ROOT / trace["source_capture"]).read_text())
    start = np.asarray(source["start_xz"], dtype=float)
    prior_actual = start.copy()
    capture_errors = []
    for frame in frames:
        actual_capture = np.asarray(frame.get(
            "evaluator_actual_xz_at_capture_m", prior_actual), dtype=float)
        actor = np.asarray(frame["actor_pose_xy_m"], dtype=float)
        capture_errors.append(float(np.linalg.norm(actor - (actual_capture - start))))
        prior_actual = np.asarray(frame["evaluator_actual_xz_m"], dtype=float)
    video = ROOT / entry["video"]
    count = decoded_frames(video)
    if count != trace["frames"] or count != len(frames):
        raise RuntimeError(f"Video/trace mismatch: {batch_path}")
    if (len(list(trace_path.parent.glob("rgb_*.png"))) != count or
            len(list(trace_path.parent.glob("depth_*.png"))) != count):
        raise RuntimeError(f"RGB/depth/trace mismatch: {batch_path}")
    native_packets = [frame["native_pose_packet_at_capture"] for frame in frames
                      if frame.get("native_pose_packet_at_capture") is not None]
    visual_rejections = [frame for frame in frames
                         if frame.get("pose_evidence") and
                         frame["pose_evidence"].get("visual") and
                         not frame["pose_evidence"]["visual"].get("accepted", False)]
    last = frames[-1]
    forward_range = last["decision_owner_actor"].get("nearest_forward_range_m")
    strict_pass = bool(trace["outcome"] == "hold_viewpoint_far" and
                       max(capture_errors) <= .08 and
                       forward_range is not None and forward_range >= .9 and
                       all(packet["accepted"] for packet in native_packets))
    return {"batch": str(batch_path.relative_to(ROOT)),
            "batch_sha256": sha(batch_path),
            "pose_mode": trace["pose_mode"],
            "native_source": trace["native_pose_source"],
            "native_source_sha256": trace["native_pose_trace_sha256"],
            "outcome": trace["outcome"],
            "stop_reason": last["planner_decision"].get("reason"),
            "frames": count, "accepted_travel_m": trace["accepted_travel_m"],
            "max_pose_error_at_capture_m": max(capture_errors),
            "final_pose_error_at_capture_m": capture_errors[-1],
            "reported_max_post_action_pose_error_m": trace["max_pose_error_m"],
            "native_packets": len(native_packets),
            "all_native_packets_healthy": all(p["accepted"] for p in native_packets),
            "minimum_native_contact_samples": min((p.get("contact_samples", 0)
                                                   for p in native_packets), default=None),
            "visual_rejected_frames": [frame["frame"] for frame in visual_rejections],
            "max_unverified_fallback_distance_m": max(
                frame.get("native_fallback_unverified_distance_m", 0.)
                for frame in frames),
            "max_unverified_fallback_frames": max(
                frame.get("native_fallback_unverified_frames", 0)
                for frame in frames),
            "final_observed_owner_depth_m": trace["final_observed_owner_depth_m"],
            "final_forward_range_m": forward_range,
            "strict_precommitted_pass": strict_pass,
            "video": entry["video"], "video_sha256": sha(video),
            "trace": entry["trace"], "trace_sha256": sha(trace_path)}


def main() -> None:
    plan = json.loads(PRECOMMIT.read_text())
    rows = []
    for source in plan["historical_native_sources"]:
        for mode in ("native_shaped", "rgbd_native_fallback"):
            matches = list(BASE.glob(
                f"owner_batch_{mode}_arbitrated*farconverse_{source}_v1.json"))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {mode}/{source} batch: {matches}")
            rows.append(audit(matches[0]))
    output = {"schema": "ape.real-capture-native-pose-fallback-audit.v1",
              "precommit": str(PRECOMMIT.relative_to(ROOT)),
              "precommit_sha256": sha(PRECOMMIT),
              "scene": plan["scene"],
              "scope": "One photo-reconstructed apartment, three historical gait error traces, two precommitted pose branches; kinematic ape, ideal yaw surrogate, no selected body or physical sensor packet.",
              "capture_error_definition": "Actor estimate at image capture minus simulator pose at that same capture, after previous accepted action but before the current command. Truth is evaluator-only.",
              "post_action_error_warning": "Legacy max_pose_error_m compares capture estimate with pose after the current command and includes one-step latency.",
              "runs": rows,
              "strict_pass_counts": {mode: sum(r["strict_precommitted_pass"]
                                               for r in rows if r["pose_mode"] == mode)
                                     for mode in ("native_shaped", "rgbd_native_fallback")}}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["strict_pass_counts"]))


if __name__ == "__main__":
    main()
