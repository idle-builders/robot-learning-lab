"""Caption the V27 late-entry video with actual packet and body-gap timing."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import held_body_intrusion_trial as late
from kitchen_trial import sha


def main() -> None:
    case = "late_right_pause"
    source = (late.OUT / "bounded_guard/videos" /
              f"{case}-mapped_low_guard-world0.mp4")
    trace = (late.OUT / "bounded_guard/episodes" /
             f"{case}-mapped_low_guard-world0.npz")
    with np.load(trace, allow_pickle=False) as saved:
        times = saved["time_s"].copy()
        gaps = saved["gap_evaluator_m"].copy()
        actions = saved["action"].copy()
        packets = [json.loads(raw) for raw in saved["packet_json"]]
    last_packet = max(p["captured_at_s"] for p in packets)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError("Missing source video")
    out = late.OUT / "late-right-held-intrusion-explained-world0.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             12.5, (960, 582))
    if not writer.isOpened():
        raise RuntimeError("MP4 writer failed")
    frames = 0
    try:
        for index in range(100):
            okay, frame = cap.read()
            if not okay:
                raise RuntimeError(f"Missing frame {index}")
            sample = index*4
            t = float(times[sample])
            gap = float(gaps[sample])
            scene = frame[:480].copy()
            cv2.rectangle(scene, (0, 0), (959, 43), (29, 32, 37), -1)
            cv2.putText(scene, "LATE PERSON ENTRY WHILE THE APE IS HOLDING",
                        (14, 29), cv2.FONT_HERSHEY_SIMPLEX, .70,
                        (230, 236, 238), 2, cv2.LINE_AA)
            footer = np.full((102, 960, 3), (29, 32, 37), dtype=np.uint8)
            color = ((83, 94, 227) if gap < 0 else
                     (83, 180, 227) if gap < .05 else
                     (227, 233, 238))
            cv2.putText(footer,
                        f"RoboCasa layout 3 + box stool | t={t:.2f} s | {actions[sample]}",
                        (14, 25), cv2.FONT_HERSHEY_SIMPLEX, .59,
                        (227, 233, 238), 1, cv2.LINE_AA)
            cv2.putText(footer,
                        f"Modeled body-person gap {gap:+.3f} m | last guard packet {last_packet:.2f} s",
                        (14, 52), cv2.FONT_HERSHEY_SIMPLEX, .60,
                        color, 1, cv2.LINE_AA)
            cv2.putText(footer,
                        "No post-deadline sensing or escape; scripted capsule and historical kinematic body",
                        (14, 79), cv2.FONT_HERSHEY_SIMPLEX, .49,
                        (167, 184, 196), 1, cv2.LINE_AA)
            writer.write(np.concatenate([scene, footer], axis=0))
            frames += 1
    finally:
        writer.release()
        cap.release()
    manifest = {"schema": "robot-learning-lab.ape-v27-explained-video.v1",
                "case": case, "world": 0, "frames": frames,
                "path": str(out), "sha256": sha(out),
                "source_video_sha256": sha(source),
                "trace_sha256": sha(trace),
                "source_sha256": sha(Path(__file__))}
    (late.OUT / "explained-video.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
