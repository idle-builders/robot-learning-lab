"""World-0 native early/late stop video: shorter hold is not always safer."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import native_early_hold_screen as screen
from kitchen_trial import sha


CASE = "late_left_pause"
WORLD = 0


def main() -> None:
    paths = [screen.OUT / "videos" / f"{mode}-{CASE}-world{WORLD}.mp4"
             for mode in screen.MODES]
    caps = [cv2.VideoCapture(str(path)) for path in paths]
    if not all(cap.isOpened() for cap in caps):
        raise RuntimeError("Missing V29 source video")
    report = json.loads((screen.OUT / "report.json").read_text())
    rows = {(r["mode"], r["world"], r["case"]): r for r in report["results"]}
    early = rows["early_2p5s", WORLD, CASE]
    late = rows["late_3p0s", WORLD, CASE]
    frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps)
    if frames != 100:
        raise RuntimeError(f"Expected 100 synchronized frames, found {frames}")
    out = screen.OUT / "world0-left-early-vs-late-native-hold.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             12.5, (1920, 582))
    if not writer.isOpened():
        raise RuntimeError("MP4 writer failed")
    try:
        for index in range(frames):
            read = [cap.read() for cap in caps]
            if not all(ok for ok, _ in read):
                raise RuntimeError(f"Missing frame {index}")
            panel = np.concatenate([frame[:480] for _, frame in read], axis=1)
            for x, title in ((0, "ZERO TWIST AT 2.5 S"),
                             (960, "ZERO TWIST AT 3.0 S")):
                cv2.rectangle(panel, (x, 0), (x+959, 43), (29, 32, 37), -1)
                cv2.putText(panel, title, (x+16, 29),
                            cv2.FONT_HERSHEY_SIMPLEX, .75,
                            (230, 236, 238), 2, cv2.LINE_AA)
            cv2.line(panel, (959, 0), (959, 479), (240, 240, 240), 2)
            footer = np.full((102, 1920, 3), (29, 32, 37), dtype=np.uint8)
            cv2.putText(footer,
                        f"RoboCasa layout 3 + box stool | late left entry | t={.02+index*.08:.2f} s",
                        (18, 27), cv2.FONT_HERSHEY_SIMPLEX, .67,
                        (227, 233, 238), 1, cv2.LINE_AA)
            cv2.putText(footer,
                        f"Early min body-person gap {early['min_body_person_gap_m']:+.3f} m; "
                        f"late {late['min_body_person_gap_m']:+.3f} m",
                        (18, 55), cv2.FONT_HERSHEY_SIMPLEX, .63,
                        (227, 233, 238), 1, cv2.LINE_AA)
            cv2.putText(footer,
                        "Same nominal policy seed; world-0 pre-command trace nearly identical. Historical kinematic body.",
                        (18, 84), cv2.FONT_HERSHEY_SIMPLEX, .56,
                        (167, 184, 196), 1, cv2.LINE_AA)
            writer.write(np.concatenate([panel, footer], axis=0))
    finally:
        writer.release()
        for cap in caps:
            cap.release()
    manifest = {"schema": "robot-learning-lab.ape-v29-native-hold-comparison.v1",
                "case": CASE, "world": WORLD, "frames": frames,
                "path": str(out), "sha256": sha(out),
                "source_video_sha256": [sha(path) for path in paths],
                "report_sha256": sha(screen.OUT / "report.json"),
                "audit_sha256": sha(screen.OUT / "audit.json"),
                "source_sha256": sha(Path(__file__))}
    (screen.OUT / "comparison-video.json").write_text(
        json.dumps(manifest, indent=2)+"\n")
    print(out)


if __name__ == "__main__":
    main()
