"""Show the same clear-fast episode with and without a prior room map."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import kitchen_trial as trial
import near_clutter_trial as clutter
import online_corridor_trial as online


def main() -> None:
    case = "fast_clear_held"
    stem = f"{case}-mapped_low_guard-world0.mp4"
    sources = [clutter.OUT / "mapped_low_guard" / "videos" / stem,
               online.OUT / "videos" / stem]
    caps = [cv2.VideoCapture(str(path)) for path in sources]
    if not all(cap.isOpened() for cap in caps):
        raise RuntimeError("Missing source video")
    frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps)
    if frames != 100:
        raise RuntimeError(f"Expected 100 synchronized frames; found {frames}")
    out = online.OUT / "fast-clear-prior-map-vs-online-world0.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             12.5, (1920, 582))
    if not writer.isOpened():
        raise RuntimeError("Cannot create paired MP4")
    try:
        for index in range(frames):
            read = [cap.read() for cap in caps]
            if not all(ok for ok, _ in read):
                raise RuntimeError(f"Missing frame {index}")
            # The inherited single-run videos have a stale "layout1" caption
            # even though both scene hashes identify the V20 layout3 + stool.
            panel = np.concatenate([frame[:480] for _, frame in read], axis=1)
            for x, title in ((0, "V20: INITIAL-VIEW PRIOR MAP"),
                             (960, "V23: ONLINE CLUSTERS ONLY")):
                cv2.rectangle(panel, (x, 0), (x + 959, 43), (29, 32, 37), -1)
                cv2.putText(panel, title, (x + 16, 29),
                            cv2.FONT_HERSHEY_SIMPLEX, .72,
                            (230, 236, 238), 2, cv2.LINE_AA)
            cv2.line(panel, (959, 0), (959, 479), (240, 240, 240), 2)
            footer = np.full((102, 1920, 3), (29, 32, 37), dtype=np.uint8)
            cv2.putText(footer,
                        "RoboCasa layout 3 + near stool | same scripted fast crossing | left continue 3.00 s; right stop 2.56 s",
                        (18, 27), cv2.FONT_HERSHEY_SIMPLEX, .67,
                        (227, 233, 238), 1, cv2.LINE_AA)
            cv2.putText(footer,
                        f"t={index/12.5:.2f} s | V23 later treats a newly visible fixed surface as risk; no map prior",
                        (18, 55), cv2.FONT_HERSHEY_SIMPLEX, .61,
                        (207, 214, 221), 1, cv2.LINE_AA)
            cv2.putText(footer,
                        "Kinematic replay; ideal pose/rays; capsule person; rejected historical Direct Frame.",
                        (18, 84), cv2.FONT_HERSHEY_SIMPLEX, .56,
                        (167, 184, 196), 1, cv2.LINE_AA)
            writer.write(np.concatenate([panel, footer], axis=0))
    finally:
        writer.release()
        for cap in caps:
            cap.release()
    manifest = {"schema": "robot-learning-lab.ape-online-guard-comparison.v1",
                "case": case, "world": 0, "frames": frames,
                "path": str(out), "sha256": trial.sha(out),
                "source_sha256": [trial.sha(path) for path in sources],
                "v20_report_sha256": trial.sha(clutter.OUT / "report.json"),
                "v23_report_sha256": trial.sha(online.OUT / "report.json"),
                "source_code_sha256": trial.sha(Path(__file__))}
    (online.OUT / "comparison-video.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"path": str(out), "frames": frames}))


if __name__ == "__main__":
    main()
