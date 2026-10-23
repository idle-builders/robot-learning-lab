"""Stack matched furnished owner episodes with clear pose-source labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


MODES = ("command", "native_shaped", "native_shaped_no_scan_correction")
LABELS = ("Command odometry + range correction",
          "Native-shaped odometry + range correction",
          "Native-shaped odometry, correction bypassed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--seed", type=int, default=85)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    root = Path(__file__).resolve().parents[3]
    selected = []
    for mode in MODES:
        found = [row for row in report["rows"]
                 if row["mode"] == mode and row["seed"] == args.seed
                 and row["absent"] is False]
        if len(found) != 1:
            raise ValueError(f"Expected one present episode for {mode}, seed {args.seed}")
        selected.append(found[0])
    captures = [cv2.VideoCapture(str(root / row["episode_video"]))
                for row in selected]
    try:
        if any(not capture.isOpened() for capture in captures):
            raise RuntimeError("Could not decode an episode video")
        counts = [int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                  for capture in captures]
        widths = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                  for capture in captures]
        heights = [int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                   for capture in captures]
        fps_values = [capture.get(cv2.CAP_PROP_FPS) for capture in captures]
        if len(set(widths)) != 1 or len(set(heights)) != 1 or len(set(fps_values)) != 1:
            raise ValueError("Unmatched video dimensions or rates")
        width, height = widths[0], heights[0]
        strip_h = 48
        output = args.report.parent / f"seed{args.seed}-present-pose-source-comparison.mp4"
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_values[0], (width, 3 * (height + strip_h)))
        if not writer.isOpened():
            raise RuntimeError("Cannot encode comparison video")
        last = [None] * 3
        frames = 0
        try:
            for index in range(max(counts)):
                parts = []
                for arm, (capture, row, title) in enumerate(
                        zip(captures, selected, LABELS)):
                    if index < counts[arm]:
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError(f"Video {arm} ended before declared count")
                        last[arm] = frame
                    frame = last[arm]
                    strip = np.zeros((strip_h, width, 3), dtype=np.uint8)
                    cv2.putText(strip,
                                f"{title} | final pose error "
                                f"{row['pose_error_m']*100:.1f} cm | "
                                f"{row['outcome']}",
                                (15, 31), cv2.FONT_HERSHEY_SIMPLEX, .73,
                                (245, 245, 245), 2, cv2.LINE_AA)
                    parts.append(np.vstack([strip, frame]))
                writer.write(np.vstack(parts))
                frames += 1
        finally:
            writer.release()
        print(json.dumps({"video": str(output), "frames": frames,
                          "source_frame_counts": counts}, indent=2))
    finally:
        for capture in captures:
            capture.release()


if __name__ == "__main__":
    main()
