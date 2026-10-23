"""Show pre-alert and post-alert front-left range blindness side by side."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from run import ROOT, sha
import run_standoff_range_blind_v155p as v155p


SOURCE = Path(__file__)
VIDEO = v155p.OUT / "range-loss-before-vs-after-warning.mp4"


def decorate(frame: np.ndarray, title: str, outcome: str) -> np.ndarray:
    image = frame.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 62), (0, 0, 0), -1)
    cv2.putText(image, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, outcome, (10, 49), cv2.FONT_HERSHEY_SIMPLEX,
                .50, (210, 225, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    report_path = v155p.OUT / "report.json"
    report = json.loads(report_path.read_text())
    rows = {row["fault_start_s"]: row for row in report["rows"]
            if row["seed"] == 170}
    if set(rows) != {17.2, 17.4}:
        raise AssertionError("Paired fault cases missing")
    captures = []
    for start in (17.2, 17.4):
        path = ROOT / rows[start]["video_path"]
        if sha(path) != rows[start]["video_sha256"]:
            raise AssertionError("Source video changed")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        captures.append(capture)
    count = 0
    try:
        with imageio.get_writer(VIDEO, fps=5, codec="libx264",
                                quality=7) as writer:
            while True:
                left_ok, left_frame = captures[0].read()
                right_ok, right_frame = captures[1].read()
                if not left_ok and not right_ok:
                    break
                if left_ok != right_ok or left_frame.shape != right_frame.shape:
                    raise AssertionError("Source videos not synchronized")
                left = decorate(left_frame,
                                "Range blind at 17.2s: before warning",
                                "No front cue; prop continues; modeled overlap")
                right = decorate(right_frame,
                                 "Range blind at 17.4s: after warning",
                                 "17.3s cue; scripted stop; no fresh image")
                paired = np.concatenate((left, right), axis=1)
                writer.append_data(cv2.cvtColor(paired, cv2.COLOR_BGR2RGB))
                count += 1
    finally:
        for capture in captures:
            capture.release()
    if count < 25:
        raise AssertionError("Paired video unexpectedly short")
    metadata = {
        "schema": "robot-learning-lab.ape-range-blind-pair-video.v1",
        "source_sha256": sha(SOURCE),
        "v155_report_sha256": sha(report_path),
        "source_videos": {str(start): {"path": rows[start]["video_path"],
                                       "sha256": rows[start]["video_sha256"]}
                          for start in (17.2, 17.4)},
        "video_path": str(VIDEO.relative_to(ROOT)),
        "video_sha256": sha(VIDEO),
        "frames": count,
        "scope": "Paired complete seed170 turns at the same held historical pose. "
                 "Both start a front-left packet-field outage. Before the "
                 "warning no cooperative prop stop is triggered; after the "
                 "warning a scripted 0.2s stop preserves modeled clearance "
                 "but loss of range prevents visual recovery. Gap captions "
                 "are evaluator-only and this is not a physical sensor fault.",
    }
    (v155p.OUT / "pair-video-report.json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(VIDEO, count)


if __name__ == "__main__":
    main()
