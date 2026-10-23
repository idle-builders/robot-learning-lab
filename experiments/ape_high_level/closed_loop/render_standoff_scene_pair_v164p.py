"""Pair the near-pose fast-wide prop's prompt stop and ignored branches."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from run import ROOT, sha
import run_standoff_scene_variants_v160p as v160p


SOURCE = Path(__file__)
VIDEO = v160p.OUT / "near-fast-wide-stop-vs-ignored.mp4"


def decorate(frame: np.ndarray, title: str, result: str) -> np.ndarray:
    image = frame.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 61), (0, 0, 0), -1)
    cv2.putText(image, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .60, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, result, (10, 47), cv2.FONT_HERSHEY_SIMPLEX,
                .48, (210, 225, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    report_file = v160p.OUT / "report.json"
    report = json.loads(report_file.read_text())
    rows = {row["scripted_prop_stop_delay_s"]: row
            for row in report["rows"]
            if row["variant"] == "fast_wide_turn" and
            row["seed"] == 170 and row["station_x_m"] == -.22}
    if set(rows) != {.2, None}:
        raise AssertionError("Paired scene cases missing")
    captures = []
    for delay in (.2, None):
        path = ROOT / rows[delay]["video_path"]
        if sha(path) != rows[delay]["video_sha256"]:
            raise AssertionError("Source episode video changed")
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
                    raise AssertionError("Episode videos not synchronized")
                left = decorate(left_frame,
                                "Wider, faster prop: halt 0.2s after alert",
                                "5.38cm minimum; range remains unclear")
                right = decorate(right_frame,
                                 "Same prop: alert ignored",
                                 "4.30cm modeled body overlap")
                writer.append_data(cv2.cvtColor(
                    np.concatenate((left, right), axis=1),
                    cv2.COLOR_BGR2RGB))
                count += 1
    finally:
        for capture in captures:
            capture.release()
    if count < 25:
        raise AssertionError("Paired video incomplete")
    metadata = {
        "schema": "robot-learning-lab.ape-standoff-scene-pair-video.v1",
        "source_sha256": sha(SOURCE),
        "v160_report_sha256": sha(report_file),
        "source_videos": {
            "scripted_stop_0p2s": {"path": rows[.2]["video_path"],
                                   "sha256": rows[.2]["video_sha256"]},
            "ignored": {"path": rows[None]["video_path"],
                        "sha256": rows[None]["video_sha256"]},
        },
        "video_path": str(VIDEO.relative_to(ROOT)),
        "video_sha256": sha(VIDEO),
        "frames": count,
        "scope": "Same historical held pose, seed170, wider/faster scripted "
                 "low prop. Left pane stops the prop0.2s after the actor's "
                 "front-left warning; the actor still lacks stable range to "
                 "resume. Right pane ignores the warning and overlaps the "
                 "modeled body. Gap captions are evaluator-only, and no "
                 "selected-body action runs.",
    }
    (v160p.OUT / "pair-video-report.json").write_text(json.dumps(
        metadata, indent=2) + "\n")
    print(VIDEO, count)


if __name__ == "__main__":
    main()
