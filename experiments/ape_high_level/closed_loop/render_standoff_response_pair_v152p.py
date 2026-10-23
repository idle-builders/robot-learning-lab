"""Pair a scripted half-second prop halt with an ignored front-left alert."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from run import ROOT, sha
import run_standoff_front_response_v150p as v150p


SOURCE = Path(__file__)
OUT = v150p.OUT
VIDEO = OUT / "far-standoff-half-second-stop-vs-ignored.mp4"


def nearest_gap(row: dict, time_s: float) -> float:
    path = row["owner_turn"]["path_10hz"]
    sample = min(path, key=lambda item: abs(item["time_s"]-time_s))
    return float(sample["evaluator_gap_m"])


def annotate(frame: np.ndarray, title: str,
             time_s: float, gap_m: float) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 49), (0, 0, 0), -1)
    cv2.putText(result, title, (12, 20), cv2.FONT_HERSHEY_SIMPLEX,
                .57, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(result,
                f"t{time_s:.1f}s | front-left alert 17.3s | evaluator gap "
                f"{100*gap_m:.1f}cm",
                (12, 42), cv2.FONT_HERSHEY_SIMPLEX,
                .48, (215, 225, 255), 1, cv2.LINE_AA)
    return result


def main() -> None:
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    sources = {}
    for label in ("stop-0.5s", "ignored"):
        row = next(row for row in report["rows"]
                   if row["seed"] == 170 and row["station_x_m"] == -.30
                   and row["response"] == label)
        path = ROOT / row["owner_turn"]["video_path"]
        if sha(path) != row["owner_turn"]["video_sha256"]:
            raise AssertionError("Source episode video changed")
        sources[label] = (row, path)
    stop_capture = cv2.VideoCapture(str(sources["stop-0.5s"][1]))
    ignored_capture = cv2.VideoCapture(str(sources["ignored"][1]))
    if not stop_capture.isOpened() or not ignored_capture.isOpened():
        raise RuntimeError("Cannot open source videos")
    frame_count = 0
    with imageio.get_writer(VIDEO, fps=5, codec="libx264", quality=7) as writer:
        while True:
            stop_ok, stop_frame = stop_capture.read()
            ignored_ok, ignored_frame = ignored_capture.read()
            if not stop_ok and not ignored_ok:
                break
            if stop_ok != ignored_ok:
                raise AssertionError("Source videos differ in frame count")
            if stop_frame.shape != ignored_frame.shape:
                raise AssertionError("Source videos differ in shape")
            time_s = 16.2 + .2*frame_count
            left = annotate(stop_frame,
                            "Scripted prop halt 0.5s after alert",
                            time_s, nearest_gap(sources["stop-0.5s"][0], time_s))
            right = annotate(ignored_frame,
                             "Alert ignored: no physical response",
                             time_s, nearest_gap(sources["ignored"][0], time_s))
            writer.append_data(cv2.cvtColor(np.concatenate((left, right), axis=1),
                                            cv2.COLOR_BGR2RGB))
            frame_count += 1
    stop_capture.release()
    ignored_capture.release()
    if frame_count < 18:
        raise AssertionError("Paired video unexpectedly short")
    video_report = {
        "schema": "robot-learning-lab.ape-standoff-response-pair-video.v1",
        "source_sha256": sha(SOURCE),
        "v150_report_sha256": sha(report_path),
        "source_videos": {
            label: {"path": str(path.relative_to(ROOT)),
                    "sha256": sha(path)}
            for label, (_, path) in sources.items()
        },
        "video_path": str(VIDEO.relative_to(ROOT)),
        "video_sha256": sha(VIDEO),
        "frames": frame_count,
        "scope": "Paired seed170 far-standoff video from the same historical "
                 "owner turn and front-left range cue. The left prop halts "
                 "0.5s after the17.3s actor alert, while the right continues. "
                 "Gap captions are evaluator-only. The ape does not execute "
                 "a motion response, and the owner/prop behavior is scripted.",
    }
    (OUT / "pair-video-report.json").write_text(json.dumps(
        video_report, indent=2) + "\n")
    print(VIDEO, frame_count)


if __name__ == "__main__":
    main()
