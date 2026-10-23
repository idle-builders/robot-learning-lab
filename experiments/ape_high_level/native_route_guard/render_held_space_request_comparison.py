"""Synchronize the V32 front/flank and compliant/ignored episode videos."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import held_space_request_loop as loop
import kitchen_trial as trial


def main() -> None:
    names = [("front_body_15hz", "complies"),
             ("front_body_15hz", "ignores"),
             ("split_body_10hz", "complies"),
             ("split_body_10hz", "ignores")]
    streams = []
    source = []
    for mode, response in names:
        path = loop.OUT / "videos" / \
            f"{mode}-late_right_pause-{response}-world0.mp4"
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames in {path}")
        streams.append(frames)
        source.append({"mode": mode, "response": response,
                       "path": str(path), "sha256": trial.sha(path),
                       "frames": len(frames)})
    frame_size = streams[0][0].shape[:2]
    if any(stream[0].shape[:2] != frame_size for stream in streams):
        raise RuntimeError("Mismatched source frame dimensions")
    h, w = frame_size
    output = loop.OUT / "videos" / "world0-right-approach-sensor-response-comparison.mp4"
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                             10., (2*w, 2*h))
    if not writer.isOpened():
        raise RuntimeError("comparison video writer failed")
    count = round((loop.END_S-loop.START_S)*10)+1
    for tick in range(count):
        elapsed = tick/10
        selected = []
        for (mode, _), frames in zip(names, streams):
            fps = loop.sensor.MODES[mode]["rate_hz"]/2
            index = min(len(frames)-1, round(elapsed*fps))
            selected.append(frames[index])
        writer.write(np.concatenate((np.concatenate(selected[:2], axis=1),
                                     np.concatenate(selected[2:], axis=1)), axis=0))
    writer.release()
    report = {"schema": "robot-learning-lab.ape-held-space-comparison-video.v1",
              "video": {"path": str(output), "sha256": trial.sha(output),
                        "frames": count, "fps": 10},
              "layout": "front/complies, front/ignores; split/complies, split/ignores",
              "synchronized_from_s": loop.START_S,
              "sources": source}
    (loop.OUT / "comparison-video.json").write_text(
        json.dumps(report, indent=2)+"\n")
    print(output)


if __name__ == "__main__":
    main()
