"""Synchronized V20 no-sensor and single-front-guard right crossing."""

from __future__ import annotations

import json

import cv2
import numpy as np

import kitchen_trial as trial
import near_clutter_trial as near


CASE = "slow_right_held"
WORLD = 0


def main() -> None:
    report = json.loads((near.OUT / "report.json").read_text())
    rows = {(row["mode"], row["world"], row["case"]): row
            for row in report["results"]}
    modes = ("none", "mapped_low_guard")
    sources = [near.OUT / mode / "videos" / f"{CASE}-{mode}-world{WORLD}.mp4"
               for mode in modes]
    traces = [np.load(near.OUT / mode / "episodes" /
                      f"{CASE}-{mode}-world{WORLD}.npz", allow_pickle=False)
              for mode in modes]
    movies = [cv2.VideoCapture(str(path)) for path in sources]
    if not all(movie.isOpened() for movie in movies):
        raise RuntimeError("Missing V20 source video")
    count = min(int(movie.get(cv2.CAP_PROP_FRAME_COUNT)) for movie in movies)
    target = near.OUT / "slow-right-no-sensor-vs-single-front-world0.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"),
                             12.5, (1280, 544))
    if not writer.isOpened():
        raise RuntimeError("Cannot create paired MP4")
    try:
        for index in range(count):
            reads = [movie.read() for movie in movies]
            if not all(ok for ok, _ in reads):
                raise RuntimeError(f"Missing frame {index}")
            panel = np.concatenate([frame[:480, :640] for _, frame in reads], axis=1)
            for x0, label in ((0, "NO RANGE GUARD"),
                              (640, "SINGLE FRONT LOW GUARD")):
                cv2.rectangle(panel, (x0, 0), (x0+639, 42), (28, 30, 34), -1)
                cv2.putText(panel, label, (x0+15, 29), cv2.FONT_HERSHEY_SIMPLEX,
                            .62, (230, 236, 238), 2)
            cv2.line(panel, (639, 0), (639, 479), (250, 250, 250), 2)
            foot = np.full((64, 1280, 3), (29, 31, 35), dtype=np.uint8)
            sample = min(index*4, len(traces[0]["time_s"])-1)
            t = float(traces[0]["time_s"][sample])
            for x0, mode, trace in ((16, modes[0], traces[0]),
                                    (657, modes[1], traces[1])):
                gap = float(trace["gap_evaluator_m"][sample])
                stop = rows[mode, WORLD, CASE]["stop_command_s"]
                cv2.putText(foot, f"t={t:.2f}s  gap={gap:+.3f}m; stop={stop:.2f}s",
                            (x0, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            .62, (226, 233, 237), 1, cv2.LINE_AA)
            cv2.putText(foot, "RoboCasa layout3 + near stool; scripted capsule; ideal rays/native trace splice",
                        (16, 52), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (160, 178, 191), 1, cv2.LINE_AA)
            writer.write(np.concatenate([panel, foot], axis=0))
    finally:
        writer.release()
        for movie in movies:
            movie.release()
        for trace in traces:
            trace.close()
    manifest = {"schema": "robot-learning-lab.ape-robocasa-near-clutter-comparison.v1",
                "report_sha256": trial.sha(near.OUT / "report.json"),
                "case": CASE, "world": WORLD, "path": str(target),
                "sha256": trial.sha(target), "frames": count,
                "source_hashes": [trial.sha(path) for path in sources]}
    (near.OUT / "comparison-video.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(target, count)


if __name__ == "__main__":
    main()
