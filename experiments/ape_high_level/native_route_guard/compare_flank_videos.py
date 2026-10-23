"""Synchronized V14 missing-sector and V15 independent-flank outcomes."""

from __future__ import annotations

import json

import cv2
import numpy as np

import blocked_sector_trial as blocked
import flank_sensor_trial as flank
import kitchen_trial as trial


PAIRS = (("slow_cross_held", 0), ("fast_clear_held", 1))


def make(case: str, world: int, blocked_rows: dict, flank_rows: dict) -> dict:
    stem = f"{case}-mapped_low_guard-world{world}"
    sources = [blocked.OUT / "ignore_health" / "videos" / f"{stem}.mp4",
               flank.OUT / "videos" / f"{stem}.mp4"]
    traces = [np.load(blocked.OUT / "ignore_health" / "episodes" / f"{stem}.npz",
                      allow_pickle=False),
              np.load(flank.OUT / "episodes" / f"{stem}.npz", allow_pickle=False)]
    movies = [cv2.VideoCapture(str(path)) for path in sources]
    if not all(movie.isOpened() for movie in movies):
        raise RuntimeError(f"Missing source video {stem}")
    count = min(int(movie.get(cv2.CAP_PROP_FRAME_COUNT)) for movie in movies)
    target = flank.OUT / f"{case}-blocked-vs-flank-world{world}.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"),
                             12.5, (1280, 544))
    if not writer.isOpened():
        raise RuntimeError("Cannot create paired MP4")
    try:
        for index in range(count):
            reads = [movie.read() for movie in movies]
            if not all(ok for ok, _ in reads):
                raise RuntimeError(f"Missing frame {index} in {stem}")
            panel = np.concatenate([frame[:480, :640] for _, frame in reads], axis=1)
            for x0, label in ((0, "FRONT-LEFT MISSING"),
                              (640, "INDEPENDENT LEFT FAN")):
                cv2.rectangle(panel, (x0, 0), (x0+639, 42), (28, 30, 34), -1)
                cv2.putText(panel, label, (x0+15, 29), cv2.FONT_HERSHEY_SIMPLEX,
                            .64, (230, 236, 238), 2)
            cv2.line(panel, (639, 0), (639, 479), (250, 250, 250), 2)
            foot = np.full((64, 1280, 3), (29, 31, 35), dtype=np.uint8)
            sample = min(index*4, len(traces[0]["time_s"])-1)
            t = float(traces[0]["time_s"][sample])
            for x0, trace, stop in ((16, traces[0],
                                    blocked_rows[world, case]["stop_command_s"]),
                                   (657, traces[1],
                                    flank_rows[world, case]["stop_command_s"])):
                gap = float(trace["gap_evaluator_m"][sample])
                cv2.putText(foot, f"t={t:.2f}s  gap={gap:+.3f}m; stop={stop:.2f}s",
                            (x0, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            .62, (226, 233, 237), 1, cv2.LINE_AA)
            cv2.putText(foot, "Scripted capsule; ideal rays and odometry; historical kinematic stop replay",
                        (16, 52), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (160, 178, 191), 1, cv2.LINE_AA)
            writer.write(np.concatenate([panel, foot], axis=0))
    finally:
        writer.release()
        for movie in movies:
            movie.release()
        for trace in traces:
            trace.close()
    return {"case": case, "world": world, "path": str(target),
            "sha256": trial.sha(target), "frames": count,
            "source_hashes": [trial.sha(source) for source in sources]}


def main() -> None:
    blocked_report = json.loads((blocked.OUT / "report.json").read_text())
    flank_report = json.loads((flank.OUT / "report.json").read_text())
    blocked_rows = {(r["world"], r["case"]): r for r in blocked_report["results"]
                    if r["response"] == "ignore_health"}
    flank_rows = {(r["world"], r["case"]): r for r in flank_report["results"]}
    clips = [make(case, world, blocked_rows, flank_rows) for case, world in PAIRS]
    manifest = {"schema": "robot-learning-lab.ape-robocasa-flank-comparison.v1",
                "blocked_report_sha256": trial.sha(blocked.OUT / "report.json"),
                "flank_report_sha256": trial.sha(flank.OUT / "report.json"),
                "clips": clips}
    (flank.OUT / "comparison-videos.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print([(clip["case"], clip["world"], clip["frames"]) for clip in clips])


if __name__ == "__main__":
    main()
