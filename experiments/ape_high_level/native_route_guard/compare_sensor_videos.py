"""Synchronized sparse/dense low-scan behavior comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import changed_speed_trial as sparse
import dense_low_trial as dense
import kitchen_trial as trial


CASES = (("fast_clear_held", 1), ("pause_path_held", 0))


def make(case: str, world: int, left_rows: dict, right_rows: dict) -> dict:
    stem = f"{case}-mapped_low_guard-world{world}"
    source_left = sparse.OUT / "videos" / f"{stem}.mp4"
    source_right = dense.OUT / "videos" / f"{stem}.mp4"
    left = cv2.VideoCapture(str(source_left))
    right = cv2.VideoCapture(str(source_right))
    if not left.isOpened() or not right.isOpened():
        raise RuntimeError(f"Missing matched videos for {stem}")
    left_trace = np.load(sparse.OUT / "episodes" / f"{stem}.npz", allow_pickle=False)
    right_trace = np.load(dense.OUT / "episodes" / f"{stem}.npz", allow_pickle=False)
    count = min(int(left.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(right.get(cv2.CAP_PROP_FRAME_COUNT)))
    target = dense.OUT / "videos" / f"{case}-17-vs-73-rays-world{world}.mp4"
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"),
                             12.5, (1280, 544))
    if not writer.isOpened():
        raise RuntimeError("Cannot create comparison video")
    try:
        for index in range(count):
            ok_l, image_l = left.read()
            ok_r, image_r = right.read()
            if not ok_l or not ok_r:
                raise RuntimeError(f"Missing source frame {index} in {stem}")
            panel = np.concatenate([image_l[:480, :640], image_r[:480, :640]], axis=1)
            cv2.rectangle(panel, (0, 0), (639, 42), (28, 30, 34), -1)
            cv2.rectangle(panel, (640, 0), (1279, 42), (28, 30, 34), -1)
            cv2.putText(panel, "17 LOW RAYS", (17, 29), cv2.FONT_HERSHEY_SIMPLEX,
                        .72, (230, 236, 238), 2)
            cv2.putText(panel, "73 LOW RAYS", (657, 29), cv2.FONT_HERSHEY_SIMPLEX,
                        .72, (230, 236, 238), 2)
            cv2.line(panel, (639, 0), (639, 479), (250, 250, 250), 2)
            foot = np.full((64, 1280, 3), (29, 31, 35), dtype=np.uint8)
            sample = min(index*4, len(left_trace["time_s"])-1)
            stamp = float(left_trace["time_s"][sample])
            for origin_x, trace, row in ((16, left_trace, left_rows[case, world]),
                                         (657, right_trace, right_rows[case, world])):
                gap = float(trace["gap_evaluator_m"][sample])
                stop = row["stop_command_s"]
                cv2.putText(foot, f"t={stamp:.2f}s  gap={gap:+.3f}m; stop={stop:.2f}s",
                            (origin_x, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            .62, (226, 233, 237), 1, cv2.LINE_AA)
            cv2.putText(foot, "Same low mount, 1.2m range, 15 Hz, scene and native body; ideal rays",
                        (16, 52), cv2.FONT_HERSHEY_SIMPLEX,
                        .57, (160, 178, 191), 1, cv2.LINE_AA)
            writer.write(np.concatenate([panel, foot], axis=0))
    finally:
        writer.release()
        left.release()
        right.release()
        left_trace.close()
        right_trace.close()
    return {"case": case, "world": world, "path": str(target),
            "sha256": trial.sha(target), "frames": count,
            "source_17_sha256": trial.sha(source_left),
            "source_73_sha256": trial.sha(source_right)}


def main() -> None:
    left = json.loads((sparse.OUT / "report.json").read_text())
    right = json.loads((dense.OUT / "report.json").read_text())
    left_rows = {(r["case"], r["world"]): r for r in left["results"]
                 if r["mode"] == "mapped_low_guard"}
    right_rows = {(r["case"], r["world"]): r for r in right["results"]}
    clips = [make(case, world, left_rows, right_rows) for case, world in CASES]
    manifest = {"schema": "robot-learning-lab.ape-robocasa-scan-comparison-media.v1",
                "sparse_report_sha256": trial.sha(sparse.OUT / "report.json"),
                "dense_report_sha256": trial.sha(dense.OUT / "report.json"),
                "clips": clips}
    (dense.OUT / "comparison-videos.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print([(c["case"], c["world"], c["frames"]) for c in clips])


if __name__ == "__main__":
    main()
