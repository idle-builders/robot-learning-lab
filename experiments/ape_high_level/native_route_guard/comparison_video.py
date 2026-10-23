"""Make a synchronized, directly inspectable paired kitchen crossing video."""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np

import kitchen_native as kitchen


OUT = kitchen.OUT / "joined-trial"
BASE = "slow_cross"
WORLD = 0


def load(mode: str):
    stem = f"{BASE}-{mode}-world{WORLD}"
    movie = cv2.VideoCapture(str(OUT / "videos" / f"{stem}.mp4"))
    if not movie.isOpened():
        raise RuntimeError(f"Cannot open {stem}")
    trace = np.load(OUT / "episodes" / f"{stem}.npz", allow_pickle=False)
    return movie, trace


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    global OUT, BASE, WORLD
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-dir", type=Path, default=OUT)
    parser.add_argument("--case", default=BASE)
    parser.add_argument("--world", type=int, default=WORLD)
    parser.add_argument("--name", default="slow-crossing-paired-world0.mp4")
    args = parser.parse_args()
    OUT, BASE, WORLD = args.trial_dir, args.case, args.world
    report = json.loads((OUT / "report.json").read_text())
    rows = {(r["case"], r["mode"], r["world"]): r for r in report["results"]}
    left, lt = load("none")
    right, rt = load("mapped_low_guard")
    frame_count = min(int(left.get(cv2.CAP_PROP_FRAME_COUNT)),
                      int(right.get(cv2.CAP_PROP_FRAME_COUNT)))
    destination = OUT / "videos" / args.name
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                             12.5, (1280, 544))
    if not writer.isOpened():
        raise RuntimeError("Cannot write paired MP4")
    try:
        for index in range(frame_count):
            ok_l, image_l = left.read()
            ok_r, image_r = right.read()
            if not ok_l or not ok_r:
                raise RuntimeError(f"Missing source frame {index}")
            panel = np.concatenate([image_l[:480, :640], image_r[:480, :640]], axis=1)
            cv2.rectangle(panel, (0, 0), (639, 42), (28, 30, 34), -1)
            cv2.rectangle(panel, (640, 0), (1279, 42), (28, 30, 34), -1)
            cv2.putText(panel, "NO SENSOR STOP", (17, 29),
                        cv2.FONT_HERSHEY_SIMPLEX, .72, (230, 236, 238), 2)
            cv2.putText(panel, "15 Hz LOW RANGE + MAP", (657, 29),
                        cv2.FONT_HERSHEY_SIMPLEX, .72, (230, 236, 238), 2)
            cv2.line(panel, (639, 0), (639, 479), (250, 250, 250), 2)
            foot = np.full((64, 1280, 3), (29, 31, 35), dtype=np.uint8)
            sample = min(index*4, len(lt["time_s"])-1)
            stamp = float(lt["time_s"][sample])
            gap_l = float(lt["gap_evaluator_m"][sample])
            gap_r = float(rt["gap_evaluator_m"][sample])
            stop = rows[BASE, "mapped_low_guard", WORLD]["stop_command_s"]
            cv2.putText(foot, f"t={stamp:.2f}s  modeled body/person gap: {gap_l:+.3f} m",
                        (16, 25), cv2.FONT_HERSHEY_SIMPLEX, .62, (226, 233, 237), 1, cv2.LINE_AA)
            cv2.putText(foot, f"t={stamp:.2f}s  modeled gap: {gap_r:+.3f} m; stop sent at {stop:.2f}s",
                        (657, 25), cv2.FONT_HERSHEY_SIMPLEX, .62, (226, 233, 237), 1, cv2.LINE_AA)
            cv2.putText(foot, "Historical articulated body + scripted capsule person; replayed stop tail",
                        (16, 52), cv2.FONT_HERSHEY_SIMPLEX, .57, (160, 178, 191), 1, cv2.LINE_AA)
            writer.write(np.concatenate([panel, foot], axis=0))
    finally:
        writer.release()
        left.release()
        right.release()
        lt.close()
        rt.close()
    manifest = {"video": str(destination), "sha256": digest(destination),
                "case": BASE, "world": WORLD,
                "frames": frame_count,
                "source_report_sha256": digest(OUT / "report.json"),
                "source_videos_sha256": {
                    mode: digest(OUT / "videos" /
                                 f"{BASE}-{mode}-world{WORLD}.mp4")
                    for mode in ("none", "mapped_low_guard")}}
    (OUT / "comparison-video.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(destination, frame_count)


if __name__ == "__main__":
    main()
