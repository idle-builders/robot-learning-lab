"""Synchronized V26 no-sensor versus no-map-guard success/failure clips."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import heldout_plan_scene as heldout
from kitchen_trial import sha


CASES = ("fast_right_new", "slow_right_new")


def make(case: str, rows: dict[tuple[str, str], dict]) -> dict:
    paths = [heldout.OUT / "no_sensor" / "videos" /
             f"{case}-none-world0.mp4",
             heldout.OUT / "online_plan_guard" / "videos" /
             f"{case}-mapped_low_guard-world0.mp4"]
    caps = [cv2.VideoCapture(str(path)) for path in paths]
    if not all(cap.isOpened() for cap in caps):
        raise RuntimeError(f"Missing V26 video for {case}")
    frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps)
    if frames != 100:
        raise RuntimeError(f"Expected 100 frames for {case}: {frames}")
    out = heldout.OUT / f"{case}-no-sensor-vs-online-guard-world0.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             12.5, (1920, 582))
    if not writer.isOpened():
        raise RuntimeError("MP4 writer failed")
    control = rows["no_sensor", case]
    guarded = rows["online_plan_guard", case]
    try:
        for index in range(frames):
            read = [cap.read() for cap in caps]
            if not all(ok for ok, _ in read):
                raise RuntimeError(f"Missing {case} frame {index}")
            panel = np.concatenate([frame[:480] for _, frame in read], axis=1)
            for x, title in ((0, "NO SENSOR"),
                             (960, "ONLINE LOW RANGE + BOUNDED FORECAST")):
                cv2.rectangle(panel, (x, 0), (x + 959, 43), (29, 32, 37), -1)
                cv2.putText(panel, title, (x+16, 29),
                            cv2.FONT_HERSHEY_SIMPLEX, .72,
                            (230, 236, 238), 2, cv2.LINE_AA)
            cv2.line(panel, (959, 0), (959, 479), (240, 240, 240), 2)
            footer = np.full((102, 1920, 3), (29, 32, 37), dtype=np.uint8)
            cv2.putText(footer,
                        f"RoboCasa layout 3 + box stool | {case} | t={index/12.5:.2f} s",
                        (18, 27), cv2.FONT_HERSHEY_SIMPLEX, .67,
                        (227, 233, 238), 1, cv2.LINE_AA)
            cv2.putText(footer,
                        f"No sensor: min body-person gap {control['min_modeled_body_person_gap_m']:+.3f} m; "
                        f"guard: stop {guarded['stop_command_s']:.2f} s, "
                        f"min gap {guarded['min_modeled_body_person_gap_m']:+.3f} m",
                        (18, 55), cv2.FONT_HERSHEY_SIMPLEX, .61,
                        (207, 214, 221), 1, cv2.LINE_AA)
            cv2.putText(footer,
                        "Scripted capsule; ideal rays/pose; rejected historical body's kinematic stop tail.",
                        (18, 84), cv2.FONT_HERSHEY_SIMPLEX, .56,
                        (167, 184, 196), 1, cv2.LINE_AA)
            writer.write(np.concatenate([panel, footer], axis=0))
    finally:
        writer.release()
        for cap in caps:
            cap.release()
    return {"case": case, "world": 0, "path": str(out),
            "frames": frames, "sha256": sha(out),
            "source_sha256": [sha(path) for path in paths]}


def main() -> None:
    report_path = heldout.OUT / "report.json"
    report = json.loads(report_path.read_text())
    rows = {(row["configuration"], row["case"]): row for row in report["results"]
            if row["world"] == 0}
    clips = [make(case, rows) for case in CASES]
    manifest = {"schema": "robot-learning-lab.ape-v26-comparison-media.v1",
                "report_sha256": sha(report_path),
                "source_sha256": sha(Path(__file__)), "clips": clips}
    (heldout.OUT / "comparison-videos.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print([(row["case"], row["frames"]) for row in clips])


if __name__ == "__main__":
    main()
