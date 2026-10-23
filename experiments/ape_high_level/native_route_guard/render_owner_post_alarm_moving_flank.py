"""Make compact paired continuation videos from audited V39 owner episodes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

import owner_post_alarm_moving_flank as trial


PAIRS = {
    "parked-raw-vs-joint-imu": (
        ("raw_rgbd", "parked_after_clear"),
        ("joint_imu", "parked_after_clear")),
    "return-joint-imu-vs-imu-lost": (
        ("joint_imu", "blind_return_flank_fast_response"),
        ("imu_lost", "blind_return_flank_fast_response")),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frames(path, skip):
    reader = imageio.get_reader(path)
    result = []
    try:
        for index, frame in enumerate(reader):
            if index >= skip:
                result.append(frame)
    finally:
        reader.close()
    assert result
    return result


def render_pair(name, left, right):
    left_run, right_run = left["run"], right["run"]
    left_frames = frames(left_run["video"]["path"],
                         left_run["source_video_frames"])
    right_frames = frames(right_run["video"]["path"],
                          right_run["source_video_frames"])
    out = trial.OUT / "comparisons" / f"{name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    total = max(len(left_frames), len(right_frames))
    with imageio.get_writer(out, fps=5, codec="libx264",
                            macro_block_size=None) as writer:
        for index in range(total):
            canvas = np.full((390, 1280, 3), (16, 22, 25), dtype=np.uint8)
            for column, item, bank in ((0, left, left_frames),
                                       (1, right, right_frames)):
                frame = bank[min(index, len(bank)-1)]
                scaled = cv2.resize(frame, (620, 315))
                x0 = 10+column*640
                canvas[62:377, x0:x0+620] = scaled
                label = item["mode"].replace("_", " + ")
                cv2.putText(canvas, label, (x0, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, .68,
                            (235, 240, 245), 2)
                outcome = item["run"]["outcome"].replace("_", " ")
                cv2.putText(canvas, outcome[:42], (x0, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, .43,
                            (160, 225, 220), 1)
            writer.append_data(canvas)
    return {"path": str(out), "sha256": sha(out), "frames": total,
            "shape": [390, 1280, 3],
            "left": {"mode": left["mode"], "case": left["case"],
                     "source_video_sha256": left_run["video"]["sha256"]},
            "right": {"mode": right["mode"], "case": right["case"],
                      "source_video_sha256": right_run["video"]["sha256"]},
            "alignment": "paired continuation frame index; packet timestamps remain in source traces"}


def main():
    report_path = trial.OUT / "report.json"
    report = json.loads(report_path.read_text())
    lookup = {(row["mode"], row["case"]): row for row in report["results"]}
    videos = {name: render_pair(name, lookup[left], lookup[right])
              for name, (left, right) in PAIRS.items()}
    comparison = {"schema": "robot-learning-lab.ape-owner-moving-flank-comparison.v1",
                  "report_sha256": sha(report_path),
                  "source_sha256": sha(Path(__file__)),
                  "videos": videos}
    (trial.OUT / "comparison.json").write_text(
        json.dumps(comparison, indent=2)+"\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
