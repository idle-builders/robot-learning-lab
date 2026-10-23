"""Show synchronized owner-head and downward-chest RGB-D scene views."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from run import ROOT, sha
import screen_standoff_camera_depth_v171p as v171p


SOURCE = Path(__file__)
VIDEO = v171p.OUT / "owner-head-vs-downward-chest-fast-wide.mp4"
CASE = "hazard-fast_wide_turn-seed170-x0.30"


def annotate(frame: np.ndarray, label: str, sample: dict) -> np.ndarray:
    image = frame.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(image, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .62, (255, 255, 255), 2, cv2.LINE_AA)
    p10 = sample["front_left_low_p10_x_m"]
    distance = "unknown" if p10 is None else f"{100*p10:.1f}cm"
    status = "LOW CHANGE CUE" if sample["cue_now"] else "no cue"
    cv2.putText(image,
                f"t{sample['capture_time_s']:.1f}s | low depth "
                f"{distance} | {status}",
                (10, 49), cv2.FONT_HERSHEY_SIMPLEX,
                .51, (210, 225, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    report_file = v171p.OUT / "report.json"
    report = json.loads(report_file.read_text())
    rows = {row["camera"]: row for row in report["rows"]
            if row["source_case_id"] == CASE}
    if set(rows) != set(v171p.CAMERAS):
        raise AssertionError("Paired cameras missing")
    source_images = []
    count = 0
    with imageio.get_writer(VIDEO, fps=10, codec="libx264",
                            quality=7) as writer:
        for head, chest in zip(rows["lower_rgb"]["samples"],
                               rows["chest_low70_x45"]["samples"]):
            if head["capture_time_s"] != chest["capture_time_s"]:
                raise AssertionError("Camera frames not synchronized")
            images = []
            for camera, sample, label in (
                    ("lower_rgb", head, "Owner-facing head RGB-D"),
                    ("chest_low70_x45", chest,
                     "Downward chest RGB-D candidate")):
                source = ROOT / sample["rgb_path"]
                if sha(source) != sample["rgb_sha256"]:
                    raise AssertionError("Source RGB frame changed")
                frame = cv2.imread(str(source))
                if frame is None:
                    raise RuntimeError(f"Cannot decode {source}")
                images.append(annotate(frame, label, sample))
                source_images.append({"camera": camera,
                                      "path": sample["rgb_path"],
                                      "sha256": sample["rgb_sha256"]})
            writer.append_data(cv2.cvtColor(
                np.concatenate(images, axis=1), cv2.COLOR_BGR2RGB))
            count += 1
    if count != len(v171p.TIMES_S):
        raise AssertionError("Paired camera video incomplete")
    metadata = {
        "schema": "robot-learning-lab.ape-standoff-camera-pair-video.v1",
        "source_sha256": sha(SOURCE),
        "v171_report_sha256": sha(report_file),
        "source_case_id": CASE,
        "source_images": source_images,
        "video_path": str(VIDEO.relative_to(ROOT)),
        "video_sha256": sha(VIDEO),
        "frames": count,
        "scope": "Matched 10Hz RGB frames from the same stationary historical "
                 "owner turn. Low-depth statistics use saved metric depth "
                 "and estimated camera pose; they are actor cues, not "
                 "prop/person labels. The two cameras are virtual body "
                 "mount candidates and no action is authorized by this "
                 "paired video alone.",
    }
    (v171p.OUT / "paired-video-report.json").write_text(json.dumps(
        metadata, indent=2) + "\n")
    print(VIDEO, count)


if __name__ == "__main__":
    main()
