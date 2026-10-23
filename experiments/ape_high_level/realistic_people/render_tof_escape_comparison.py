"""Show held and sensor-driven escape episodes at matching cube travel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load(run: Path) -> tuple[list[np.ndarray], list[float]]:
    tag = run.name.rsplit("-", 1)[-1]
    path = run / f"low-tof-intrusion-explainer-{tag}.mp4"
    cap = cv2.VideoCapture(str(path))
    images = []
    while True:
        good, image = cap.read()
        if not good:
            break
        images.append(image)
    cap.release()
    assert len(images) == (36 - 8) * 5, (path, len(images))
    evidence = run / "evidence"
    stamps = json.loads((evidence / "range-fan-trace.json").read_text())["frames"]
    tof = json.loads((evidence / "low-tof-trace.json").read_text())["samples"]
    first_x = tof[0]["cube_center_xy_m_evaluator_only"][0]
    travel = []
    for frame in range(8, 36):
        stamp = stamps[frame]["timeline_time_s"]
        sample = max((row for row in tof if row["sim_time_s"] <= stamp + 1e-6),
                     key=lambda row: row["sim_time_s"])
        travel.append(abs(sample["cube_center_xy_m_evaluator_only"][0] - first_x))
    return images, travel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("held_run", type=Path)
    parser.add_argument("escape_run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    held_images, held_travel = load(args.held_run)
    escape_images, escape_travel = load(args.escape_run)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (1920, 400))
    for target in np.linspace(0., .5, 26):
        indexes = [min(range(1, 28), key=lambda index: abs(travel[index] - target))
                   for travel in (held_travel, escape_travel)]
        pair = np.hstack([images[index * 5 + 2]
                          for images, index in ((held_images, indexes[0]),
                                                (escape_images, indexes[1]))])
        canvas = np.full((400, 1920, 3), (22, 25, 29), dtype=np.uint8)
        canvas[40:] = pair
        cv2.putText(canvas, "HELD: no response", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .68, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, "SENSOR-DRIVEN: 28 cm side escape", (972, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .68, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"cube travel {target * 100:.0f} cm | separate live runs",
                    (500, 28), cv2.FONT_HERSHEY_SIMPLEX, .49,
                    (210, 210, 210), 1, cv2.LINE_AA)
        for _ in range(4):
            writer.write(canvas)
    writer.release()
    print(args.output)


if __name__ == "__main__":
    main()
