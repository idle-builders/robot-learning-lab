"""Readable stage reel for the integrated photographic office task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def labeled(frame: np.ndarray, title: str) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 43), fill=(12, 17, 24))
    draw.text((12, 13), title, fill=(255, 255, 255))
    return np.asarray(image)


def frames(video: Path):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video}")
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run = args.run
    walk = json.loads((run / "walk/trace.json").read_text())
    route_start = next(i for i, row in enumerate(walk["actions"])
                       if row.get("route_decision") is not None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(args.output), fps=5, codec="libx264",
            quality=7, macro_block_size=8) as writer:
        for image_name, title in (("owner/owner_rgb.png", "1  OWNER FOUND"),
                                  ("teach/teach_rgb.png", "2  ITEM TAUGHT")):
            frame = np.asarray(Image.open(run / image_name).convert("RGB"))
            for _ in range(4):
                writer.append_data(labeled(frame, title))
        for frame in frames(run / "find_present/first_person.mp4"):
            writer.append_data(labeled(frame, "3  ITEM FOUND IN ROOM"))
        for index, frame in enumerate(frames(run / "walk/first_person.mp4")):
            title = ("4  DIRECT APPROACH" if index < route_start else
                     "5  OBSERVED-FREE ROUTE AROUND FURNITURE")
            writer.append_data(labeled(frame, title))
        for frame in frames(run / "final_view/first_person.mp4"):
            writer.append_data(labeled(frame, "6  FRESH ITEM RECHECK"))
    print(json.dumps({"output": str(args.output), "route_start_frame": route_start}))


if __name__ == "__main__":
    main()
