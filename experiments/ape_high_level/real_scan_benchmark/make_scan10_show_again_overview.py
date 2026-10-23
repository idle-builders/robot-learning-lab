"""Short first-person comparison of the three owner-show-again continuations."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "assets/real_indoor_scenes/scan10_show_again_20261015"
RAW = OUT / "scan10_show_again_overview_raw.mp4"
VIDEO = OUT / "scan10_show_again_overview_h264.mp4"
SHEET = OUT / "scan10_show_again_contact_sheet.jpg"
CASES = (
    ("capture_e2974710c05b456681cbebe768d0db68", "Kitchen / Cookies", "Still unsure"),
    ("capture_morning_geometry_west_0p8m", "Morning room", "Confirmed"),
    ("capture_dining_north_1m", "Dining room", "Confirmed"),
)
W, H, FPS = 320, 208, 5


def fit(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(W / width, (H - 30) / height)
    resized = cv2.resize(frame, (max(1, round(width * scale)),
                                  max(1, round(height * scale))))
    canvas = np.zeros((H, W, 3), np.uint8)
    x = (W - resized.shape[1]) // 2
    y = 30 + (H - 30 - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def frames(path: Path) -> list[np.ndarray]:
    if path.suffix == ".png":
        frame = cv2.imread(str(path))
        return [frame] if frame is not None else []
    if not path.is_file():
        return []
    cap = cv2.VideoCapture(str(path))
    out = []
    try:
        while len(out) < 40:
            okay, frame = cap.read()
            if not okay:
                break
            out.append(frame)
    finally:
        cap.release()
    return out


@lru_cache(maxsize=12)
def clip(key: str, phase: str) -> list[np.ndarray]:
    folder = OUT / key
    action = folder / "continuation_action" / key / "healthy"
    path = {"show": folder / "teach/rgb_000.png",
            "find": folder / "find_present/episode.mp4",
            "walk": action / "guarded_walk/first_person.mp4",
            "review": action / "final_view/episode.mp4"}[phase]
    result = frames(path)
    return result or frames(folder / "find_present/episode.mp4")


def panel(key: str, name: str, outcome: str, phase: str,
          tick: int) -> np.ndarray:
    data = clip(key, phase)
    if not data:
        frame = np.zeros((H, W, 3), np.uint8)
    else:
        frame = fit(data[min(len(data) - 1, tick * max(1, len(data) // 10))])
    label = f"{name}  |  {phase}  |  {outcome}"
    cv2.putText(frame, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .43,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for phase in ("show", "find", "walk", "review"):
        rows.append(np.hstack([panel(*case, phase, 5) for case in CASES]))
    cv2.imwrite(str(SHEET), np.vstack(rows),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    writer = cv2.VideoWriter(str(RAW), cv2.VideoWriter_fourcc(*"mp4v"),
                            FPS, (W * len(CASES), H))
    if not writer.isOpened():
        raise RuntimeError("Could not open overview writer")
    try:
        for phase in ("show", "find", "walk", "review"):
            for tick in range(FPS * 2):
                writer.write(np.hstack([panel(*case, phase, tick)
                                        for case in CASES]))
    finally:
        writer.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(RAW),
        "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(VIDEO)], check=True)
    RAW.unlink()
    print(SHEET.relative_to(ROOT), VIDEO.relative_to(ROOT))


if __name__ == "__main__":
    main()
