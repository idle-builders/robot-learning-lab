"""Contact sheet and short reel from actual frozen Scan30 task videos."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
OUT = ROOT / "assets/real_indoor_scenes/scan30_unified_policy_20261015"
REPORT = ART / "scan30_unified_policy_report.json"
SHEET = OUT / "scan30_room_contact_sheet.jpg"
RAW = OUT / "scan30_room_overview_raw.mp4"
VIDEO = OUT / "scan30_room_overview_h264.mp4"
W, H, FPS = 480, 270, 10


def first(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def media(row: dict, phase: str) -> Path | None:
    base = (ROOT / row["base_summary"]).parent
    show = (ROOT / row["show_summary"]).parent if row.get("show_summary") else None
    key = row["scene_key"]
    if phase == "owner":
        return first([base / "initial/owner_approach/first_person.mp4"])
    if phase == "search":
        return first(([show / "find_present/episode.mp4"] if show else []) + [
            base / "help_present/episode.mp4",
            base / "wide_present/episode.mp4",
            base / "initial/find_present/episode.mp4"])
    return first(([show / "continuation_action" / key / "range_gap/guarded_walk/first_person.mp4",
                   show / "continuation_action" / key / "range_gap/final_view/episode.mp4",
                   show / "continuation_action" / key / "healthy/guarded_walk/first_person.mp4",
                   show / "continuation_action" / key / "healthy/final_view/episode.mp4"]
                  if show else []) + [base / "walk/first_person.mp4",
                                         base / "final_view/episode.mp4"])


def frame(path: Path | None, fraction: float) -> np.ndarray:
    canvas = np.full((H, W, 3), 30, np.uint8)
    if path is None:
        return canvas
    reader = cv2.VideoCapture(str(path))
    count = max(1, int(reader.get(cv2.CAP_PROP_FRAME_COUNT)))
    reader.set(cv2.CAP_PROP_POS_FRAMES, round(fraction * (count - 1)))
    okay, picture = reader.read()
    reader.release()
    if not okay:
        return canvas
    scale = min(W / picture.shape[1], H / picture.shape[0])
    picture = cv2.resize(picture, (round(picture.shape[1] * scale),
                                   round(picture.shape[0] * scale)))
    x, y = (W - picture.shape[1]) // 2, (H - picture.shape[0]) // 2
    canvas[y:y + picture.shape[0], x:x + picture.shape[1]] = picture
    return canvas


def tile(row: dict | None, phase: str, fraction: float) -> np.ndarray:
    if row is None:
        return np.full((H, W, 3), 20, np.uint8)
    image = frame(media(row, phase), fraction)
    shade = image.copy()
    cv2.rectangle(shade, (0, 0), (W, 37), (0, 0, 0), -1)
    image = cv2.addWeighted(shade, .75, image, .25, 0)
    name = row["scene_key"].removeprefix("capture_").replace("_", " ")[:30]
    cv2.putText(image, f"{row['index']:02d}  {name}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, .54, (255, 255, 255), 1, cv2.LINE_AA)
    status = row["result"]
    if row.get("fault_exposure"):
        status = f"{row['fault_exposure']}  |  {status}"
    cv2.putText(image, f"{phase}  |  {status}", (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    rows = json.loads(REPORT.read_text())["rows"]
    if len(rows) != 30:
        raise ValueError("Expected 30 finished room attempts")
    cases = [{**row, "index": i + 1} for i, row in enumerate(rows)]
    contact = np.vstack([np.hstack([tile(row, "owner", .65)
        for row in cases[i:i + 5]]) for i in range(0, 30, 5)])
    cv2.imwrite(str(SHEET), contact, [cv2.IMWRITE_JPEG_QUALITY, 88])
    writer = cv2.VideoWriter(str(RAW), cv2.VideoWriter_fourcc(*"mp4v"),
                            FPS, (W * 2, H * 2))
    if not writer.isOpened():
        raise RuntimeError("Cannot open overview video writer")
    try:
        for start in range(0, 30, 4):
            group = cases[start:start + 4] + [None] * (4 - len(cases[start:start + 4]))
            for phase in ("owner", "search", "approach"):
                for tick in range(FPS * 2):
                    alpha = tick / (FPS * 2 - 1)
                    tiles = [tile(row, phase, alpha) for row in group]
                    writer.write(np.vstack((np.hstack(tiles[:2]),
                                             np.hstack(tiles[2:]))))
    finally:
        writer.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(RAW),
                    "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(VIDEO)], check=True)
    RAW.unlink()
    print(SHEET.relative_to(ROOT), VIDEO.relative_to(ROOT))


if __name__ == "__main__":
    main()
