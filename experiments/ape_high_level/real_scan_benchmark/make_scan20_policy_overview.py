"""Show actual healthy-arm room videos in a compact 20-room contact reel."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
SOURCE = ART / "scan20_shared_policy_sources.json"
OUT = ROOT / "assets/real_indoor_scenes/scan20_shared_owner_policy_20261015"
RAW_VIDEO = OUT / "scan20_room_overview_raw.mp4"
VIDEO = OUT / "scan20_room_overview_h264.mp4"
SHEET = OUT / "scan20_room_contact_sheet.jpg"
W, H = 480, 270
FPS = 10


def label(scene: str) -> str:
    return scene.removeprefix("capture_").replace("_", " ")[:30]


def clip(folder: Path, phase: str) -> Path | None:
    candidates = ({
        "owner": [folder / "initial/owner_approach/first_person.mp4"],
        "search": [folder / "help_present/episode.mp4",
                   folder / "wide_present/episode.mp4",
                   folder / "initial/find_present/episode.mp4"],
        "approach": [folder / "walk/first_person.mp4",
                     folder / "final_view/episode.mp4",
                     folder / "initial/find_present/episode.mp4"],
    })[phase]
    return next((p for p in candidates if p.is_file()), None)


def frame(path: Path | None, alpha: float) -> np.ndarray:
    if path is None:
        return np.full((H, W, 3), 35, np.uint8)
    capture = cv2.VideoCapture(str(path))
    count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, round(alpha * (count - 1)))
    okay, image = capture.read()
    capture.release()
    if not okay:
        return np.full((H, W, 3), 35, np.uint8)
    scale = min(W / image.shape[1], H / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale),
                                 round(image.shape[0] * scale)))
    canvas = np.full((H, W, 3), 22, np.uint8)
    x = (W - resized.shape[1]) // 2
    y = (H - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def tile(case: dict, phase: str, alpha: float) -> np.ndarray:
    index = case["index"]
    light = "cool_dim" if index % 2 else "none"
    folder = OUT / case["scene_key"] / light / "healthy"
    image = frame(clip(folder, phase), alpha)
    shade = image.copy()
    cv2.rectangle(shade, (0, 0), (W, 35), (0, 0, 0), -1)
    image = cv2.addWeighted(shade, .7, image, .3, 0)
    cv2.putText(image, f"{index + 1:02d}  {label(case['scene_key'])}",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .57, (255, 255, 255), 1,
                cv2.LINE_AA)
    status = "missing"
    summary = folder / "summary.json"
    if summary.is_file():
        status = json.loads(summary.read_text()).get("result", "unknown")
    cv2.putText(image, f"{phase}  |  {status}", (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, .47, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main() -> None:
    cases = json.loads(SOURCE.read_text())["sources"]
    if len(cases) != 20:
        raise ValueError("Expected 20 independent captured rooms")
    cases = [{**case, "index": i} for i, case in enumerate(cases)]
    OUT.mkdir(parents=True, exist_ok=True)
    contact = np.vstack([np.hstack([tile(c, "owner", .65) for c in
        cases[i:i + 4]]) for i in range(0, 20, 4)])
    cv2.imwrite(str(SHEET), contact, [cv2.IMWRITE_JPEG_QUALITY, 88])
    writer = cv2.VideoWriter(str(RAW_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"),
                            FPS, (W * 2, H * 2))
    if not writer.isOpened():
        raise RuntimeError("Could not open overview video writer")
    try:
        for i in range(0, 20, 4):
            group = cases[i:i + 4]
            for phase in ("owner", "search", "approach"):
                for tick in range(FPS * 2):
                    alpha = tick / max(1, FPS * 2 - 1)
                    tiles = [tile(c, phase, alpha) for c in group]
                    writer.write(np.vstack((np.hstack(tiles[:2]),
                                             np.hstack(tiles[2:]))))
    finally:
        writer.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i",
        str(RAW_VIDEO), "-c:v", "libx264", "-crf", "22", "-pix_fmt",
        "yuv420p", "-movflags", "+faststart", str(VIDEO)], check=True)
    RAW_VIDEO.unlink()
    print(SHEET.relative_to(ROOT), VIDEO.relative_to(ROOT))


if __name__ == "__main__":
    main()
