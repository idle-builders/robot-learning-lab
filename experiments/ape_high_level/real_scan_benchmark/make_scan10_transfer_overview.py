"""Contact sheet and short reel from the actual ten transfer episodes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

import make_scan20_policy_overview as shared


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-real-room-next-captures-20261014/scan10_policy_transfer_sources.json"
OUT = ROOT / "assets/real_indoor_scenes/scan10_policy_transfer_20261015"
SHEET = OUT / "scan10_transfer_contact_sheet.jpg"
RAW = OUT / "scan10_transfer_overview_raw.mp4"
VIDEO = OUT / "scan10_transfer_overview_h264.mp4"


def main() -> None:
    cases = json.loads(SOURCE.read_text())["sources"]
    if len(cases) != 10:
        raise ValueError("Expected ten transfer rooms")
    shared.OUT = OUT
    cases = [{**case, "index": i} for i, case in enumerate(cases)]
    OUT.mkdir(parents=True, exist_ok=True)
    contact = np.vstack([np.hstack([shared.tile(c, "owner", .65)
        for c in cases[i:i + 2]]) for i in range(0, 10, 2)])
    cv2.imwrite(str(SHEET), contact, [cv2.IMWRITE_JPEG_QUALITY, 88])
    writer = cv2.VideoWriter(str(RAW), cv2.VideoWriter_fourcc(*"mp4v"),
                            shared.FPS, (shared.W * 2, shared.H))
    if not writer.isOpened():
        raise RuntimeError("Could not open transfer overview")
    try:
        for i in range(0, 10, 2):
            pair = cases[i:i + 2]
            for phase in ("owner", "search", "approach"):
                for tick in range(shared.FPS * 2):
                    alpha = tick / max(1, shared.FPS * 2 - 1)
                    writer.write(np.hstack([shared.tile(c, phase, alpha)
                                            for c in pair]))
    finally:
        writer.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(RAW),
        "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(VIDEO)], check=True)
    RAW.unlink()
    print(SHEET.relative_to(ROOT), VIDEO.relative_to(ROOT))


if __name__ == "__main__":
    main()
