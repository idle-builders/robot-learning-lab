"""Make a labeled comparison from the audited complete-episode videos."""

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
BAD = ROOT / "artifacts/ape-robocasa-reacquire-frozen-sector-v596/report.json"
FIXED = ROOT / "artifacts/ape-robocasa-reacquire-frozen-sector-dual-v597/report.json"
OUT = FIXED.parent / "frozen-sector-one-vs-two-heights.mp4"


def source(report, index):
    episode = json.loads(report.read_text())["episodes"][index]
    path = Path(episode["video"]["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == episode["video"]["sha256"]
    return path, episode["video"]["frames"]


def main():
    bad_path, bad_count = source(BAD, 0)
    fixed_path, fixed_count = source(FIXED, 0)
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    font = ImageFont.truetype(font_path, 20)
    small = ImageFont.truetype(font_path, 16)
    readers = [imageio.get_reader(bad_path), imageio.get_reader(fixed_path)]
    try:
        with imageio.get_writer(OUT, fps=5, codec="libx264", quality=8) as writer:
            for index in range(max(bad_count, fixed_count)):
                canvas = Image.new("RGB", (1568, 496), "#10171b")
                draw = ImageDraw.Draw(canvas)
                draw.text((16, 12), "30 cm owner range repeats stale values while person approaches", font=font,
                          fill="white")
                for panel, count in enumerate((bad_count, fixed_count)):
                    frame = Image.fromarray(readers[panel].get_data(min(index, count - 1)))
                    canvas.paste(frame.resize((780, 396)), (panel * 780, 43))
                draw.text((15, 448), "30 cm only: no interruption; 2.6 cm person gap",
                          font=small, fill="#ff9e9e")
                draw.text((795, 448), "30 + 45 cm: cancel + retreat; 26.4 cm person gap",
                          font=small, fill="#a8e6bc")
                writer.append_data(np.asarray(canvas))
    finally:
        for reader in readers:
            reader.close()
    manifest = {
        "video": str(OUT),
        "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
        "frames": max(bad_count, fixed_count),
        "fps": 5,
        "left_source": str(bad_path),
        "right_source": str(fixed_path),
        "alignment": "Frames are played by capture sequence; differing asynchronous detector timings mean this is not exact simulation-time synchronization.",
    }
    OUT.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
