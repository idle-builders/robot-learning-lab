"""One-page visual audit of the official Replica moving-camera room probes."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
SCENES = (
    "apartment_0", "apartment_1", "apartment_2",
    "frl_apartment_0", "frl_apartment_1", "frl_apartment_2",
    "frl_apartment_3", "frl_apartment_4", "frl_apartment_5",
    "hotel_0", "office_0", "office_1", "office_2", "office_3", "office_4",
    "room_0", "room_1", "room_2",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=Path, default=ROOT / "assets/real_indoor_scenes/replica/ptex_routes")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/real_indoor_scenes/replica/replica_18_ptex_gallery.jpg")
    args = parser.parse_args()
    width, height, caption = 320, 240, 35
    page = Image.new("RGB", (6 * width, 3 * (height + caption) + 48), "#202020")
    draw = ImageDraw.Draw(page)
    draw.text((10, 12), "Replica v1: 18 real-capture scene variants | 0.42 m moving-camera middle frames", fill="white")
    for index, scene in enumerate(SCENES):
        folder = args.routes / scene
        trace = json.loads((folder / "trace.json").read_text())
        image = Image.open(folder / "middle.jpg").convert("RGB")
        image = image.resize((width, height))
        x = (index % 6) * width
        y = 48 + (index // 6) * (height + caption)
        page.paste(image, (x, y + caption))
        draw.text((x + 7, y + 9), "%s | %.1f m" %
                  (scene, trace["rendered_path_polyline_m"]), fill="white")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    page.save(output, quality=88)
    print(output)


if __name__ == "__main__":
    main()
