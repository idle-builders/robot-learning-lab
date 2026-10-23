"""Inspect ape-height mesh video frames without relabeling them photographic."""

import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from prepare_arkit_corpus import ASSETS


def main() -> None:
    results = json.loads((ASSETS / "mesh_probe_batch.json").read_text())["results"]
    font = ImageFont.load_default()
    for moment in ("first", "last"):
        canvas = Image.new("RGB", (5 * 320, 5 * 250), "#212121")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(results):
            x, y = index % 5 * 320, index // 5 * 250
            scene_id = row["scene_id"]
            video = ASSETS / scene_id / "mesh_probe/first_person.mp4"
            if video.exists():
                reader = imageio.get_reader(str(video))
                try:
                    frame_index = 0 if moment == "first" else reader.count_frames() - 1
                    frame = Image.fromarray(reader.get_data(frame_index)).convert("RGB")
                finally:
                    reader.close()
                frame.thumbnail((316, 226))
                canvas.paste(frame, (x + (320 - frame.width) // 2, y + 22))
            label = (f"{scene_id}  "
                     f"{'stop' if row.get('truth_standoff_within_band') else 'FAIL'}  "
                     f"{row.get('mean_valid_depth_fraction', 0):.0%} depth")
            draw.text((x + 5, y + 5), label, font=font, fill="white")
        output = ASSETS / f"mesh_probe_{moment}_25.jpg"
        canvas.save(output, quality=87)
        print(output)


if __name__ == "__main__":
    main()
