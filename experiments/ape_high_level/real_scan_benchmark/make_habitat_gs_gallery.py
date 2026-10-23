"""Contact sheet of actual moving-camera episodes, first and final frames."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw

from screen_habitat_gs import ASSETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    batch = ("habitat_gs_probe_extended_batch.json" if args.extended
             else "habitat_gs_probe_batch.json")
    summary = json.loads((ASSETS / batch).read_text())
    rows = [row for row in summary["scenes"] if row["outcome"] != "probe_error"]
    page = Image.new("RGB", (960, 385 * len(rows)), "#222222")
    draw = ImageDraw.Draw(page)
    for slot, row in enumerate(rows):
        scene = row["scene_id"]
        split = "train" if int(scene[5:]) <= 55 else "val"
        video = ASSETS / split / scene / (
            "gs_probe_extended" if args.extended else "gs_probe") / "first_person.mp4"
        reader = imageio.get_reader(str(video))
        first = Image.fromarray(reader.get_data(0))
        last = Image.fromarray(reader.get_data(reader.count_frames() - 1))
        reader.close()
        y = slot * 385
        page.paste(first, (0, y + 25))
        page.paste(last, (480, y + 25))
        text = (f"{scene} | {row['outcome']} | stand-off truth "
                f"{row['true_target_surface_gap_m']:.2f} m | "
                f"path {row['actual_path_length_m']:.2f} m | "
                f"black pixels {row['mean_black_background_fraction']:.1%}")
        draw.text((6, y + 6), text, fill="white")
    path = ASSETS / ("habitat_gs_extended_gallery.jpg" if args.extended
                     else "habitat_gs_moving_camera_gallery.jpg")
    page.save(path, quality=90)
    print(path)


if __name__ == "__main__":
    main()
