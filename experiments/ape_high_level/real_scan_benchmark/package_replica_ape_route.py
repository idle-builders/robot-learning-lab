"""Turn a Replica route's first-person frames into inspectable video and sheet."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw


def package(route_dir):
    trace = json.loads((route_dir / "trace.json").read_text())
    files = [route_dir / row["image"] for row in trace["frames"]]
    if not files or any(not path.is_file() for path in files):
        raise ValueError("missing route frames")

    video = route_dir / "ape_height_route.mp4"
    with imageio.get_writer(str(video), fps=12, codec="libx264", quality=7) as writer:
        for path in files:
            writer.append_data(imageio.imread(path))

    selected = sorted(set(round(i * (len(files) - 1) / 8) for i in range(9)))
    thumb_w, thumb_h = 320, 240
    sheet = Image.new("RGB", (thumb_w * 3, (thumb_h + 27) * 3), "#1e1e1e")
    draw = ImageDraw.Draw(sheet)
    for slot, frame_index in enumerate(selected):
        thumbnail = Image.open(files[frame_index]).convert("RGB")
        thumbnail.thumbnail((thumb_w, thumb_h))
        x = (slot % 3) * thumb_w
        y = (slot // 3) * (thumb_h + 27)
        sheet.paste(thumbnail, (x, y + 27))
        draw.text((x + 7, y + 6), "%s | frame %d" % (trace["scene_id"], frame_index), fill="white")
    sheet.save(route_dir / "ape_height_route_9.jpg", quality=88)
    for label, frame_index in (("first", 0), ("middle", len(files) // 2),
                               ("last", len(files) - 1)):
        Image.open(files[frame_index]).convert("RGB").save(
            route_dir / (label + ".jpg"), quality=90
        )
    print(video)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("route_dir", type=Path)
    package(parser.parse_args().route_dir)


if __name__ == "__main__":
    main()
