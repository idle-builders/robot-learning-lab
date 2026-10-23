"""Inspect spatially separated, relatively low iPhone room recordings."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from screen_mushroom_camera_heights import POSES, camera_centres


ROOT = Path(__file__).resolve().parents[3] / "assets/real_indoor_scenes/mushroom"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", required=True)
    args = parser.parse_args()
    room = args.room
    archive_path = ROOT / room / f"{room}_iphone.tar.gz"
    output = ROOT / room / "relative_low_recorded_gallery.jpg"
    pose_file = POSES / f"{room}_images.txt"
    names = [tokens[-1] for line in pose_file.read_text().splitlines()
             if len(tokens := line.replace(",", "").split()) == 10 and tokens[0].isdigit()
             and tokens[-1].endswith(".jpg")]
    xyz = camera_centres(pose_file)
    if len(names) != len(xyz):
        raise ValueError("Pose and image names differ")
    low = np.flatnonzero(xyz[:, 2] <= xyz[:, 2].min() + 0.30)
    chosen = [int(low[np.argmin(xyz[low, 2])])]
    while len(chosen) < min(10, len(low)):
        separation = np.min(np.linalg.norm(
            xyz[low, None, :2] - xyz[np.asarray(chosen)][None, :, :2], axis=2), axis=1)
        separation[np.isin(low, chosen)] = -1
        chosen.append(int(low[np.argmax(separation)]))
    gallery = Image.new("RGB", (5 * 270, 2 * 300), "#202124")
    draw = ImageDraw.Draw(gallery)
    font = ImageFont.load_default()
    selections = []
    with tarfile.open(archive_path) as archive:
        for i, index in enumerate(chosen):
            name = names[index]
            member = archive.getmember(f"room_datasets/{room}/iphone/long_capture/images/{name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(name)
            image = Image.open(source).convert("RGB")
            image.thumbnail((256, 265))
            x, y = i % 5 * 270, i // 5 * 300
            gallery.paste(image, (x + (270 - image.width)//2, y + 25))
            draw.text((x + 5, y + 4), f"{name}  z={xyz[index, 2]:.2f}", fill="white", font=font)
            selections.append({"frame": name, "camera_center_xyz_m": xyz[index].tolist()})
    gallery.save(output, quality=91)
    (ROOT / room / "relative_low_recorded_gallery.json").write_text(json.dumps({
        "source": "https://zenodo.org/records/10230733", "selection": "30 cm band above lowest recorded camera, farthest XY sampled",
        "caution": "The band is relative to the phone's lowest pose. It is not 0.42 m above Faro floor until aligned.",
        "frames": selections,
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
