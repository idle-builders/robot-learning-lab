"""Contact sheet of MuSHRoom source photographs; never label it as simulation."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[3] / "assets/real_indoor_scenes/mushroom"
CELL_W, CELL_H = 150, 204
LABEL_W = 130


def main() -> None:
    manifest = json.loads((ROOT / "preview_manifest.json").read_text())
    rooms = sorted(manifest["rooms"])
    gallery = Image.new("RGB", (LABEL_W + 8 * CELL_W, 44 + len(rooms) * CELL_H), "#ffffff")
    draw = ImageDraw.Draw(gallery)
    draw.text((12, 12), "MuSHRoom: actual iPhone frames (recorded views, not ape simulation)", fill="#202020")
    for row, room in enumerate(rooms):
        y = 44 + row * CELL_H
        draw.text((12, y + 8), room.replace("_", " "), fill="#202020")
        paths = sorted((ROOT / "previews" / room).glob("*.jpg"))[:8]
        for col, path in enumerate(paths):
            with Image.open(path) as raw:
                photo = ImageOps.exif_transpose(raw).convert("RGB")
                thumb = ImageOps.fit(photo, (CELL_W - 4, CELL_H - 26))
            x = LABEL_W + col * CELL_W
            gallery.paste(thumb, (x, y))
            draw.text((x + 2, y + CELL_H - 22), path.stem.removeprefix("frame_"), fill="#333333")
    out = ROOT / "mushroom_recorded_10_room_gallery.jpg"
    gallery.save(out, quality=90)
    print(out)


if __name__ == "__main__":
    main()
