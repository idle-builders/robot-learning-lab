"""Make an explicit real-photo versus novel-render sheet for one room trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def tile(path: Path, label: str, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    canvas = Image.new("RGB", size, (25, 27, 31))
    image = ImageOps.contain(image, (size[0] - 12, size[1] - 38))
    canvas.paste(image, ((size[0] - image.width) // 2, 32 + (size[1] - 38 - image.height) // 2))
    ImageDraw.Draw(canvas).text((8, 8), label, fill="white")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", required=True)
    parser.add_argument("--appearance", required=True, type=Path)
    parser.add_argument("--route", required=True, type=Path)
    args = parser.parse_args()
    screen = json.loads((args.appearance / "appearance_screen.json").read_text())
    if len(screen["heldout"]) != 2:
        raise ValueError("This sheet expects two withheld real frames")
    sheet = Image.new("RGB", (1200, 810), (17, 19, 23))
    for pair, item in enumerate(screen["heldout"]):
        for col, (key, kind) in enumerate((("real", "RECORDED"), ("render", "HELD-OUT RENDER"))):
            label = f"{args.room} {item['frame']}  {kind}"
            sheet.paste(tile(Path(item[key]), label, (300, 510)), ((pair * 2 + col) * 300, 0))
    route_frames = sorted(args.route.glob("frame_*.png"))
    if len(route_frames) != 3:
        raise ValueError("Expected first, middle and last route frames")
    for col, path in enumerate(route_frames):
        label = f"NOVEL 0.42 m ROUTE  {['start', 'middle', 'end'][col]}"
        sheet.paste(tile(path, label, (400, 300)), (col * 400, 510))
    output = args.appearance / "real_vs_render_and_route.jpg"
    sheet.save(output, quality=91)
    print(output)


if __name__ == "__main__":
    main()
