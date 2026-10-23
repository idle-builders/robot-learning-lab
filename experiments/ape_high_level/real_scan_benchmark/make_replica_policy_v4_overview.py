"""Make a short, captioned first-person reel from the frozen v4 run."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "assets/real_indoor_scenes/replica_policy_v4_cached_three_20261015"
OUT = RUN / "overview.mp4"


def font(size: int):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def frames(path: Path) -> list[Image.Image]:
    reader = imageio.get_reader(str(path))
    try:
        images = [Image.fromarray(frame).convert("RGB") for frame in reader]
    finally:
        reader.close()
    if not images:
        raise ValueError(f"Empty first-person video: {path}")
    if len(images) > 15:
        step = max(1, len(images) // 15)
        images = images[::step][:15]
    while len(images) < 10:
        images += images[:10 - len(images)]
    return images


def owner_point(folder: Path) -> tuple[Image.Image, tuple[float, float]]:
    summary = json.loads((folder / "summary.json").read_text())
    help_record = summary["owner_help"]
    image = Image.open(ROOT / help_record["rgb"]).convert("RGB")
    return image, tuple(help_record["click_xy"])


def compose(image: Image.Image, heading: str, footer: str,
            point: tuple[float, float] | None = None,
            wrong: bool = False) -> Image.Image:
    canvas = Image.new("RGB", (640, 560), "#10151b")
    image = image.resize((640, 480))
    canvas.paste(image, (0, 48))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), heading, font=font(20), fill="#f4f7ff")
    draw.text((12, 532), footer, font=font(15), fill="#d7e4f4")
    if point is not None:
        x, y = point
        x, y = x * 640 / image.width, y * 480 / image.height + 48
        color = "#ff6961" if wrong else "#64e6a5"
        draw.ellipse((x - 10, y - 10, x + 10, y + 10),
                     outline=color, width=4)
        draw.line((x - 14, y, x + 14, y), fill=color, width=2)
        draw.line((x, y - 14, x, y + 14), fill=color, width=2)
    return canvas


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    apt_nominal = RUN / "apartment_0/original_nominal/healthy"
    apt_help = RUN / "apartment_0/lookalike_accurate/healthy"
    apt_wrong = RUN / "apartment_0/lookalike_wrong/healthy"
    hotel = RUN / "hotel_0/original_nominal/healthy"
    office = RUN / "office_3/original_nominal/healthy"
    clips = [
        (apt_nominal / "find_present/station_0/first_person.mp4",
         "Apartment: taught item search", "RGB-D proposals and owner teaching"),
        (apt_nominal / "walk/first_person.mp4",
         "Apartment: guarded approach", "Fresh depth and independent range checked"),
        (apt_nominal / "final_view/first_person.mp4",
         "Apartment: fresh final review", "Correct item offered to owner"),
        (hotel / "final_view/first_person.mp4",
         "Hotel: second room", "Correct fresh final offer"),
        (apt_help / "final_relook/breadcrumb_route/first_person.mp4",
         "Changed object and lighting", "Return for clear view stops when pose source ends"),
        (office / "walk/first_person.mp4",
         "Office: item identified", "No known-free standoff route; ask for help"),
    ]
    with imageio.get_writer(str(OUT), fps=5, codec="libx264", quality=6,
                            macro_block_size=8) as writer:
        for video, heading, footer in clips[:4]:
            for frame in frames(video):
                writer.append_data(np.asarray(compose(frame, heading, footer)))
        for folder, heading, footer, wrong in (
            (apt_help, "Owner points to changed item",
             "Point selects an already depth-localized proposal", False),
            (apt_wrong, "Wrong point rejected",
             "Identity stays unknown; no item walk", True),
        ):
            frame, point = owner_point(folder)
            for _ in range(12):
                writer.append_data(np.asarray(compose(frame, heading, footer, point, wrong)))
        for video, heading, footer in clips[4:]:
            for frame in frames(video):
                writer.append_data(np.asarray(compose(frame, heading, footer)))
    print(OUT)


if __name__ == "__main__":
    main()
