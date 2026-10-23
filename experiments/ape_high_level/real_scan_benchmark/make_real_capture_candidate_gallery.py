"""Show the actual middle views of visually screened real-capture routes."""

import json
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "assets/real_indoor_scenes"
INDEX = ASSETS / "real_capture_33_moving_route_index.json"
OUTPUT = ASSETS / "real_capture_15_candidate_gallery.jpg"


def middle_view(row):
    trace_dir = ROOT / row["trace"]
    if row["scene_id"].startswith("habitat_gs/"):
        sheet = Image.open(trace_dir.parent / "full_route_9.jpg").convert("RGB")
        tile_width, tile_height = sheet.width // 3, sheet.height // 3
        image = sheet.crop((tile_width, tile_height, 2 * tile_width, 2 * tile_height))
        return image.crop((0, 24, image.width, image.height))
    return Image.open(trace_dir.parent / "middle.jpg").convert("RGB")


def main():
    index = json.loads(INDEX.read_text())
    rows = [row for row in index["scenes"]
            if row["owner_controller_split"] in ("development", "holdout")]
    assert len(rows) == 15
    width, height, label_height = 320, 240, 34
    page = Image.new("RGB", (5 * width, 3 * (height + label_height) + 35), "#202020")
    draw = ImageDraw.Draw(page)
    draw.text((10, 10), "15 provisional real-capture moving-view candidates | 0.42 m camera", fill="white")
    for index, row in enumerate(rows):
        image = middle_view(row).resize((width, height))
        x, y = (index % 5) * width, 35 + (index // 5) * (height + label_height)
        page.paste(image, (x, y + label_height))
        label = row["scene_id"] + (" [H]" if row["owner_controller_split"] == "holdout" else "")
        draw.text((x + 5, y + 9), label, fill="white")
    page.save(OUTPUT, quality=90)
    print(OUTPUT)


if __name__ == "__main__":
    main()
