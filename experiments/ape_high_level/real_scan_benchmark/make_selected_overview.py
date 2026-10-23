"""One actual, registered camera photo per visually screened indoor room."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from make_recorded_gallery import DATA, turn_upright


ROOT = Path(__file__).resolve().parent
FRAMES = {
    "42444750": 8, "47670315": 5, "42444474": 4, "42445057": 8,
    "42445680": 11, "42446492": 3, "42897815": 0, "42898555": 2,
    "42899795": 7, "43649754": 1, "44358289": 5, "44796450": 5,
    "45261080": 11, "45261598": 3, "47334539": 7, "45662809": 5,
    "47332764": 5, "47333138": 7, "47331159": 8, "47429889": 8,
    "47204505": 7, "47430726": 4, "47669984": 9, "47670184": 5,
    "48458675": 3,
}


def main() -> None:
    screen = json.loads((ROOT / "splat_recorded_view_screen.json").read_text())
    split = {s["video_id"]: s for s in
             json.loads((ROOT / "splat_candidate_split.json").read_text())["scenes"]}
    ids = screen["qualified_appearance_video_ids"]
    if set(ids) != set(FRAMES):
        raise ValueError("Representative photo choices do not match frozen visual screen")
    tile_w, tile_h = 320, 286
    canvas = Image.new("RGB", (5 * tile_w, 5 * tile_h), "#1c1c1c")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for count, video_id in enumerate(ids):
        scene_dir = DATA / video_id
        record = json.loads((scene_dir / "scene.json").read_text())
        index = FRAMES[video_id]
        photo = Image.open(scene_dir / "frames" / f"{index:02d}" / "rgb_vga.png").convert("RGB")
        pose = np.loadtxt(scene_dir / "camera.traj")
        photo = turn_upright(photo, pose, record["selected_frames"][index]["timestamp_s"])
        photo.thumbnail((tile_w - 10, tile_h - 47))
        x, y = (count % 5) * tile_w, (count // 5) * tile_h
        canvas.paste(photo, (x + (tile_w - photo.width) // 2, y + 32))
        kind = split[video_id]["tier"]
        fold = split[video_id]["agent_fold"]
        draw.text((x + 5, y + 5), f"{count + 1:02d}  {video_id}  {kind} / {fold}", fill="white", font=font)
        draw.text((x + 5, y + tile_h - 16),
                  ", ".join(list(record["annotation_labels"])[:4])[:48], fill="#d0d0d0", font=font)
    output = DATA / "splat_matched_25_recorded_rooms.jpg"
    canvas.save(output, quality=91)
    print(output)


if __name__ == "__main__":
    main()
