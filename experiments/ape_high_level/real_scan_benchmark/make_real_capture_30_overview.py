"""Show thirty distinct real indoor captures without confusing photos and renders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from make_recorded_gallery import turn_upright
from make_selected_overview import FRAMES


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "assets/real_indoor_scenes"
OUT = DATA / "real_capture_30_scene_overview.jpg"
MANIFEST = DATA / "real_capture_30_scene_overview.json"

HABITAT = (
    ("scene12", "gallery corridor"), ("scene14", "meeting room"),
    ("scene32", "bedroom"), ("scene36", "book gallery"),
    ("scene37", "painted corridor"), ("scene38", "hotel bedroom"),
    ("scene54", "dining showroom"), ("scene55", "hotel lobby"),
    ("scene63", "workshop / music"), ("scene64", "studio"),
)
MUSHROOM = (
    ("activity", "frame_00569.jpg", "activity lounge"),
    ("classroom", "frame_00113.jpg", "classroom"),
    ("coffee_room", "frame_00088.jpg", "coffee / meeting"),
    ("computer", "frame_00027.jpg", "computer lab"),
    ("honka", "frame_00332.jpg", "meeting room"),
    ("koivu", "frame_00134.jpg", "conference space"),
    ("kokko", "frame_00420.jpg", "kitchen / lounge"),
    ("olohuone", "frame_00569.jpg", "lounge"),
    ("sauna", "frame_00569.jpg", "sauna"),
    ("vr_room", "frame_00244.jpg", "VR room"),
)
APPLE = (
    ("42444750", "kitchen"), ("47670315", "living room"),
    ("42444474", "living room"), ("42898555", "lounge"),
    ("43649754", "bathroom"), ("44358289", "bar / dining"),
    ("45261080", "kitchen"), ("45261598", "sitting room"),
    ("47334539", "bedroom"), ("47204505", "kitchen"),
)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main() -> None:
    entries: list[dict] = []
    for scene, label in HABITAT:
        split = "val" if scene in {"scene63", "scene64"} else "train"
        path = DATA / "habitat_gs" / split / scene / "preview_0.jpg"
        entries.append(dict(dataset="Habitat-GS", scene_id=scene, setting=label,
                            evidence="ape-height simulated Gaussian-splat view",
                            source="https://huggingface.co/datasets/RukawaY/gs_scenes", image=str(path)))
    for scene, photo, label in MUSHROOM:
        path = DATA / "mushroom" / "previews" / scene / photo
        entries.append(dict(dataset="MuSHRoom", scene_id=scene, setting=label,
                            evidence="recorded iPhone photo; simulated view pending",
                            source="https://github.com/TUTvision/MuSHRoom", image=str(path)))
    for scene, label in APPLE:
        frame = FRAMES[scene]
        path = DATA / "arkit_v1" / scene / "frames" / f"{frame:02d}" / "rgb_vga.png"
        entries.append(dict(dataset="ARKitScenes", scene_id=scene, setting=label,
                            evidence="recorded pose-near photo; photographic simulator pending",
                            source="https://github.com/apple/ARKitScenes", image=str(path)))
    assert len(entries) == 30
    assert len({(x["dataset"], x["scene_id"]) for x in entries}) == 30

    tile_w, tile_h, columns = 336, 262, 5
    canvas = Image.new("RGB", (columns * tile_w, 6 * tile_h), "#202124")
    draw = ImageDraw.Draw(canvas)
    title_font, label_font = font(18), font(14)
    for i, entry in enumerate(entries):
        path = Path(entry["image"])
        if not path.exists():
            raise FileNotFoundError(path)
        photo = Image.open(path).convert("RGB")
        if entry["dataset"] == "ARKitScenes":
            scene_dir = path.parents[2]
            record = json.loads((scene_dir / "scene.json").read_text())
            pose = np.loadtxt(scene_dir / "camera.traj")
            frame = FRAMES[entry["scene_id"]]
            photo = turn_upright(photo, pose, record["selected_frames"][frame]["timestamp_s"])
        photo.thumbnail((tile_w - 12, tile_h - 58))
        x, y = (i % columns) * tile_w, (i // columns) * tile_h
        draw.rectangle((x + 1, y + 1, x + tile_w - 2, y + tile_h - 2), outline="#555a61", width=1)
        canvas.paste(photo, (x + (tile_w - photo.width) // 2, y + 30 + (tile_h - 58 - photo.height) // 2))
        draw.text((x + 7, y + 6), f"{i + 1:02d}  {entry['dataset']}  {entry['scene_id']}", fill="white", font=title_font)
        kind = "RENDER" if entry["dataset"] == "Habitat-GS" else "REAL PHOTO"
        draw.text((x + 7, y + tile_h - 24), f"{entry['setting']}   |   {kind}", fill="#dae5f0", font=label_font)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, quality=91)
    MANIFEST.write_text(json.dumps({"schema": "ape.real-capture-30-scene-overview.v1", "count": len(entries),
                                    "qualification": "Only Habitat-GS tiles are simulated novel views; no full owner episode is certified.",
                                    "entries": entries}, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
