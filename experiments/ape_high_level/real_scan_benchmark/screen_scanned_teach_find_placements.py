"""Freeze photographed-room teach/find placements before model outcomes.

Only previously reached owner viewpoints are screened. Object placement uses
reconstructed floor support and rendered visibility, never detector scores.
Every room remains in the output, including unsupported viewpoints.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from real_capture_dynamic_scene import (ScannedSittingPerson,
                                        TexturedCaptureScene,
                                        TexturedFloorObject)
from run_real_capture_owner_loop import SCENES, owner_pose


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen"
BATCH = BASE / "owner_batch_rgbd_yaw_arbitrated_crosschecked_personmasked_rangeprojected_v1.json"
OUT = BASE / "teach_find_placement_screen"
PROPS = ROOT / ".cache/ape-high-level/hope_meshes_eval"


def view(scene: TexturedCaptureScene, xy: np.ndarray, yaw: float,
         owner_xz: tuple[float, float] | None,
         width: int = 640, height: int = 360) -> tuple[np.ndarray, np.ndarray]:
    radians = math.radians(yaw)
    forward = np.array([math.sin(radians), 0., math.cos(radians)])
    eye = np.array([xy[0], scene.floor_y + .42, xy[1]])
    rgb, _, ids = scene.observe(eye, forward, owner_xz=owner_xz,
                                 width=width, height=height, return_prop_ids=True)
    return rgb, ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, default=BATCH)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    batch = json.loads(args.batch.read_text())
    human = ScannedSittingPerson(ROOT /
        "assets/real_indoor_scenes/scansceneverse/human_previews/sitting_person_isolated.glb")
    target = TexturedFloorObject(PROPS / "Parmesan.obj", scale=.012)
    distractor = TexturedFloorObject(PROPS / "CreamCheese.obj", scale=.012)
    results = []
    gallery = Image.new("RGB", (640 * 3, 390 * max(1, len(batch["scenes"]))), "#202020")
    draw = ImageDraw.Draw(gallery)
    gallery_row = 0
    for row in batch["scenes"]:
        item = {"scene_id": row["scene_id"], "name": row["name"],
                "source_owner_trace": row.get("trace"),
                "source_owner_status": row.get("outcome")}
        if not row.get("observed_standoff_band_met"):
            item["status"] = "no_owner_viewpoint"
            results.append(item)
            continue
        cfg_key = next(k for k, cfg in SCENES.items()
                       if k.startswith("capture_") and cfg["name"] == row["name"])
        cfg = SCENES[cfg_key]
        source = json.loads(cfg["trace"].read_text())
        trace = json.loads((ROOT / row["trace"]).read_text())["trace"]
        stop = trace[-1]
        scene = TexturedCaptureScene(ROOT / source["source_mesh"],
            floor_y=source["floor_y_1pct_m"],
            scale=source.get("scale_to_metres", 1.), scanned_person=human,
            person_facing_deg=cfg["person_facing_deg"])
        xy = np.asarray(stop["evaluator_actual_xz_m"], float)
        yaw = float(stop["evaluator_actual_heading_deg"])
        radians = math.radians(yaw)
        forward = np.array([math.sin(radians), math.cos(radians)])
        right = np.array([-math.cos(radians), math.sin(radians)])
        teach = xy + forward * .9
        teach_distractor = teach + right * .35
        search = xy + forward * .45 - right * .8
        search_distractor = xy + forward * .8 + right * .35
        positions = (teach, teach_distractor, search, search_distractor)
        item.update(scene_key=cfg_key,
            owner_endpoint_xz=xy.tolist(), owner_endpoint_yaw_deg=yaw,
            teach_target_xz=teach.tolist(),
            teach_distractor_xz=teach_distractor.tolist(),
            find_target_xz=search.tolist(),
            find_distractor_xz=search_distractor.tolist(),
            floor_supported=[scene.floor_supported(*x) for x in positions],
            search_yaw_offsets_deg=[0., 45., -45.])
        scene.set_props({"Parmesan": (target, tuple(teach), 0.),
                         "CreamCheese": (distractor, tuple(teach_distractor), 0.)})
        owner = owner_pose(float(stop["decision_time_s"]), cfg)
        teach_rgb, teach_ids = view(scene, xy, yaw, owner)
        ys, xs = np.where(teach_ids == "Parmesan")
        click = ([int(np.median(xs)), int(np.median(ys))] if len(xs) else None)
        teach_pixels = {n: int((teach_ids == n).sum())
                        for n in ("Parmesan", "CreamCheese")}
        scene.set_props({"Parmesan": (target, tuple(search), 0.),
                         "CreamCheese": (distractor, tuple(search_distractor), 0.)})
        search_rgb, search_ids = view(scene, xy, yaw, None)
        left_rgb, left_ids = view(scene, xy, yaw + 45., None)
        search_pixels = {"center_target": int((search_ids == "Parmesan").sum()),
            "center_distractor": int((search_ids == "CreamCheese").sum()),
            "left_target": int((left_ids == "Parmesan").sum())}
        item.update(click_xy_at_640x360=click, teach_visible_px=teach_pixels,
                    find_visible_px=search_pixels)
        item["status"] = ("provisional_supported_visible" if
            all(item["floor_supported"]) and
            teach_pixels["Parmesan"] >= 400 and
            teach_pixels["CreamCheese"] >= 150 and
            search_pixels["center_target"] < 50 and
            search_pixels["center_distractor"] >= 150 and
            search_pixels["left_target"] >= 300 else
            "placement_screen_failed")
        for column, (label, rgb) in enumerate((
            ("teach", teach_rgb), ("find center", search_rgb),
            ("find +45", left_rgb))):
            gallery.paste(Image.fromarray(rgb), (column * 640,
                                                  gallery_row * 390))
            draw.text((column * 640 + 8, gallery_row * 390 + 364),
                      f"{row['name']} | {label} | {item['status']}", fill="white")
        gallery_row += 1
        print(row["name"], item["status"], teach_pixels, search_pixels,
              flush=True)
        results.append(item)
    sheet = output / "placement_contact_sheet.jpg"
    gallery.save(sheet, quality=90)
    report = {"schema": "ape.real-scan-teach-find-placement-screen.v1",
        "source_owner_batch": str(args.batch.resolve().relative_to(ROOT)),
        "selection_rule": "floor support and actor-camera object visibility only; no detector or retrieval outcome",
        "owner_endpoint_source": "evaluator endpoint from earlier successful owner run, used only to freeze scenario geometry",
        "teaching_click_source": "evaluator-selected visible object pixel; scripted teaching input, not user interaction",
        "object_source": "textured HOPE grocery meshes at 1.2x their prior metric scale assumption",
        "photographic_room_qualification": "provisional",
        "supported_count": sum(r["status"] == "provisional_supported_visible" for r in results),
        "scenes": results,
        "contact_sheet": str(sheet.relative_to(ROOT))}
    (output / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    print(output / "screen.json")


if __name__ == "__main__":
    main()
