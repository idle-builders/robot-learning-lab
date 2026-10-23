"""Freeze an owner start in an independently sourced textured indoor scan.

The evaluator searches only the geometry candidates frozen before actor work.
Owner masks, floor checks and scene truth never enter the later actor loop.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from real_capture_dynamic_scene import ScannedSittingPerson, TexturedCaptureScene


ROOT = Path(__file__).resolve().parents[3]
HUMAN = ROOT / "assets/real_indoor_scenes/scansceneverse/human_previews/sitting_person_isolated.glb"


def screen(source_path: Path, output: Path, *, minimum_valid_depth: float = .70,
           precommit: str = "artifacts/ape-real-room-next-captures-20261014/skokloster_owner_screen_precommit.json") -> dict:
    source = json.loads(source_path.read_text())
    geometry = json.loads((ROOT / source["geometry_screen"]).read_text())
    # Some small or cluttered captures legitimately produce fewer than 40
    # geometry-supported starts. Screen every available candidate, including
    # the empty set, rather than treating the search cap as a requirement.
    person = ScannedSittingPerson(HUMAN)
    scene = TexturedCaptureScene(
        ROOT / source["source_mesh"], floor_y=source["floor_y_1pct_m"],
        scale=source["scale_to_metres"], source_z_up=source["source_z_up"],
        scanned_person=person)
    rows = []
    eligible = []
    for index, candidate in enumerate(geometry["candidates"]):
        heading = float(candidate["heading_deg"])
        yaw = math.radians(heading)
        forward = np.array([math.sin(yaw), 0., math.cos(yaw)])
        eye = np.array([*candidate["start_xz"][:1],
                        source["floor_y_1pct_m"] + .42,
                        candidate["start_xz"][1]])
        owner = tuple(candidate["owner_xz"])
        static_rgb, static_depth = scene.observe(
            eye, forward, owner_xz=None, width=640, height=360)
        scene.person_facing_deg = (heading + 180.) % 360.
        rgb, depth = scene.observe(eye, forward, owner_xz=owner,
                                   width=640, height=360)
        black = float(np.mean(np.max(static_rgb, axis=-1) < 8))
        depth_valid = float(np.mean(np.isfinite(static_depth)))
        mask = np.isfinite(depth) & (
            ~np.isfinite(static_depth) | (depth + .025 < static_depth))
        yy, xx = np.where(mask)
        pixels = int(len(xx))
        click = ([int(np.median(xx)), int(np.median(yy))] if pixels else None)
        status = ("supported_visible" if black <= .05 and
                  depth_valid >= minimum_valid_depth and
                  pixels >= 1200 and click is not None and
                  70 <= click[0] <= 570 and 40 <= click[1] <= 320
                  else "failed_appearance_or_owner_visibility")
        row = {"candidate_index": index, "status": status,
               "static_near_black_fraction": black,
               "static_valid_depth_fraction": depth_valid,
               "visible_person_pixels": pixels, "click_xy_at_640x360": click,
               "geometry_candidate": candidate}
        rows.append(row)
        if status == "supported_visible":
            eligible.append((pixels, -index, row, static_rgb, rgb))
    output.mkdir(parents=True, exist_ok=True)
    selected = max(eligible, key=lambda value: (value[0], value[1])) if eligible else None
    if selected is None:
        result = {"status": "no_supported_visible_owner_placement"}
    else:
        _, _, chosen, static_rgb, rgb = selected
        candidate = chosen["geometry_candidate"]
        heading = float(candidate["heading_deg"])
        lateral = [math.cos(math.radians(heading)),
                   -math.sin(math.radians(heading))]
        static_path = output / "initial_static.jpg"
        owner_path = output / "owner_start.jpg"
        Image.fromarray(static_rgb).save(static_path, quality=91)
        marked = Image.fromarray(rgb)
        draw = ImageDraw.Draw(marked)
        click = chosen["click_xy_at_640x360"]
        draw.line((click[0]-8,click[1],click[0]+8,click[1]),fill=(0,255,0),width=2)
        draw.line((click[0],click[1]-8,click[0],click[1]+8),fill=(0,255,0),width=2)
        marked.save(owner_path, quality=91)
        result = {"status": "provisional_supported_visible",
                  "start_xz": candidate["start_xz"],
                  "start_heading_deg": heading,
                  "owner_xz": candidate["owner_xz"],
                  "owner_facing_deg": (heading + 180.) % 360.,
                  "owner_lateral_xz": lateral,
                  "owner_distance_m": candidate["owner_distance_m"],
                  "owner_position_static_clearance_m":
                      candidate["minimum_static_clearance_m"],
                  "visible_person_pixels_at_sensor_resolution":
                      chosen["visible_person_pixels"],
                  "click_xy_at_640x360": click,
                  "selected_candidate_index": chosen["candidate_index"],
                  "initial_static_view": str(static_path.relative_to(ROOT)),
                  "initial_view": str(owner_path.relative_to(ROOT))}
    report = {"schema": "ape.textured-capture-owner-screen.v1",
              "scene_id": source["scene_id"], "name": source["name"],
              "source_trace": str(source_path.relative_to(ROOT)),
              "precommit": precommit,
              "minimum_valid_static_depth_fraction": minimum_valid_depth,
              "screen_resolution_px": [640, 360],
              "candidate_count": len(rows),
              "eligible_count": len(eligible),
              "candidate_results": rows,
              **result}
    (output / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--minimum-valid-depth", type=float, default=.70)
    p.add_argument("--precommit", default=
                   "artifacts/ape-real-room-next-captures-20261014/skokloster_owner_screen_precommit.json")
    args = p.parse_args()
    result = screen(args.source.resolve(), args.output.resolve(),
                    minimum_valid_depth=args.minimum_valid_depth,
                    precommit=args.precommit)
    print(result["status"], result["eligible_count"], "/", result["candidate_count"])


if __name__ == "__main__":
    main()
