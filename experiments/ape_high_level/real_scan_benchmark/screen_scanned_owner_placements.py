"""Place a scanned owner in each captured ScanSceneverse room before task runs.

Placement is evaluator-side scenario construction, not a perception result. It
requires support and clearance in the static scan and a visible person in the
initial low camera view. The owner-loop actor later receives only the image,
depth, range and a click derived from the frozen placement.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from real_capture_dynamic_scene import ScannedSittingPerson, TexturedCaptureScene


ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / "assets/real_indoor_scenes/real_capture_21_working_set.json"
OUT = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen"
HUMAN = ROOT / "assets/real_indoor_scenes/scansceneverse/human_previews/sitting_person_isolated.glb"
ANGLES = np.arange(-180., 180., 15., np.float32)


def visible_mask(depth: np.ndarray, background: np.ndarray) -> np.ndarray:
    return np.isfinite(depth) & (
        ~np.isfinite(background) | (depth + .025 < background))


def screen_room(row: dict, person: ScannedSittingPerson,
                output_dir: Path = OUT, centered_owner: bool = False) -> dict:
    source = json.loads((ROOT / row["trace"]).read_text())
    frame = source["frames"][0]
    eye = np.asarray(frame["eye_m"], float)
    direction = np.asarray(frame["forward"], float)
    direction /= np.linalg.norm(direction)
    heading = math.degrees(math.atan2(direction[0], direction[2])) % 360
    scene = TexturedCaptureScene(ROOT / source["source_mesh"],
                                 floor_y=source["floor_y_1pct_m"],
                                 scale=source.get("scale_to_metres", 1.),
                                 scanned_person=person,
                                 person_facing_deg=(heading + 180) % 360)
    # Render the background once. Its depth is only for scenario placement;
    # it is never supplied separately to the owner actor.
    _, background = scene.observe(eye, direction, owner_xz=None,
                                  width=320, height=180)
    lateral = np.array([direction[2], -direction[0]])
    candidates = []
    # Require room for an actual approach decision; a 1.4 m placement lets
    # the actor stop almost immediately and would inflate benchmark coverage.
    for distance in (2.4, 2.2, 2.0, 1.8):
        for offset in ((0.,) if centered_owner else (0., -.35, .35, -.7, .7)):
            xz = eye[[0, 2]] + distance * direction[[0, 2]] + offset * lateral
            if not scene.floor_supported(*xz):
                continue
            ranges = scene.range_fan(np.array([xz[0], eye[1], xz[1]]),
                                     heading, owner_xz=None, angles_deg=ANGLES)
            clearance = float(np.nanmin(ranges))
            if clearance < .30:
                continue
            # The scripted person's small lateral motion must remain on the
            # captured floor and away from static walls/furniture too.
            moving_supported = True
            for shift in (-.22, .22):
                moved = xz + shift * lateral
                if not scene.floor_supported(*moved):
                    moving_supported = False
                    break
                moved_ranges = scene.range_fan(
                    np.array([moved[0], eye[1], moved[1]]), heading,
                    owner_xz=None, angles_deg=ANGLES)
                if float(np.nanmin(moved_ranges)) < .30:
                    moving_supported = False
                    break
            if not moving_supported:
                continue
            rgb, depth = scene.observe(eye, direction, owner_xz=tuple(xz),
                                       width=320, height=180)
            mask = visible_mask(depth, background)
            yy, xx = np.where(mask)
            if len(xx) < 500:
                continue
            center = np.array([float(np.median(xx)), float(np.median(yy))])
            # The click must land on the visible figure, away from the border.
            if not (35 <= center[0] <= 285 and 25 <= center[1] <= 155):
                continue
            score = (-abs(distance - 2.1) * 2 +
                     min(len(xx), 3000) / 3000 * .25 +
                     min(clearance, .8) / .8 * .15 -
                     abs(offset) * .2 -
                     abs(center[0] - 160) / 320)
            candidates.append((score, distance, offset, xz, clearance,
                               center, rgb, len(xx)))
    if not candidates:
        return {"scene_id": row["scene_id"], "name": row["name"],
                "status": "no_supported_visible_owner_placement",
                "source_trace": row["trace"]}
    score, distance, offset, xz, clearance, center, rgb, pixels = max(
        candidates, key=lambda p: p[0])
    click = np.rint(center * 2).astype(int)
    large = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_LINEAR)
    cv2.drawMarker(large, tuple(click), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
    image = output_dir / f"{row['scene_id'].split('/')[-1]}_owner_start.jpg"
    Image.fromarray(large).save(image, quality=90)
    return {"scene_id": row["scene_id"], "name": row["name"],
            "status": "provisional_supported_visible",
            "source_trace": row["trace"],
            "start_xz": eye[[0, 2]].tolist(), "start_heading_deg": heading,
            "owner_xz": xz.tolist(), "owner_facing_deg": (heading + 180) % 360,
            "owner_lateral_xz": lateral.tolist(),
            "owner_distance_m": distance, "owner_lateral_offset_m": offset,
            "owner_position_static_clearance_m": clearance,
            "visible_person_pixels_at_320x180": pixels,
            "click_xy_at_640x360": click.tolist(),
            "score": score, "initial_view": str(image.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--scene-id", action="append", default=[],
                        help="Screen only these indexed captures")
    parser.add_argument("--centered-owner", action="store_true",
                        help="Development screen with owner on the initial heading axis")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    person = ScannedSittingPerson(HUMAN)
    rows = json.loads(args.index.read_text())["scenes"]
    rows = [r for r in rows if r["source_family"] == "scansceneverse" and
            (not args.scene_id or r["scene_id"] in args.scene_id)]
    if args.scene_id and len(rows) != len(set(args.scene_id)):
        raise ValueError("Every requested scene ID must appear once in the index")
    results = []
    for row in rows:
        print("screen", row["name"], flush=True)
        results.append(screen_room(row, person, output,
                                   centered_owner=args.centered_owner))
        gc.collect()
    report = {"schema": "ape.scanned-owner-placement-screen.v1",
              "source_index": str(args.index.resolve().relative_to(ROOT)),
              "purpose": "evaluator-side placement for frozen initial owner-click tasks; not a task outcome",
              "body": "0.42 m camera and 0.17 m radius kinematic proxy; no selected articulated body",
              "centered_owner_only": args.centered_owner,
              "person": "rigidly moved CC-BY sitting-person photogrammetry scan",
              "placement_gate": ">=1.8 m initial owner distance, static floor support and >=0.30 m 2-height static range at center and +/-0.22 m lateral motion, >=500 owner-only visible pixels and central click",
              "scene_count": len(results),
              "supported_visible_count": sum(r["status"] == "provisional_supported_visible" for r in results),
              "scenes": results}
    (output / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["supported_visible_count"], "/", report["scene_count"],
          "provisional placements")


if __name__ == "__main__":
    main()
