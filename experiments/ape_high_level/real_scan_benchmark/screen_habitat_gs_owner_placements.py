"""Freeze owner placement in a real Gaussian-capture room without actor outcomes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from real_capture_dynamic_scene import GaussianCaptureScene, ScannedSittingPerson
from screen_scanned_owner_placements import visible_mask


ROOT = Path(__file__).resolve().parents[3]
SOURCES = ROOT / "assets/real_indoor_scenes/habitat_gs/owner_source_index_20261014.json"
OUT = ROOT / "assets/real_indoor_scenes/habitat_gs/owner_placement_screen"
HUMAN = ROOT / "assets/real_indoor_scenes/scansceneverse/human_previews/sitting_person_isolated.glb"
ANGLES = np.arange(-180., 180., 15., np.float32)
DISTANCES = (3.4, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2)
OFFSETS = (0., -.35, .35)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recorded-view", action="store_true",
                        help="Use saved clean capture heading and reject black splat coverage")
    parser.add_argument("--full-sensor-screen", action="store_true",
                        help="Render placement at the actual 640x360 owner sensor size")
    args = parser.parse_args()
    if args.full_sensor_screen and not args.recorded_view:
        parser.error("full sensor screen requires recorded view")
    index = json.loads(SOURCES.read_text())
    source_row = next((r for r in index["scenes"] if r["scene_id"] ==
                       "habitat_gs/" + args.scene_id), None)
    if source_row is None:
        raise ValueError("Unknown source scene")
    version = "v3" if args.full_sensor_screen else "v2" if args.recorded_view else "v1"
    output = (args.output or OUT / version /
              args.scene_id).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((ROOT / source_row["source_capture"]).read_text())
    start = np.asarray(source["start_xz"], float)
    target = np.asarray(source["route_target_xz"], float)
    forward = target - start
    forward /= np.linalg.norm(forward)
    lateral = np.array([forward[1], -forward[0]])
    heading = math.degrees(math.atan2(forward[0], forward[1])) % 360.
    view_heading = heading
    if args.recorded_view:
        replay = json.loads((ROOT / source["source_probe"]).parent.joinpath(
            "sensor_replay/sensor_trace.json").read_text())
        view_heading = math.degrees(-float(replay["frames"][0]["heading_rad"])) % 360.
    view_forward = np.array([math.sin(math.radians(view_heading)),
                             math.cos(math.radians(view_heading))])
    eye = np.array([start[0], source["floor_y_1pct_m"] + .42, start[1]])
    scene = GaussianCaptureScene(
        ROOT / source["source_mesh"], ROOT / source["source_splat"],
        floor_y=source["floor_y_1pct_m"],
        scanned_person=ScannedSittingPerson(HUMAN),
        person_facing_deg=(heading + 180.) % 360.)
    width, height = (640, 360) if args.full_sensor_screen else (320, 180)
    background_rgb, background_depth = scene.observe(
        eye, np.r_[view_forward[0], 0., view_forward[1]], owner_xz=None,
        width=width, height=height)
    Image.fromarray(background_rgb).save(output / "initial_static.jpg", quality=90)
    black_fraction = float(np.mean(np.max(background_rgb, axis=-1) < 8))
    attempts = []
    supported = []
    for distance in DISTANCES:
        for offset in OFFSETS:
            xz = start + distance * forward + offset * lateral
            attempt = {"distance_m": distance, "offset_m": offset,
                       "owner_xz": xz.tolist()}
            stations = [start + alpha * (xz - start)
                        for alpha in np.linspace(0., 1., 11)]
            attempt["path_floor_supported"] = all(
                scene.floor_supported(*station) for station in stations)
            attempt["owner_floor_supported"] = scene.floor_supported(*xz)
            if not (attempt["path_floor_supported"] and
                    attempt["owner_floor_supported"]):
                attempt["status"] = "floor_unsupported"
                attempts.append(attempt)
                continue
            endpoint_clearances = []
            for shift in (-.22, 0., .22):
                point = xz + shift * lateral
                if not scene.floor_supported(*point):
                    endpoint_clearances.append(0.)
                    continue
                origin = np.array([point[0], eye[1], point[1]])
                ranges = scene.range_fan(origin, heading, owner_xz=None,
                                         angles_deg=ANGLES)
                endpoint_clearances.append(float(np.nanmin(ranges)))
            clearance = min(endpoint_clearances)
            attempt["minimum_owner_motion_clearance_m"] = clearance
            if clearance < .30:
                attempt["status"] = "owner_motion_clearance"
                attempts.append(attempt)
                continue
            rgb, depth = scene.observe(
                eye, np.r_[view_forward[0], 0., view_forward[1]],
                owner_xz=tuple(xz), width=width, height=height)
            yy, xx = np.where(visible_mask(depth, background_depth))
            center = ([float(np.median(xx)), float(np.median(yy))]
                      if len(xx) else None)
            attempt.update(visible_person_pixels=len(xx),
                           click_center_xy=center)
            scale = 2 if args.full_sensor_screen else 1
            if len(xx) < 300 * scale ** 2 or center is None or not (
                    35 * scale <= center[0] <= 285 * scale and
                    25 * scale <= center[1] <= 155 * scale):
                attempt["status"] = "person_not_sufficiently_visible"
                attempts.append(attempt)
                continue
            if args.recorded_view and black_fraction > .05:
                attempt["status"] = "black_splat_coverage"
                attempts.append(attempt)
                continue
            score = (-abs(distance - 3.) * 2. + min(len(xx), 3000 * scale ** 2) /
                     (3000 * scale ** 2) * .25 +
                     min(clearance, .8) / .8 * .15 - abs(offset) * .2 -
                     abs(center[0] - width / 2) / width)
            attempt.update(status="provisional_supported_visible", score=score)
            attempts.append(attempt)
            supported.append((score, attempt, rgb))
    result = {"schema": "ape.habitat-gs-owner-placement-screen.v1",
              "source_trace": source_row["source_capture"],
              "precommit": ("artifacts/ape-real-room-next-captures-20261014/"
                            "habitat_gs_owner_screen_v3_precommit.json" if args.full_sensor_screen
                            else "artifacts/ape-real-room-next-captures-20261014/"
                            "habitat_gs_owner_screen_v2_precommit.json" if args.recorded_view
                            else "artifacts/ape-real-room-next-captures-20261014/"
                            "habitat_gs_owner_screen_precommit.json"),
              "scene_id": source_row["scene_id"],
              "name": f"Habitat-GS {args.scene_id}",
              "status": "no_supported_visible_owner_placement",
              "attempts": attempts,
              "screen_resolution_px": [width, height],
              "static_near_black_fraction": black_fraction,
              "camera_heading_source": "recorded_clean_capture" if args.recorded_view
              else "direct_route_target",
              "initial_static_view": str((output / "initial_static.jpg").relative_to(ROOT))}
    if supported:
        _, winner, rgb = max(supported, key=lambda row: row[0])
        click = np.rint(np.asarray(winner["click_center_xy"]) *
                        (1 if args.full_sensor_screen else 2)).astype(int)
        large = (rgb if args.full_sensor_screen else
                 cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_LINEAR))
        cv2.drawMarker(large, tuple(click), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
        image = output / "owner_start.jpg"
        Image.fromarray(large).save(image, quality=90)
        result.update({"status": "provisional_supported_visible",
                       "start_xz": start.tolist(), "start_heading_deg": view_heading,
                       "owner_xz": winner["owner_xz"],
                       "owner_facing_deg": (heading + 180.) % 360.,
                       "owner_lateral_xz": lateral.tolist(),
                       "owner_distance_m": winner["distance_m"],
                       "owner_lateral_offset_m": winner["offset_m"],
                       "owner_position_static_clearance_m": winner[
                           "minimum_owner_motion_clearance_m"],
                       "visible_person_pixels_at_sensor_resolution": winner[
                           "visible_person_pixels"],
                       "click_xy_at_640x360": click.tolist(),
                       "score": winner["score"],
                       "initial_view": str(image.relative_to(ROOT))})
    (output / "screen.json").write_text(json.dumps(result, indent=2) + "\n")
    print(args.scene_id, result["status"],
          result.get("owner_distance_m"),
          result.get("visible_person_pixels_at_sensor_resolution"), flush=True)


if __name__ == "__main__":
    main()
