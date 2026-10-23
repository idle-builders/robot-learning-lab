"""Counterfactual floor-object placement sweep for ideal two-height range.

This is evaluator-only sensor coverage, not an actor policy or held-out task.
It perturbs each frozen find placement while retaining visible objects and
floor support, then asks whether the same 5-degree range fan samples the can.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from real_capture_dynamic_scene import TexturedCaptureScene, TexturedFloorObject
from run_real_capture_owner_loop import SCENES


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/teach_find_placement_screen"
SCREEN = BASE / "screen.json"
OUT = BASE / "range_offset_sweep_v1.json"
PROPS = ROOT / ".cache/ape-high-level/hope_meshes_eval"
LATERAL_OFFSETS_M = (-.1, -.05, 0., .05, .1)
FORWARD_OFFSETS_M = (-.05, 0., .05)
SENSOR_HEIGHTS_M = (.05, .1, .15, .2, .5)
ANGULAR_STEPS_DEG = (5, 10, 15, 20)


def main() -> None:
    placements = json.loads(SCREEN.read_text())["scenes"]
    target = TexturedFloorObject(PROPS / "Parmesan.obj", scale=.012)
    distractor = TexturedFloorObject(PROPS / "CreamCheese.obj", scale=.012)
    previews = BASE / "range_offset_previews"
    previews.mkdir(exist_ok=True)
    rooms = []
    for placement in placements:
        if placement["status"] != "provisional_supported_visible":
            continue
        cfg = SCENES[placement["scene_key"]]
        source = json.loads(cfg["trace"].read_text())
        scene = TexturedCaptureScene(ROOT / source["source_mesh"],
            floor_y=source["floor_y_1pct_m"],
            scale=source.get("scale_to_metres", 1.))
        xy = np.asarray(placement["owner_endpoint_xz"], float)
        yaw = float(placement["owner_endpoint_yaw_deg"] + 45.)
        rad = math.radians(placement["owner_endpoint_yaw_deg"])
        forward = np.array([math.sin(rad), math.cos(rad)])
        right = np.array([-math.cos(rad), math.sin(rad)])
        eye = np.array([xy[0], scene.floor_y + .42, xy[1]])
        view_forward = np.array([math.sin(math.radians(yaw)), 0.,
                                 math.cos(math.radians(yaw))])
        angles = np.arange(-90., 91., 1., dtype=np.float32)
        other = tuple(placement["find_distractor_xz"])
        scene.set_props({"CreamCheese": (distractor, other, 0.)})
        without = scene.range_fan(eye, yaw, owner_xz=None,
                                  angles_deg=angles,
                                  heights_m=SENSOR_HEIGHTS_M)
        samples = []
        example_miss = None
        for forward_offset in FORWARD_OFFSETS_M:
            for lateral_offset in LATERAL_OFFSETS_M:
                xz = (np.asarray(placement["find_target_xz"]) +
                      forward * forward_offset + right * lateral_offset)
                floor = scene.floor_supported(*xz)
                scene.set_props({"CreamCheese": (distractor, other, 0.),
                                 "Parmesan": (target, tuple(xz), 0.)})
                rgb, _, ids = scene.observe(eye, view_forward, owner_xz=None,
                                            return_prop_ids=True)
                y, x = np.where(ids == "Parmesan")
                visible = int(len(x))
                if len(x):
                    center_u = float(np.median(x))
                    focal = 360 / (2 * math.tan(math.radians(75.) / 2))
                    bearing = math.degrees(math.atan((center_u - 320) / focal))
                else:
                    bearing = None
                evaluated = bool(floor and visible >= 300)
                near_low = near_mid = closest_low = closest_mid = False
                sensor_hit = {}
                if evaluated:
                    with_target = scene.range_fan(eye, yaw, owner_xz=None,
                        angles_deg=angles, heights_m=SENSOR_HEIGHTS_M)
                    hit = with_target + .01 < without
                    offset = -bearing
                    five = np.where(angles.astype(int) % 5 == 0)[0]
                    nearest = int(five[np.argmin(abs(angles[five] - offset))])
                    neighborhood = (abs(angles - offset) <= 5.01) & \
                        (angles.astype(int) % 5 == 0)
                    near_low = bool(hit[1, neighborhood].any())
                    near_mid = bool(hit[4, neighborhood].any())
                    closest_low = bool(hit[1, nearest])
                    closest_mid = bool(hit[4, nearest])
                    for level, height in enumerate(SENSOR_HEIGHTS_M):
                        for step in ANGULAR_STEPS_DEG:
                            indices = np.where(angles.astype(int) % step == 0)[0]
                            index = int(indices[np.argmin(abs(
                                angles[indices] - offset))])
                            sensor_hit[f"h{height:.2f}_step{step}"] = bool(
                                hit[level, index])
                    if not near_low and example_miss is None:
                        path = previews / f"{placement['scene_id'].split('/')[-1]}_low_miss.jpg"
                        im = Image.fromarray(rgb)
                        draw = ImageDraw.Draw(im)
                        draw.text((8, 8), "Visible can, ideal low range missed",
                                  fill="white", stroke_width=2,
                                  stroke_fill="black")
                        im.save(path, quality=92)
                        example_miss = str(path.relative_to(ROOT))
                samples.append({"forward_offset_m": forward_offset,
                    "lateral_offset_m": lateral_offset,
                    "position_xz": xz.tolist(),
                    "floor_supported": floor,
                    "target_visible_px": visible,
                    "visual_bearing_deg": bearing,
                    "evaluated": evaluated,
                    "nearest_low_hit_target": closest_low,
                    "nearest_mid_hit_target": closest_mid,
                    "within_5deg_low_hit_target": near_low,
                    "within_5deg_mid_hit_target": near_mid,
                    "sensor_hit_matrix": sensor_hit})
        eligible = [r for r in samples if r["evaluated"]]
        room = {"scene_id": placement["scene_id"], "name": placement["name"],
            "evaluated_count": len(eligible),
            "low_nearest_hit_count": sum(r["nearest_low_hit_target"] for r in eligible),
            "low_within_5deg_hit_count": sum(r[
                "within_5deg_low_hit_target"] for r in eligible),
            "mid_within_5deg_hit_count": sum(r[
                "within_5deg_mid_hit_target"] for r in eligible),
            "example_low_miss": example_miss,
            "sensor_hit_matrix": {f"h{height:.2f}_step{step}": sum(
                r["sensor_hit_matrix"].get(f"h{height:.2f}_step{step}", False)
                for r in eligible)
                for height in SENSOR_HEIGHTS_M for step in ANGULAR_STEPS_DEG},
            "samples": samples}
        rooms.append(room)
        print(placement["name"], room["low_within_5deg_hit_count"],
              "/", len(eligible), flush=True)
    report = {"schema": "ape.real-scan-object-range-offset-sweep.v1",
        "placement_screen": str(SCREEN.relative_to(ROOT)),
        "evaluator_only": True,
        "candidate_offsets_m": {"forward": FORWARD_OFFSETS_M,
                                "lateral": LATERAL_OFFSETS_M},
        "sensor_heights_m": SENSOR_HEIGHTS_M,
        "angular_steps_deg": ANGULAR_STEPS_DEG,
        "evaluated_rule": "reconstructed floor support and >=300 visible target pixels at 640x360",
        "sample_count": sum(r["evaluated_count"] for r in rooms),
        "low_nearest_hit_count": sum(r["low_nearest_hit_count"] for r in rooms),
        "low_within_5deg_hit_count": sum(r["low_within_5deg_hit_count"] for r in rooms),
        "mid_within_5deg_hit_count": sum(r["mid_within_5deg_hit_count"] for r in rooms),
        "sensor_hit_matrix": {f"h{height:.2f}_step{step}": sum(
            room["sensor_hit_matrix"][f"h{height:.2f}_step{step}"]
            for room in rooms)
            for height in SENSOR_HEIGHTS_M for step in ANGULAR_STEPS_DEG},
        "rooms": rooms}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
