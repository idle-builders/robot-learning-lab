"""Freeze geometry-only teach/find placements at safely stopped delayed views.

The seven source owner runs are development episodes. This screen makes no
object-model calls and leaves all unsupported rooms in the record.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from real_capture_dynamic_scene import ScannedSittingPerson, TexturedCaptureScene, TexturedFloorObject
from run_real_capture_owner_loop import SCENES, owner_pose
from screen_scanned_teach_find_placements import view


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/teach_find_placement_screen"
PROPS = ROOT / ".cache/ape-high-level/hope_meshes_eval"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrated-owner-endpoints", action="store_true")
    parser.add_argument("--owner-batch", type=Path,
                        help="Screen every successful endpoint in a frozen owner batch")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-object", default="Parmesan")
    parser.add_argument("--distractor-object", default="CreamCheese")
    args = parser.parse_args()
    target_name, distractor_name = args.target_object, args.distractor_object
    if target_name == distractor_name or not all(
            (PROPS / f"{name}.obj").is_file() for name in
            (target_name, distractor_name)):
        parser.error("Choose two distinct available HOPE meshes")
    out = (args.output.resolve() if args.output else BASE / ("far_owner_11_placement_screen" if args.owner_batch
        else "integrated_placement_screen" if args.integrated_owner_endpoints
        else "delayed_conversation_placement_screen"))
    out.mkdir(parents=True, exist_ok=True)
    batch_path = (args.owner_batch.resolve() if args.owner_batch else
        (BASE / "delayed_conversation_placement_screen/integrated_batch_v1.json"
         if args.integrated_owner_endpoints else BASE / "batch_delay850ms_v1.json"))
    batch = json.loads(batch_path.read_text())
    human = ScannedSittingPerson(ROOT /
        "assets/real_indoor_scenes/scansceneverse/human_previews/sitting_person_isolated.glb")
    target = TexturedFloorObject(PROPS / f"{target_name}.obj", scale=.012)
    distractor = TexturedFloorObject(PROPS / f"{distractor_name}.obj", scale=.012)
    gallery = Image.new("RGB", (1920, 390 * len(batch["scenes"])), "#202020")
    draw = ImageDraw.Draw(gallery)
    rows = []
    gallery_row = 0
    for row in batch["scenes"]:
        trace_source = (row.get("trace", "") if args.owner_batch else
                        row.get("owner_video", "").replace("first_person.mp4", "trace.json")
                        if args.integrated_owner_endpoints else
                        row.get("summary", "").replace(
                            "summary.json", "owner_approach/trace.json"))
        key = next((k for k, v in SCENES.items() if v["name"] == row["name"]), None)
        if args.integrated_owner_endpoints and not args.owner_batch and not trace_source and key is not None:
            standalone = list(SCENES[key]["trace"].parent.glob(
                "dynamic_owner_*farconverse*episode/trace.json"))
            if len(standalone) == 1:
                trace_source = str(standalone[0].relative_to(ROOT))
        item = {"scene_id": row["scene_id"], "name": row["name"],
                "source_delayed_owner_trace": trace_source}
        if not trace_source or (not args.integrated_owner_endpoints and not args.owner_batch and
                row["task_status"] != "owner_viewpoint_not_reached"):
            item["status"] = "no_delayed_owner_stop"
            rows.append(item)
            continue
        if key is None:
            raise RuntimeError(f"No owner scene configuration for {row['name']}")
        config = SCENES[key]
        trace = json.loads((ROOT / item["source_delayed_owner_trace"]).read_text())
        if args.owner_batch and trace["outcome"] not in ("hold_viewpoint",
                                                        "hold_viewpoint_far"):
            item["status"] = "no_owner_viewpoint"
            item["owner_outcome"] = trace["outcome"]
            rows.append(item)
            continue
        stop = trace["trace"][-1]
        social = stop["owner_actor"]
        nearest = float(stop["planner_decision"].get("nearest_forward_range_m", 0.))
        expected_reason = ("far_conversation_viewpoint_near_range_stop" if
                           (args.integrated_owner_endpoints or args.owner_batch) else
                           "near_range_before_owner_standoff")
        far_eligible = (stop["planner_decision"].get("reason") ==
            expected_reason and stop["pose_healthy"] and
            1.45 <= float(social["local_depth_m"]) <= 1.65 and
            abs(float(social["owner_bearing_deg"])) <= 20. and
            nearest >= .9 and stop["commanded_step_m"] == 0.)
        close_eligible = bool(args.owner_batch and trace["outcome"] ==
            "hold_viewpoint" and trace.get("observed_standoff_band_met") and
            stop["pose_healthy"] and stop["commanded_step_m"] == 0.)
        eligible = far_eligible or close_eligible
        item.update(eligible_far_viewpoint=far_eligible,
            eligible_owner_viewpoint=eligible, owner_viewpoint_type=(
                "far" if far_eligible else "close" if close_eligible else None),
            owner_depth_m=social["local_depth_m"],
            nearest_forward_range_m=nearest,
            owner_bearing_deg=social["owner_bearing_deg"])
        if not eligible:
            item["status"] = "far_viewpoint_not_eligible"
            rows.append(item)
            continue
        source = json.loads(config["trace"].read_text())
        scene = TexturedCaptureScene(ROOT / source["source_mesh"],
            floor_y=source["floor_y_1pct_m"],
            scale=source.get("scale_to_metres", 1.), scanned_person=human,
            person_facing_deg=config["person_facing_deg"],
            source_z_up=source.get("source_z_up", False))
        xy = np.asarray(stop["evaluator_actual_xz_m"], float)
        yaw = float(stop["evaluator_actual_heading_deg"])
        radians = math.radians(yaw)
        forward = np.array([math.sin(radians), math.cos(radians)])
        right = np.array([-math.cos(radians), math.sin(radians)])
        teach = xy + forward * .9
        search = xy + forward * .45 - right * .8
        owner = owner_pose(float(stop["decision_time_s"] + .2), config)
        options = (.35, .25, -.35) if (args.integrated_owner_endpoints or args.owner_batch) else (.35,)
        attempts = []
        for side in options:
            candidate = teach + right * side
            scene.set_props({target_name: (target, tuple(teach), 0.),
                             distractor_name: (distractor, tuple(candidate), 0.)})
            candidate_rgb, candidate_ids = view(scene, xy, yaw, owner)
            pixels = {n: int((candidate_ids == n).sum())
                      for n in (target_name, distractor_name)}
            attempt = {"side_offset_m": side,
                       "floor_supported": scene.floor_supported(*candidate),
                       "visible_pixels": pixels}
            attempts.append(attempt)
            if (attempt["floor_supported"] and pixels[target_name] >= 400 and
                    pixels[distractor_name] >= 150):
                break
        teach_distractor = teach + right * side
        teach_rgb, teach_ids, teach_pixels = candidate_rgb, candidate_ids, pixels
        ys, xs = np.where(teach_ids == target_name)
        click = ([int(np.median(xs)), int(np.median(ys))] if len(xs) else None)
        search_options = ((.45, -.8), (.3, .4), (.3, .6)) if args.owner_batch else ((.45, -.8),)
        search_attempts = []
        distractor_options = ((.8, .35), (.7, .35)) if args.owner_batch else ((.8, .35),)
        found_search = False
        for distractor_forward, distractor_side in distractor_options:
            search_distractor = xy + forward * distractor_forward + right * distractor_side
            for search_forward, search_side in search_options:
                search = xy + forward * search_forward + right * search_side
                scene.set_props({target_name: (target, tuple(search), 0.),
                                 distractor_name: (distractor, tuple(search_distractor), 0.)})
                search_rgb, search_ids = view(scene, xy, yaw, None)
                left_rgb, left_ids = view(scene, xy, yaw + 45., None)
                right_rgb, right_ids = view(scene, xy, yaw - 45., None)
                search_pixels = {"center_target": int((search_ids == target_name).sum()),
                    "center_distractor": int((search_ids == distractor_name).sum()),
                    "left_target": int((left_ids == target_name).sum()),
                    "right_target": int((right_ids == target_name).sum())}
                attempt = {"forward_m": search_forward, "side_m": search_side,
                           "distractor_forward_side_m": [distractor_forward, distractor_side],
                           "floor_supported": scene.floor_supported(*search) and
                               scene.floor_supported(*search_distractor),
                           "visible_pixels": search_pixels}
                search_attempts.append(attempt)
                if (attempt["floor_supported"] and search_pixels["center_target"] < 50 and
                        search_pixels["center_distractor"] >= 150 and
                        max(search_pixels["left_target"], search_pixels["right_target"]) >= 300):
                    found_search = True
                    break
            if found_search:
                break
        positions = (teach, teach_distractor, search, search_distractor)
        item.update(scene_key=key, owner_endpoint_xz=xy.tolist(),
            owner_endpoint_yaw_deg=yaw, teach_target_xz=teach.tolist(),
            teach_distractor_xz=teach_distractor.tolist(),
            teach_distractor_offset_m=side,
            teach_distractor_geometry_attempts=attempts,
            find_target_xz=search.tolist(),
            find_target_forward_side_m=[search_forward, search_side],
            find_target_geometry_attempts=search_attempts,
            find_distractor_xz=search_distractor.tolist(),
            find_distractor_forward_side_m=[distractor_forward, distractor_side],
            floor_supported=[scene.floor_supported(*x) for x in positions],
            search_yaw_offsets_deg=[0., 45., -45.])
        item.update(click_xy_at_640x360=click, teach_visible_px=teach_pixels,
                    find_visible_px=search_pixels,
                    target_name=target_name, distractor_name=distractor_name,
                    object_scale_to_metres=.012)
        item["status"] = ("provisional_supported_visible" if
            all(item["floor_supported"]) and
            teach_pixels[target_name] >= 400 and
            teach_pixels[distractor_name] >= 150 and
            search_pixels["center_target"] < 50 and
            search_pixels["center_distractor"] >= 150 and
            max(search_pixels["left_target"], search_pixels["right_target"]) >= 300 else
            "placement_screen_failed")
        turn_deg, turn_rgb = ((45., left_rgb) if
            search_pixels["left_target"] >= search_pixels["right_target"] else
            (-45., right_rgb))
        for column, (label, rgb) in enumerate((
            ("teach", teach_rgb), ("find center", search_rgb),
            (f"find {turn_deg:+.0f}", turn_rgb))):
            gallery.paste(Image.fromarray(rgb), (column * 640,
                                                  gallery_row * 390))
            draw.text((column * 640 + 8, gallery_row * 390 + 364),
                      f"{row['name']} | {label} | {item['status']}", fill="white")
        gallery_row += 1
        rows.append(item)
        print(row["name"], item["status"], teach_pixels, search_pixels,
              flush=True)
    gallery = gallery.crop((0, 0, 1920, 390 * gallery_row))
    gallery.save(out / "placement_contact_sheet.jpg", quality=90)
    report = {"schema": "ape.delayed-owner-conversation-placement-screen.v1",
        "source_batch": str(batch_path.relative_to(ROOT)),
        "integrated_owner_endpoints": args.integrated_owner_endpoints,
        "owner_batch_endpoints": bool(args.owner_batch),
        "far_viewpoint_rule": "Stopped on near range; healthy pose; torso depth 1.45-1.65 m; bearing <=20 deg; nearest forward range >=0.9 m; no forward command",
        "selection_rule": "floor support and rendered visibility only; teaching distractor offsets +0.35, +0.25, -0.35 m; search distractor (forward,side) offsets (+0.8,+0.35), then (+0.7,+0.35) m where owner-batch mode permits; for each, search target (+0.45,-0.8), (+0.3,+0.4), (+0.3,+0.6) m in that order; no object-model outcome",
        "target": f"HOPE {target_name}", "distractor": f"HOPE {distractor_name}",
        "scene_count": len(rows),
        "eligible_count": sum(x.get("eligible_owner_viewpoint", False) for x in rows),
        "supported_count": sum(x["status"] == "provisional_supported_visible" for x in rows),
        "scenes": rows}
    (out / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    print("eligible", report["eligible_count"], "supported", report["supported_count"])


if __name__ == "__main__":
    main()
