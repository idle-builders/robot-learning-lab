"""Freeze new taught-object appearances at already retained owner stops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from real_capture_dynamic_scene import ScannedSittingPerson, TexturedCaptureScene, TexturedFloorObject
from run_real_capture_owner_loop import SCENES, owner_pose
from screen_scanned_teach_find_placements import view


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "artifacts/ape-real-room-next-captures-20261014/common_policy_15_group_report.json"
PRECOMMIT = "artifacts/ape-real-room-next-captures-20261014/common_policy_object_variation_precommit.json"
OUTPUT = ROOT / "assets/real_indoor_scenes/object_variation"
PROPS = ROOT / ".cache/ape-high-level/hope_meshes_eval"
PAIRS = (("Ketchup", "Mustard"), ("Yogurt", "Corn"))
HOLDOUT_PAIRS = (("BBQSauce", "SaladDressing"), ("Peaches", "Pineapple"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--pair-set", choices=("original", "holdout"),
                        default="original")
    args = parser.parse_args()
    args.output = args.output.resolve()
    pairs = HOLDOUT_PAIRS if args.pair_set == "holdout" else PAIRS
    if args.width < 320 or args.height < 180 or args.width * 9 != args.height * 16:
        raise ValueError("Camera must be at least 320x180 and 16:9")
    batch = json.loads(REPORT.read_text())
    placements = {}
    for row in batch["rooms"]:
        source = row["source_screen"]
        if source not in placements:
            placements[source] = json.loads((ROOT / source).read_text())["scenes"]
    human = ScannedSittingPerson(ROOT /
        "assets/real_indoor_scenes/scansceneverse/human_previews/sitting_person_isolated.glb")
    results = {pair: [] for pair in pairs}
    thumbnails = {pair: [] for pair in pairs}
    for row in batch["rooms"]:
        if row["placement_status"] != "provisional_supported_visible":
            continue
        base = next(p for p in placements[row["source_screen"]]
                    if p["scene_id"] == row["scene_id"])
        if row.get("task_status") != "teach_find_completed_far_viewpoint":
            for pair in pairs:
                results[pair].append({"scene_id": row["scene_id"],
                    "capture_group": row["capture_group"], "name": row["name"],
                    "scene_key": base["scene_key"], "status": "no_owner_viewpoint",
                    "source_owner_status": row["task_status"]})
            continue
        cfg = SCENES[base["scene_key"]]
        source = json.loads(cfg["trace"].read_text())
        owner_trace = (ROOT / row["summary"]).parent / "owner_approach/trace.json"
        stop = json.loads(owner_trace.read_text())["trace"][-1]
        xy = np.asarray(stop["evaluator_actual_xz_m"], float)
        yaw = float(stop["evaluator_actual_heading_deg"])
        shift = xy - np.asarray(base["owner_endpoint_xz"], float)
        positions = {key: tuple(np.asarray(base[key], float) + shift) for key in (
            "teach_target_xz", "teach_distractor_xz",
            "find_target_xz", "find_distractor_xz")}
        scene = TexturedCaptureScene(ROOT / source["source_mesh"],
            floor_y=source["floor_y_1pct_m"],
            scale=source.get("scale_to_metres", 1.),
            source_z_up=source.get("source_z_up", False),
            scanned_person=human, person_facing_deg=cfg["person_facing_deg"])
        floor = {key: bool(scene.floor_supported(*point)) for key, point in positions.items()}
        owner = owner_pose(float(stop["decision_time_s"] + .2), cfg)
        for pair in pairs:
            target_name, distractor_name = pair
            target = TexturedFloorObject(PROPS / f"{target_name}.obj", scale=.012)
            distractor = TexturedFloorObject(PROPS / f"{distractor_name}.obj", scale=.012)
            scene.set_props({target_name: (target, positions["teach_target_xz"], 0.),
                             distractor_name: (distractor, positions["teach_distractor_xz"], 0.)})
            teach_rgb, teach_ids = view(scene, xy, yaw, owner,
                                       width=args.width, height=args.height)
            ys, xs = np.where(teach_ids == target_name)
            click = ([int(np.median(xs)), int(np.median(ys))] if len(xs) else None)
            teach_counts = {"target": int((teach_ids == target_name).sum()),
                            "distractor": int((teach_ids == distractor_name).sum())}
            scene.set_props({target_name: (target, positions["find_target_xz"], 0.),
                             distractor_name: (distractor, positions["find_distractor_xz"], 0.)})
            views = [view(scene, xy, yaw + offset, None,
                          width=args.width, height=args.height)
                     for offset in (0., 45., -45.)]
            search_counts = [{"yaw_offset_deg": offset,
                "target": int((ids == target_name).sum()),
                "distractor": int((ids == distractor_name).sum())}
                for offset, (_, ids) in zip((0., 45., -45.), views)]
            supported = (all(floor.values()) and click is not None and
                teach_counts["target"] >= 400 and teach_counts["distractor"] >= 150 and
                search_counts[0]["target"] < 50 * args.width * args.height / (640 * 360) and
                search_counts[0]["distractor"] >= 150 and
                max(c["target"] for c in search_counts[1:]) >= 300)
            result = {"scene_id": row["scene_id"], "capture_group": row["capture_group"],
                "name": row["name"], "scene_key": base["scene_key"],
                "status": "provisional_supported_visible" if supported else "placement_screen_failed",
                "source_delayed_owner_trace": str(owner_trace.relative_to(ROOT)),
                "eligible_far_viewpoint": True,
                "owner_endpoint_xz": xy.tolist(), "owner_endpoint_yaw_deg": yaw,
                **{key: list(point) for key, point in positions.items()},
                "source_endpoint_shift_m": shift.tolist(),
                "floor_supported": floor,
                "click_xy_at_640x360" if (args.width, args.height) == (640, 360)
                    else "click_xy_at_camera_resolution": click,
                "teach_visible_px": teach_counts, "find_visible_px": search_counts,
                "target_name": target_name, "distractor_name": distractor_name,
                "object_scale_to_metres": .012}
            results[pair].append(result)
            turned = max(range(1, 3), key=lambda i: search_counts[i]["target"])
            rgb = [teach_rgb, views[0][0], views[turned][0]]
            thumb = np.concatenate([np.asarray(Image.fromarray(x).resize((320, 180)))
                                    for x in rgb], axis=1)
            thumbnails[pair].append((row["name"], result["status"], thumb))
            print(target_name, row["name"], result["status"],
                  teach_counts, search_counts, flush=True)
    for pair in pairs:
        target_name, distractor_name = pair
        folder = args.output / f"{target_name.lower()}_vs_{distractor_name.lower()}"
        folder.mkdir(parents=True, exist_ok=True)
        sheet = Image.new("RGB", (960, 210 * len(thumbnails[pair])), "#202020")
        draw = ImageDraw.Draw(sheet)
        for index, (name, status, thumb) in enumerate(thumbnails[pair]):
            sheet.paste(Image.fromarray(thumb), (0, index * 210))
            draw.text((8, index * 210 + 184), f"{name} | {status}", fill="white")
        sheet_path = folder / "placement_contact_sheet.jpg"
        sheet.save(sheet_path, quality=88)
        report = {"schema": "ape.common-policy-object-variation-screen.v1",
            "precommit": ("artifacts/ape-real-room-next-captures-20261014/"
                          "common_policy_multicue_holdout_precommit.json"
                          if args.pair_set == "holdout" else
                          "artifacts/ape-real-room-next-captures-20261014/"
                          "common_policy_camera_detail_precommit.json"
                          if (args.width, args.height) == (1280, 720) else PRECOMMIT),
            "source_policy_report": str(REPORT.relative_to(ROOT)),
            "target_name": target_name, "distractor_name": distractor_name,
            "object_scale_to_metres": .012,
            "camera_resolution_px": [args.width, args.height],
            "screened_owner_stop_count": len(thumbnails[pair]),
            "supported_count": sum(x["status"] == "provisional_supported_visible"
                                   for x in results[pair]),
            "scenes": results[pair],
            "contact_sheet": str(sheet_path.relative_to(ROOT))}
        (folder / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
        print(folder / "screen.json", report["supported_count"])


if __name__ == "__main__":
    main()
