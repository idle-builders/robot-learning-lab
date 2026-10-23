"""Evaluator-only audit of floor-object visibility in the saved range fans."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from real_capture_dynamic_scene import TexturedCaptureScene, TexturedFloorObject
from run_real_capture_owner_loop import SCENES


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "assets/real_indoor_scenes/scansceneverse/dynamic_placement_screen/teach_find_placement_screen"
BATCH = BASE / "batch_v1.json"
SCREEN = BASE / "screen.json"
OUT = BASE / "range_visibility_audit_v1.json"
PROPS = ROOT / ".cache/ape-high-level/hope_meshes_eval"


def main() -> None:
    batch = json.loads(BATCH.read_text())
    placements = {r["scene_id"]: r for r in json.loads(SCREEN.read_text())["scenes"]}
    target = TexturedFloorObject(PROPS / "Parmesan.obj", scale=.012)
    other = TexturedFloorObject(PROPS / "CreamCheese.obj", scale=.012)
    rows = []
    for item in batch["scenes"]:
        if item["task_status"] != "completed":
            continue
        placement = placements[item["scene_id"]]
        cfg = SCENES[placement["scene_key"]]
        source = json.loads(cfg["trace"].read_text())
        scene = TexturedCaptureScene(ROOT / source["source_mesh"],
            floor_y=source["floor_y_1pct_m"],
            scale=source.get("scale_to_metres", 1.))
        run = ROOT / item["summary"]
        folder = run.parent
        owner = json.loads((folder / "owner_approach/trace.json").read_text())["trace"][-1]
        present = json.loads((folder / "find_present/trace.json").read_text())
        xy = owner["evaluator_actual_xz_m"]
        eye = np.array([xy[0], scene.floor_y + .42, xy[1]])
        initial_view = next((r for r in present["observations"]
                             if r["evaluator_target_visible_px"] > 0), None)
        initial = None
        if initial_view is not None:
            initial_box = initial_view["evaluator_target_box_xyxy"]
            center_u = (initial_box[0] + initial_box[2]) / 2
            focal = 360 / (2 * math.tan(math.radians(75.) / 2))
            initial_bearing = math.degrees(math.atan((center_u - 320) / focal))
            initial_angles = np.asarray(initial_view["range_angles_deg"], np.float32)
            scene.set_props({"CreamCheese": (other,
                tuple(placement["find_distractor_xz"]), 0.)})
            without_initial = scene.range_fan(eye,
                initial_view["camera_heading_deg"], owner_xz=None,
                angles_deg=initial_angles)
            scene.set_props({"CreamCheese": (other,
                tuple(placement["find_distractor_xz"]), 0.),
                "Parmesan": (target, tuple(placement["find_target_xz"]), 0.)})
            with_initial = scene.range_fan(eye,
                initial_view["camera_heading_deg"], owner_xz=None,
                angles_deg=initial_angles)
            saved_initial = np.asarray([[np.nan if v is None else v
                for v in initial_view[n]] for n in ("range_low_m", "range_mid_m")])
            observed_initial = np.where(np.isfinite(with_initial),
                                        with_initial, np.nan)
            if not np.allclose(saved_initial, observed_initial,
                               atol=.002, equal_nan=True):
                raise RuntimeError(f"Initial range packet mismatch: {item['name']}")
            initial_hit = with_initial + .01 < without_initial
            initial_offset = -initial_bearing
            initial_nearest = int(np.argmin(abs(initial_angles - initial_offset)))
            initial_neighbor = abs(initial_angles - initial_offset) <= 5.01
            initial = {"frame": initial_view["frame"],
                "actor_camera_target_box_available": any(
                    p["evaluator_target_iou"] >= .25
                    for p in initial_view["proposals"]),
                "visual_target_bearing_deg": initial_bearing,
                "nearest_beam_angle_deg": float(initial_angles[initial_nearest]),
                "nearest_low_hit_target": bool(initial_hit[0, initial_nearest]),
                "nearest_mid_hit_target": bool(initial_hit[1, initial_nearest]),
                "within_5deg_low_hit_target": bool(
                    initial_hit[0, initial_neighbor].any()),
                "within_5deg_mid_hit_target": bool(
                    initial_hit[1, initial_neighbor].any())}
        verification = present["verification"]
        if verification is None:
            rows.append({"scene_id": item["scene_id"], "name": item["name"],
                         "status": "no_verified_candidate",
                         "initial_target_visible_view": initial})
            continue
        location = verification.get("candidate_localization", {})
        if location.get("status") != "located":
            rows.append({"scene_id": item["scene_id"], "name": item["name"],
                         "status": "candidate_depth_unavailable",
                         "localization": location,
                         "initial_target_visible_view": initial})
            continue
        yaw = verification["camera_heading_deg"]
        angles = np.asarray(verification["range_angles_deg"], np.float32)
        scene.set_props({"CreamCheese": (other,
            tuple(placement["find_distractor_xz"]), 0.)})
        without = scene.range_fan(eye, yaw, owner_xz=None, angles_deg=angles)
        scene.set_props({"CreamCheese": (other,
            tuple(placement["find_distractor_xz"]), 0.),
            "Parmesan": (target, tuple(placement["find_target_xz"]), 0.)})
        with_target = scene.range_fan(eye, yaw, owner_xz=None, angles_deg=angles)
        saved = np.asarray([[np.nan if v is None else v for v in verification[n]]
                            for n in ("range_low_m", "range_mid_m")])
        observed = np.where(np.isfinite(with_target), with_target, np.nan)
        if not np.allclose(saved, observed, atol=.002, equal_nan=True):
            raise RuntimeError(f"Range packet disagrees with evaluator replay: {item['name']}")
        object_hit = with_target + .01 < without
        visual_range = float(np.hypot(location["camera_forward_m"],
                                      location["camera_right_m"]))
        offset = -location["bearing_deg"]
        nearest = int(np.argmin(abs(angles - offset)))
        neighbor = abs(angles - offset) <= 5.01
        rows.append({"scene_id": item["scene_id"], "name": item["name"],
            "status": "scored", "visual_slant_range_m": visual_range,
            "initial_target_visible_view": initial,
            "visual_bearing_deg": location["bearing_deg"],
            "evaluator_rgbd_position_error_m": location[
                "evaluator_target_position_error_m"],
            "nearest_beam_angle_deg": float(angles[nearest]),
            "nearest_low_m": float(with_target[0, nearest]),
            "nearest_mid_m": float(with_target[1, nearest]),
            "nearest_low_hit_target": bool(object_hit[0, nearest]),
            "nearest_mid_hit_target": bool(object_hit[1, nearest]),
            "within_5deg_low_hit_target": bool(object_hit[0, neighbor].any()),
            "within_5deg_mid_hit_target": bool(object_hit[1, neighbor].any()),
            "all_low_target_beam_count": int(object_hit[0].sum()),
            "all_mid_target_beam_count": int(object_hit[1].sum())})
        print(item["name"], "depth_error", round(location[
            "evaluator_target_position_error_m"], 3), "low", bool(
            object_hit[0, neighbor].any()), "mid", bool(
            object_hit[1, neighbor].any()), flush=True)
    scored = [r for r in rows if r["status"] == "scored"]
    report = {"schema": "ape.real-scan-object-range-audit.v1",
        "source_batch": str(BATCH.relative_to(ROOT)),
        "evaluator_only": True,
        "truth_rule": "a target hit is a ray at least 1 cm closer with the target mesh than without it",
        "scored_count": len(scored),
        "within_5deg_low_target_count": sum(r[
            "within_5deg_low_hit_target"] for r in scored),
        "within_5deg_mid_target_count": sum(r[
            "within_5deg_mid_hit_target"] for r in scored),
        "initial_target_visible_count": sum(r.get(
            "initial_target_visible_view") is not None for r in rows),
        "initial_within_5deg_low_target_count": sum(bool((r.get(
            "initial_target_visible_view") or {}).get(
            "within_5deg_low_hit_target")) for r in rows),
        "initial_within_5deg_mid_target_count": sum(bool((r.get(
            "initial_target_visible_view") or {}).get(
            "within_5deg_mid_hit_target")) for r in rows),
        "rows": rows}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
