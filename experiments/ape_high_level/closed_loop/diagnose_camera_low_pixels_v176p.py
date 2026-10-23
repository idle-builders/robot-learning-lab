"""Evaluator-only origin check for V171p camera low-region depth pixels.

The actor never receives these segmentation labels. They test whether the
camera cue's supporting depth samples came from the prop, person, furniture
or the ape's own visible body in the saved development scenes.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from articulated_person_scene import (FAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person, set_person_pose)
import direct_structural_rooms_v270 as structural
from run import ROOT, sha
import run_early_standoff_owner_v148p as v148p
import run_live_articulated_owner_v343 as owner
import run_pre_show_reserve_screen_v100 as v100
import run_standoff_scene_variants_v160p as v160p
import screen_standoff_camera_depth_v171p as v171p
import stress_standoff_range_packets_v162p as v162p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-camera-low-pixel-origin"
KEY_TIMES_S = (17.1, 17.3, 17.5, 17.7)


def category(model, geom_id: int) -> str:
    if geom_id < 0 or geom_id >= model.ngeom:
        return "none"
    name = model.geom(geom_id).name
    if name == "loop_stool_geom":
        return "moving_low_prop"
    if name.startswith("loop_person_"):
        return "person"
    if model.geom_group[geom_id] in (1, 4):
        return "ape_self"
    return "other_scene"


def selected_from_packet(packet) -> tuple[np.ndarray, np.ndarray,
                                          np.ndarray]:
    depth = packet["sampled_depth_m"].astype(float)
    height, width = map(int, packet["full_shape_hw"])
    rows, cols = np.mgrid[v171p.DEPTH_OFFSET:height:v171p.DEPTH_STRIDE,
                          v171p.DEPTH_OFFSET:width:v171p.DEPTH_STRIDE]
    focal = height/(2*math.tan(math.radians(float(packet["fovy_deg"]))/2))
    valid = np.isfinite(depth) & (depth > .08) & (
        depth < v171p.MAX_DEPTH_M)
    rays = np.stack(((cols-width/2)/focal,
                     -(rows-height/2)/focal,
                     -np.ones_like(depth)), axis=-1)[valid]
    origin = packet["camera_origin_xyz_m"]
    rotation = packet["camera_rotation"]
    xyz = origin+(rays*depth[valid, None])@rotation.T
    estimate = packet["estimated_pose_xyyaw"]
    delta = xyz[:, :2]-estimate[:2]
    c, s = math.cos(float(estimate[2])), math.sin(float(estimate[2]))
    x = c*delta[:, 0]+s*delta[:, 1]
    y = -s*delta[:, 0]+c*delta[:, 1]
    z = xyz[:, 2]
    selected = ((x > .08) & (x < .50) &
                (y > .05) & (y < .50) &
                (z > .015) & (z < .10))
    selected_x = x[selected]
    nearest = np.zeros_like(selected)
    if len(selected_x):
        nearest[selected] = selected_x <= np.quantile(selected_x, .1)
    return valid, selected, nearest


def scan_case(case: dict, prior: dict, staging: Path,
              source_rows: dict) -> list[dict]:
    if case["class"] == "hazard":
        v160p.VariantRoom.variant = v160p.VARIANTS[case["variant"]]
        room_type = v160p.VariantRoom
    else:
        v148p.StandoffRoom.profile = case["variant"]
        room_type = v148p.StandoffRoom
    with np.load(staging, allow_pickle=False) as packet:
        qpos = packet["actor_qpos"].copy()
        estimate = packet["estimated_pose_xyyaw"].copy()
    room = room_type(case=tuple(prior["case"]),
                     pose_source="command", branch=OUT)
    try:
        room.layout(case["seed"], absent=False, furnished=True)
        qpos[0] += case["station_x_m"]
        estimate[0] += case["station_x_m"]
        shoulder = room.model.joint("left_shoulder_pitch").qposadr[0]
        qpos[shoulder] = .6
        room.data.qpos[:] = qpos
        room.actual[:] = (qpos[0], qpos[1],
                          2*math.atan2(qpos[6], qpos[3]))
        room.estimated[:] = estimate
        start_head = float(qpos[room.head_addr])
        room.head = start_head
        rows = []
        for time_s in KEY_TIMES_S:
            room.time_s = time_s
            shown = min(1., max(0., (time_s-16.6)/.4))
            hand = (START_HAND_XYZ_M+
                    shown*(FAR_HAND_XYZ_M-START_HAND_XYZ_M))
            set_person_pose(room.model, hand,
                            carried_item="loop_TomatoSauce")
            if time_s >= 17.-1e-6:
                room.model.body_pos[room.model.body(
                    "loop_TomatoSauce").id] = owner.OWNER_SHOW_XYZ_M
            look = min(1., max(0., (time_s-17.)/.4))
            room.head = start_head*(1.-look)
            room._forward()
            for camera in v171p.CAMERAS:
                saved = next(s for s in source_rows[camera]["samples"]
                             if s["capture_time_s"] == time_s)
                with np.load(ROOT / saved["depth_packet_path"],
                             allow_pickle=False) as packet:
                    valid, selected, nearest = selected_from_packet(packet)
                rgb, _ = room.capture(camera, depth=False)
                with Image.open(ROOT / saved["rgb_path"]) as prior_rgb:
                    saved_rgb = np.asarray(prior_rgb.convert("RGB"),
                                           dtype=np.int16)
                difference = float(np.mean(np.abs(
                    np.asarray(rgb, dtype=np.int16)-saved_rgb)))
                room.renderer.enable_segmentation_rendering()
                room.renderer.update_scene(room.data, camera=camera)
                segmentation = room.renderer.render().copy()
                room.renderer.disable_segmentation_rendering()
                height, width = segmentation.shape[:2]
                grid_rows, grid_cols = np.mgrid[
                    v171p.DEPTH_OFFSET:height:v171p.DEPTH_STRIDE,
                    v171p.DEPTH_OFFSET:width:v171p.DEPTH_STRIDE]
                geom_ids = segmentation[
                    grid_rows[valid], grid_cols[valid], 0]
                geom_types = segmentation[
                    grid_rows[valid], grid_cols[valid], 1]
                selected_ids = geom_ids[selected]
                nearest_ids = geom_ids[nearest]
                counts = Counter(category(room.model, int(gid))
                                 for gid in selected_ids)
                nearest_counts = Counter(category(room.model, int(gid))
                                         for gid in nearest_ids)
                packet_path = (OUT / case["id"] / camera /
                               f"segmentation-{time_s:.1f}.npz")
                packet_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    packet_path,
                    sampled_geom_ids=segmentation[
                        grid_rows, grid_cols, 0].astype(np.int32),
                    sampled_geom_types=segmentation[
                        grid_rows, grid_cols, 1].astype(np.int32),
                    capture_time_s=time_s)
                rows.append({
                    "source_case_id": case["id"],
                    "class": case["class"],
                    "camera": camera,
                    "time_s": time_s,
                    "source_depth_packet_path": saved["depth_packet_path"],
                    "source_depth_packet_sha256": saved[
                        "depth_packet_sha256"],
                    "source_rgb_path": saved["rgb_path"],
                    "source_rgb_sha256": saved["rgb_sha256"],
                    "rerender_mean_rgb_absolute_difference": round(
                        difference, 5),
                    "segmentation_packet_path": str(
                        packet_path.relative_to(ROOT)),
                    "segmentation_packet_sha256": sha(packet_path),
                    "valid_segmentation_type_count": int(np.count_nonzero(
                        geom_types == int(mujoco.mjtObj.mjOBJ_GEOM))),
                    "low_roi_depth_support": int(np.count_nonzero(selected)),
                    "low_roi_origin_counts_evaluator_only": dict(counts),
                    "nearest_10pct_origin_counts_evaluator_only": dict(
                        nearest_counts),
                })
        return rows
    finally:
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_file = v171p.OUT / "report.json"
    source = json.loads(source_file.read_text())
    source_rows = {}
    for row in source["rows"]:
        source_rows.setdefault(row["source_case_id"], {})[
            row["camera"]] = row
    cases_file = v162p.OUT / "report.json"
    cases = json.loads(cases_file.read_text())["source_cases"]
    v98_file = v100.V98 / "live-moving-owner-offer-audit.json"
    v98 = json.loads(v98_file.read_text())
    old_model = structural.structural_model
    structural.structural_model = lambda kind, side: add_person(
        old_model(kind, side))
    rows = []
    try:
        for case in cases:
            seed = case["seed"]
            prior = next(row for row in v98["rows"]
                         if row["condition"] == "moving" and
                         row["case"][0] == seed)
            staging = (v100.V98 / "moving" / f"seed{seed}-present" /
                       f"seed{seed}-present/side-probe-00.npz")
            rows.extend(scan_case(case, prior, staging,
                                  source_rows[case["id"]]))
            print(case["id"], flush=True)
    finally:
        structural.structural_model = old_model
    summary = {}
    for camera in v171p.CAMERAS:
        group = [row for row in rows if row["camera"] == camera]
        summary[camera] = {
            "diagnostic_frames": len(group),
            "maximum_rerender_rgb_mean_difference": max(
                row["rerender_mean_rgb_absolute_difference"]
                for row in group),
            "low_roi_self_pixels": sum(row[
                "low_roi_origin_counts_evaluator_only"].get("ape_self", 0)
                for row in group),
            "nearest_10pct_self_pixels": sum(row[
                "nearest_10pct_origin_counts_evaluator_only"].get(
                    "ape_self", 0) for row in group),
            "hazard_low_roi_prop_pixels": sum(row[
                "low_roi_origin_counts_evaluator_only"].get(
                    "moving_low_prop", 0)
                for row in group if row["class"] == "hazard"),
        }
    report = {
        "schema": "robot-learning-lab.ape-camera-low-pixel-origin.v1",
        "source_sha256": sha(SOURCE),
        "v171_report_sha256": sha(source_file),
        "v162_report_sha256": sha(cases_file),
        "key_times_s": KEY_TIMES_S,
        "scope": "Evaluator-only segmentation rerender at four fixed times "
                 "in each of sixteen saved owner scenes and two cameras. "
                 "Actor depth packets provide the low-region pixel mask; "
                 "the simulator segmentation identifies visible geom "
                 "origins for diagnosis only. No actor classification, "
                 "sensor selection or physical visibility guarantee follows.",
        "summary": summary,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(OUT / "report.json")


if __name__ == "__main__":
    main()
