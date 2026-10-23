"""Bound pre-show low-shell coverage if simulator far-plane misses cleared rays.

Real depth cameras can return invalid depth on dark, reflective or occluded
surfaces, so the optimistic bound is not actor-usable free-space evidence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap, VOXEL_M
from run import ROOT, sha
import screen_preshow_camera_shell_v181p as v181p


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-preshow-camera-miss-bound"
SIMULATOR_FAR_DEPTH_MIN_M = 100.
MAX_ACTOR_DEPTH_M = 1.4


def add_simulator_miss_rays(voxel_map: DepthVoxelMap,
                            depth: np.ndarray,
                            origin: np.ndarray,
                            rotation: np.ndarray,
                            fovy_deg: float,
                            time_s: float) -> dict:
    height, width = depth.shape
    stride = v181p.DEPTH_STRIDE
    rows, cols = np.mgrid[stride//2:height:stride,
                          stride//2:width:stride]
    sampled = depth[rows, cols]
    miss = np.isfinite(sampled) & (
        sampled >= SIMULATOR_FAR_DEPTH_MIN_M)
    focal = height/(2*math.tan(math.radians(fovy_deg)/2))
    local = np.stack(((cols-width/2)/focal,
                      -(rows-height/2)/focal,
                      -np.ones_like(sampled)), axis=-1)[miss]
    direction = local @ rotation.T
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    cells = set()
    for distance in np.arange(.04, MAX_ACTOR_DEPTH_M, VOXEL_M):
        xyz = origin+distance*direction
        indices = np.floor(xyz/VOXEL_M).astype(int)
        cells.update(map(tuple, indices))
    for cell in cells:
        voxel_map.free[cell] = time_s
    return {"simulator_far_rays": int(np.count_nonzero(miss)),
            "extra_free_cells": len(cells)}


def states(voxel_map: DepthVoxelMap, samples: np.ndarray,
           time_s: float) -> dict:
    counts = {"free": 0, "occupied": 0, "unknown": 0}
    for point in samples:
        counts[voxel_map.state(point, time_s)] += 1
    return counts


def center_pixel_reasons(depth: np.ndarray, origin: np.ndarray,
                         rotation: np.ndarray, fovy_deg: float,
                         samples: np.ndarray) -> dict:
    height, width = depth.shape
    focal = height/(2*math.tan(math.radians(fovy_deg)/2))
    local = (samples-origin) @ rotation
    camera_depth = -local[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        columns = np.rint(width/2+focal*local[:, 0]/camera_depth).astype(int)
        rows = np.rint(height/2-focal*local[:, 1]/camera_depth).astype(int)
    counts = {"outside_field_or_near_limit": 0,
              "self_masked_or_invalid": 0,
              "simulator_far_plane": 0,
              "surface_before_sample": 0,
              "surface_at_or_beyond_sample": 0}
    for index, expected in enumerate(camera_depth):
        if (not .08 < expected < MAX_ACTOR_DEPTH_M or
            not 0 <= rows[index] < height or
            not 0 <= columns[index] < width):
            counts["outside_field_or_near_limit"] += 1
            continue
        observed = float(depth[rows[index], columns[index]])
        if not np.isfinite(observed) or observed <= .08:
            counts["self_masked_or_invalid"] += 1
        elif observed >= SIMULATOR_FAR_DEPTH_MIN_M:
            counts["simulator_far_plane"] += 1
        elif observed < expected-.02:
            counts["surface_before_sample"] += 1
        else:
            counts["surface_at_or_beyond_sample"] += 1
    return counts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_file = v181p.OUT / "report.json"
    source = json.loads(source_file.read_text())
    rows = []
    for source_row in source["rows"]:
        with np.load(ROOT / source_row["shell_packet"],
                     allow_pickle=False) as shell:
            samples = shell["shell_samples_xyz_m"].copy()
        maps = {camera: DepthVoxelMap() for camera in v181p.CAMERAS}
        camera_rows = {}
        for camera in v181p.CAMERAS:
            packets = [p for p in source_row["pre_show_camera_packets"]
                       if p["camera"] == camera]
            miss_stats = []
            last_packet = None
            for saved in packets:
                path = ROOT / saved["packet_path"]
                if sha(path) != saved["packet_sha256"]:
                    raise AssertionError("Source camera depth changed")
                with np.load(path, allow_pickle=False) as packet:
                    depth = packet["filtered_depth_m"].copy()
                    origin = packet["camera_origin_xyz_m"].copy()
                    rotation = packet["camera_rotation"].copy()
                    fovy = float(packet["fovy_deg"])
                    time_s = float(packet["capture_time_s"])
                maps[camera].observe(depth, (origin, rotation), time_s,
                                     fovy_deg=fovy,
                                     stride=v181p.DEPTH_STRIDE,
                                     max_depth_m=MAX_ACTOR_DEPTH_M,
                                     self_masked=True)
                miss_stats.append(add_simulator_miss_rays(
                    maps[camera], depth, origin, rotation, fovy, time_s))
                last_packet = (depth, origin, rotation, fovy)
            now = v181p.v167p.SCAN_TIMES_S[-1]
            optimistic = states(maps[camera], samples, now)
            conservative = source_row["final_low_shell"][camera]
            if optimistic["free"] < conservative["free"]:
                raise AssertionError("Optimistic bound lost conservative free cells")
            camera_rows[camera] = {
                "conservative_return_only": {name: conservative[name]
                                             for name in optimistic},
                "simulator_miss_as_free_upper_bound": optimistic,
                "last_frame_center_pixel_reasons": center_pixel_reasons(
                    *last_packet, samples),
                "simulator_far_rays_nine_frames": sum(
                    s["simulator_far_rays"] for s in miss_stats),
            }
        joint = v181p.merged(*maps.values())
        camera_rows["both_cameras"] = {
            "conservative_return_only": {name: source_row[
                "final_low_shell"]["both_cameras"][name]
                for name in ("free", "occupied", "unknown")},
            "simulator_miss_as_free_upper_bound": states(
                joint, samples, v181p.v167p.SCAN_TIMES_S[-1]),
        }
        rows.append({
            "source_case_id": source_row["source_case_id"],
            "source_shell_packet": source_row["shell_packet"],
            "source_shell_packet_sha256": source_row[
                "shell_packet_sha256"],
            "shell_cells": len(samples),
            "camera_coverage": camera_rows,
        })
        print(source_row["source_case_id"],
              camera_rows["both_cameras"], flush=True)
    summary = {}
    for camera in (*v181p.CAMERAS, "both_cameras"):
        summary[camera] = {
            "cases": len(rows),
            "shell_cells": sum(r["shell_cells"] for r in rows),
            "return_only_free": sum(r["camera_coverage"][camera][
                "conservative_return_only"]["free"] for r in rows),
            "optimistic_miss_as_free": sum(r["camera_coverage"][camera][
                "simulator_miss_as_free_upper_bound"]["free"] for r in rows),
            "optimistic_unknown": sum(r["camera_coverage"][camera][
                "simulator_miss_as_free_upper_bound"]["unknown"] for r in rows),
        }
    report = {
        "schema": "robot-learning-lab.ape-preshow-camera-miss-bound.v1",
        "source_sha256": sha(SOURCE),
        "v181_report_sha256": sha(source_file),
        "simulator_far_depth_min_m": SIMULATOR_FAR_DEPTH_MIN_M,
        "optimistic_ray_limit_m": MAX_ACTOR_DEPTH_M,
        "scope": "Post-hoc upper bound on V181's same 144 self-masked pre-show "
                 "depth frames. Only explicit simulator far-plane depths "
                 ">=100m are treated as empty rays to 1.4m; ordinary "
                 "missing/self-masked values remain unknown. Real depth "
                 "no-return is not evidence of free space. The 4cm voxel "
                 "map and center-pixel projection are diagnostics, not a "
                 "whole-cell visibility or physical safety certificate.",
        "summary": summary,
        "rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
