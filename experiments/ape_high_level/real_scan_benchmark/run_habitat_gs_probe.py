"""Moving-camera perception probe in a real Habitat-GS indoor capture.

This is a sensor/route integration experiment with a 50 cm kinematic proxy,
not a body-qualified owner interaction. RGB comes from the real-world Gaussian
reconstruction; collision and range come from its separately supplied mesh.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d

from run_mesh_probe import segment_free, visual_target, xy_at
from screen_habitat_gs import ASSETS, REPO
from huggingface_hub import hf_hub_download
from fan_view_route import FanViewRoute


def read_nav_raster(path: Path, level_y: float, resolution: float = .05,
                    anchor_xz: np.ndarray | None = None) -> dict:
    """Read Detour v2 tile polygons and rasterize a single floor only."""
    data = path.read_bytes()
    magic, version, tile_count = struct.unpack_from("<III", data)
    if magic != 0x4d534554 or version != 2:
        raise ValueError(f"Unsupported navmesh: {magic:x} v{version}")
    offset = 96  # Detour set header and 56-byte NavMeshSettings.
    polygons = []
    for _ in range(tile_count):
        _, size = struct.unpack_from("<II", data, offset)
        offset += 8
        tile = data[offset:offset + size]
        header = struct.unpack_from("<15i10f", tile)
        count_polys, count_vertices = header[6:8]
        vertices = np.frombuffer(tile, dtype="<f4", count=3 * count_vertices,
                                 offset=100).reshape(-1, 3)
        poly_offset = 100 + 12 * count_vertices
        for index in range(count_polys):
            row = struct.unpack_from("<I6H6HHBB", tile, poly_offset + 32 * index)
            polygon = vertices[list(row[1:1 + row[-2]])]
            if len(polygon) >= 3 and abs(float(np.median(polygon[:, 1])) - level_y) < .35:
                polygons.append(polygon[:, [0, 2]].copy())
        offset += size
    if not polygons:
        raise ValueError("No navigation polygons on the selected floor")
    all_vertices = np.concatenate(polygons)
    minimum = all_vertices.min(axis=0) - .1
    maximum = all_vertices.max(axis=0) + .1
    shape = np.ceil((maximum - minimum)[::-1] / resolution).astype(int) + 1
    mask = np.zeros(tuple(shape), dtype=np.uint8)
    for polygon in polygons:
        pixel = np.rint((polygon - minimum) / resolution).astype(np.int32)
        cv2.fillPoly(mask, [pixel], 1)
    # The source navmesh already encodes a 20 cm agent radius, so do not erode
    # it a second time. Its 1.5 m agent height is not a 50 cm body model.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        raise ValueError("No connected free floor after erosion")
    if anchor_xz is None:
        main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    else:
        pixel = np.rint((anchor_xz - minimum) / resolution).astype(int)
        col, row = pixel
        main = int(labels[row, col]) if (0 <= row < labels.shape[0] and
                                      0 <= col < labels.shape[1]) else 0
        if main == 0:
            free_cells = np.argwhere(labels > 0)
            nearest = free_cells[np.argmin(np.linalg.norm(
                free_cells - pixel[::-1], axis=1))]
            main = int(labels[tuple(nearest)])
    return {"xy_min_m": minimum, "resolution_m": resolution,
            "main_component": labels == main, "floor_y_m": level_y,
            "polygon_count": len(polygons)}


def recorded_start_route(nav_path: Path, episodes: list[dict], level: float,
                         mesh_path: Path, extended: bool = False):
    """Choose a clear short segment near a published camera-covered start."""
    candidates = []
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    clearance_scene = o3d.t.geometry.RaycastingScene()
    clearance_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    for index in sorted({0, len(episodes) // 3, 2 * len(episodes) // 3}):
        episode = episodes[index]
        start_recorded = np.asarray(episode["start_position"])[[0, 2]]
        if abs(float(episode["start_position"][1]) - level) > .25:
            continue
        nav = read_nav_raster(nav_path, level, anchor_xz=start_recorded)
        free = np.argwhere(nav["main_component"])
        points = np.asarray([xy_at(nav, cell) for cell in free])
        nearest_distance = np.linalg.norm(points - start_recorded, axis=1)
        start = points[int(np.argmin(nearest_distance))]
        if float(np.min(nearest_distance)) > .3:
            continue
        goal = np.asarray(episode["goals"][0]["position"])[[0, 2]]
        direction = goal - start
        if np.linalg.norm(direction) < .1:
            continue
        direction /= np.linalg.norm(direction)
        distances = ((4.0, 3.5, 3.0, 2.5, 2.0, 1.8, 1.6, 1.4) if extended
                     else (2.0, 1.8, 1.6, 1.4))
        for distance in distances:
            for yaw in np.linspace(-.6, .6, 25):
                rotated = np.array([direction[0] * math.cos(yaw) -
                                    direction[1] * math.sin(yaw),
                                    direction[0] * math.sin(yaw) +
                                    direction[1] * math.cos(yaw)])
                target = start + distance * rotated
                if segment_free(nav, start, target):
                    eye = np.array([start[0], level + .42, start[1]])
                    marker_center = np.array([target[0], level + .14, target[1]])
                    cast = clearance_scene.cast_rays(o3d.core.Tensor(
                        np.r_[eye, marker_center - eye][None, :].astype(np.float32)))
                    if float(cast["t_hit"].numpy()[0]) < .95:
                        continue
                    candidates.append((distance - abs(yaw) - 3 * float(np.min(nearest_distance)),
                                       nav, start, target, episode["episode_id"]))
    if not candidates:
        raise ValueError("No short free segment from sampled published camera starts")
    _, nav, start, target, episode_id = max(candidates, key=lambda row: row[0])
    return nav, start, target, episode_id


def make_scene(scene_number: int, target_xz: np.ndarray | None, floor_y: float):
    split = "train" if scene_number <= 55 else "val"
    scene_id = f"scene{scene_number:02d}"
    folder = ASSETS / split / scene_id
    gs_path = Path(hf_hub_download(repo_id=REPO, repo_type="dataset",
        filename=f"{split}/{scene_id}/{scene_id}.gs.ply", local_dir=ASSETS))
    mesh_path = Path(hf_hub_download(repo_id=REPO, repo_type="dataset",
        filename=f"{split}/{scene_id}/{scene_id}.mesh.ply", local_dir=ASSETS))
    splats = o3d.t.io.read_point_cloud(str(gs_path))
    logits = splats.point.opacity.numpy()
    splats.point.opacity = o3d.core.Tensor(
        (1 / (1 + np.exp(-np.clip(logits, -30, 30)))).astype(np.float32))
    renderer = o3d.visualization.rendering.OffscreenRenderer(480, 360)
    renderer.scene.set_background([0, 0, 0, 1])
    appearance = o3d.visualization.rendering.MaterialRecord()
    appearance.shader = "gaussianSplat"
    appearance.gaussian_splat_sh_degree = 0
    renderer.scene.add_geometry("real_capture", splats, appearance)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    if target_xz is not None:
        target = o3d.geometry.TriangleMesh.create_sphere(radius=.12, resolution=12)
        target.translate([float(target_xz[0]), floor_y + .14, float(target_xz[1])])
        target.paint_uniform_color([1., .08, .55])
        # Open3D's Gaussian pass does not depth-composite this mesh with the
        # captured collision mesh. The camera below draws only target pixels
        # whose calibrated ray first hits the target.
        target_geometry_id = ray_scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(target))
        if target_geometry_id != 1:
            raise ValueError("Unexpected ray-scene geometry ordering")
    return renderer, ray_scene, folder


def observe(renderer, ray_scene, xz: np.ndarray, heading: float, floor_y: float):
    eye = np.array([xz[0], floor_y + .42, xz[1]])
    # With Y up and initial forward +Z, Open3D's image-right is world -X.
    # Positive heading is therefore a right turn toward world -X.
    forward = np.array([-math.sin(heading), 0., math.cos(heading)])
    right = np.array([-math.cos(heading), 0., -math.sin(heading)])
    up = np.array([0., 1., 0.])
    renderer.setup_camera(75., eye + forward, eye, up)
    rgb = np.asarray(renderer.render_to_image())[:, :, :3].copy()
    height, width = rgb.shape[:2]
    focal = height / (2 * math.tan(math.radians(75.) / 2))
    u, v = np.meshgrid(np.arange(width // 2) * 2 + .5,
                       np.arange(height // 2) * 2 + .5)
    directions = (forward + ((u - (width - 1) / 2) / focal)[..., None] * right +
                  (((height - 1) / 2 - v) / focal)[..., None] * up)
    origins = np.broadcast_to(eye, directions.shape)
    rays = np.concatenate((origins, directions), axis=-1).astype(np.float32)
    cast = ray_scene.cast_rays(o3d.core.Tensor(rays))
    depth_small = cast["t_hit"].numpy()
    depth = cv2.resize(depth_small, (width, height), interpolation=cv2.INTER_NEAREST)
    visible_target = cv2.resize(
        (cast["geometry_ids"].numpy() == 1).astype(np.uint8),
        (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    rgb[visible_target] = [255, 20, 140]
    angles = np.arange(-180., 180., 10., dtype=np.float32)
    directions = np.column_stack((-np.sin(heading + np.deg2rad(angles)),
                                  np.zeros_like(angles),
                                  np.cos(heading + np.deg2rad(angles)))).astype(np.float32)
    readings = []
    for sensor_height in (.12, .32):
        origin = np.array([xz[0], floor_y + sensor_height, xz[1]], np.float32)
        rays = np.column_stack((np.repeat(origin[None, :], len(angles), axis=0),
                                directions)).astype(np.float32)
        hits = ray_scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(float)
        hits[hits > 2.] = np.inf
        readings.append(hits)
    return rgb, depth, np.stack(readings), angles


def run(scene_number: int, steps: int = 80, extended: bool = False) -> dict:
    split = "train" if scene_number <= 55 else "val"
    scene_id = f"scene{scene_number:02d}"
    nav_path = Path(hf_hub_download(repo_id=REPO, repo_type="dataset",
        filename=f"{split}/{scene_id}/{scene_id}.navmesh", local_dir=ASSETS))
    mesh_path = Path(hf_hub_download(repo_id=REPO, repo_type="dataset",
        filename=f"{split}/{scene_id}/{scene_id}.mesh.ply", local_dir=ASSETS))
    episode_path = Path(hf_hub_download(repo_id=REPO, repo_type="dataset",
        filename=f"episodes/pointnav/{split}/content/{scene_id}.json.gz",
        local_dir=ASSETS))
    episodes = json.load(gzip.open(episode_path, "rt"))["episodes"]
    level = float(np.median([row["start_position"][1] for row in episodes]))
    nav, start, target, episode_id = recorded_start_route(
        nav_path, episodes, level, mesh_path, extended)
    heading = math.atan2(start[0] - target[0], target[1] - start[1]) + math.radians(30)
    renderer, ray_scene, folder = make_scene(scene_number, target, level)
    output = folder / ("gs_probe_extended" if extended else "gs_probe")
    output.mkdir(exist_ok=True)
    actual = start.copy()
    actor = np.zeros(2)
    actor_heading = 0.
    planner = FanViewRoute(proxy_radius_m=.17, required_free_planes=(0, 1))
    last_target = None
    previous_command = previous_measured = 0.
    consecutive_reobserve = 0
    trace = []
    writer = imageio.get_writer(str(output / "first_person.mp4"), fps=5,
                                codec="libx264", quality=7, macro_block_size=8)
    try:
        for index in range(steps):
            rgb, depth, ranges, angles = observe(renderer, ray_scene, actual, heading, level)
            visual = visual_target(rgb, depth, actor, math.radians(actor_heading))
            if visual["seen"]:
                last_target = np.asarray(visual["estimated_xy_m"])
            if previous_command > .02 and previous_measured < .5 * previous_command:
                decision = {"intent": "hold_request_help", "reason": "blocked_motion"}
            elif last_target is None:
                decision = {"intent": "search_turn", "yaw_delta_deg": 10.,
                            "reason": "taught_proxy_not_visible"}
            else:
                distance_to_ultimate = float(np.linalg.norm(last_target - actor))
                intermediate = extended and distance_to_ultimate > 2.2
                local_target = (actor + (last_target - actor) *
                                (1.8 / distance_to_ultimate)) if intermediate else last_target
                decision = planner.plan(ranges, angles, actor, actor_heading,
                                        local_target, index * .2)
                if intermediate:
                    decision["intermediate_goal"] = True
                    decision["ultimate_target_distance_m"] = distance_to_ultimate
                    if decision["intent"] == "hold_viewpoint":
                        decision["intent"] = "reobserve_at_intermediate_goal"
            if decision["intent"] in ("hold_reobserve", "reobserve_at_intermediate_goal"):
                consecutive_reobserve += 1
                if consecutive_reobserve >= 5:
                    decision = {"intent": "hold_request_help",
                                "reason": "persistent_local_uncertainty",
                                "reobserve_cycles": consecutive_reobserve}
            else:
                consecutive_reobserve = 0
            command_yaw = float(decision.get("yaw_delta_deg", 0.))
            command_step = float(decision.get("forward_step_m", 0.))
            accepted = True
            if command_step:
                candidate = actual + command_step * np.array([
                    -math.sin(heading), math.cos(heading)])
                accepted = segment_free(nav, actual, candidate)
                if accepted:
                    actual = candidate
                    actor += command_step * np.array([
                        math.sin(math.radians(actor_heading)),
                        math.cos(math.radians(actor_heading))])
            previous_command, previous_measured = command_step, command_step if accepted else 0.
            heading += math.radians(command_yaw)
            actor_heading += command_yaw
            frame = rgb.copy()
            cv2.putText(frame, f"{index:03d} {decision['intent']} | object: {'seen' if visual['seen'] else 'unseen'}",
                        (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            writer.append_data(frame)
            trace.append({"frame": index, "observation": {
                "visual": visual, "range_low_m": ranges[0].tolist(),
                "range_mid_m": ranges[1].tolist(), "angles_deg": angles.tolist(),
                "valid_depth_fraction": float(np.mean(np.isfinite(depth))),
                "black_background_fraction": float(np.mean(np.all(rgb == 0, axis=2)))},
                "decision": decision, "motion_accepted": accepted,
                "actor_odometry_xz_m": actor.tolist(),
                "truth_xz_m": actual.tolist(), "truth_heading_rad": heading})
            if decision["intent"] in ("hold_viewpoint", "hold_request_help"):
                break
    finally:
        writer.close()
    gap = float(np.linalg.norm(actual - target) - .12)
    result = {"schema": "ape.habitat-gs-real-capture-probe.v1",
              "scene_id": scene_id, "source": f"https://huggingface.co/datasets/{REPO}",
              "status": "provisional_kinematic_appearance_and_sensor_probe",
              "published_start_episode_id": episode_id,
              "start_xz_m": start.tolist(), "target_xz_m": target.tolist(),
              "route_mode": "extended" if extended else "short_wiring",
              "initial_target_center_distance_m": float(np.linalg.norm(target - start)),
              "navmesh_floor_y_m": level, "navmesh_polygon_count": nav["polygon_count"],
              "frames": len(trace), "outcome": trace[-1]["decision"]["intent"],
              "true_target_surface_gap_m": gap,
              "truth_standoff_within_band": bool(trace[-1]["decision"]["intent"] == "hold_viewpoint" and 1.20 <= gap <= 1.35),
              "move_rejections": sum(not row["motion_accepted"] for row in trace),
              "actual_path_length_m": float(np.linalg.norm(np.diff(
                  np.asarray([row["truth_xz_m"] for row in trace]), axis=0),
                  axis=1).sum()) if len(trace) > 1 else 0.,
              "mean_black_background_fraction": float(np.mean([
                  row["observation"]["black_background_fraction"] for row in trace])),
              "limitations": ["Gaussian novel-view quality depends on capture height and position",
                              "virtual distinct-color object and simplified target detector",
                              "idealized motion feedback, no articulated selected body",
                              "no person or owner interaction"],
              "trace": trace}
    (output / "trace.json").write_text(json.dumps(result, indent=2) + "\n")
    return {key: value for key, value in result.items() if key != "trace"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-number", type=int, default=32)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.scene_number, args.steps, args.extended), indent=2), flush=True)
