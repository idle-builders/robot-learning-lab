"""Find a conservative ape-proxy floor route in one Faro-aligned real room."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_closing, distance_transform_edt, label


ROOT = Path(__file__).resolve().parents[3] / "assets/real_indoor_scenes/mushroom"
RESOLUTION = 0.05
PROXY_RADIUS = 0.17
PROXY_HEIGHT = 0.55


def shortest_path(free: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    frontier = [(0.0, start)]
    cost = {start: 0.0}
    previous = {}
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    while frontier:
        value, node = heapq.heappop(frontier)
        if node == goal:
            break
        if value > cost[node]:
            continue
        for dr, dc in directions:
            nxt = node[0] + dr, node[1] + dc
            if nxt[0] < 0 or nxt[1] < 0 or nxt[0] >= free.shape[0] or nxt[1] >= free.shape[1] or not free[nxt]:
                continue
            step = 1.41421356237 if dr and dc else 1.0
            new_cost = value + step
            if new_cost < cost.get(nxt, float("inf")):
                cost[nxt] = new_cost
                previous[nxt] = node
                heapq.heappush(frontier, (new_cost, nxt))
    if goal not in cost:
        return []
    route = [goal]
    while route[-1] != start:
        route.append(previous[route[-1]])
    return route[::-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", required=True)
    args = parser.parse_args()
    room_dir = ROOT / args.room / "extracted" / "room_datasets" / args.room
    screen = json.loads((ROOT / args.room / "metric_low_view_screen.json").read_text())
    floor_z = screen["floor_from_captured_depth_mode_z_m"]
    transform = np.linalg.inv(np.asarray(json.loads((room_dir / "icp_iphone.json").read_text())["gt_transformation"]))
    cloud = o3d.io.read_point_cloud(str(room_dir / "gt_pd.ply"))
    cloud.transform(transform)
    xyz = np.asarray(cloud.points)
    lower = np.quantile(xyz[:, :2], 0.001, axis=0) - 0.1
    upper = np.quantile(xyz[:, :2], 0.999, axis=0) + 0.1
    width, height = np.ceil((upper - lower) / RESOLUTION).astype(int) + 1
    if width * height > 2_000_000:
        raise ValueError("Faro raster extent is unreasonable")
    col = np.floor((xyz[:, 0] - lower[0]) / RESOLUTION).astype(int)
    row = np.floor((xyz[:, 1] - lower[1]) / RESOLUTION).astype(int)
    inside = (row >= 0) & (col >= 0) & (row < height) & (col < width)
    z = xyz[:, 2]
    floor_points = inside & (np.abs(z - floor_z) < 0.045)
    obstacle_points = inside & (z > floor_z + 0.07) & (z < floor_z + PROXY_HEIGHT)
    linear = row * width + col
    floor_counts = np.bincount(linear[floor_points], minlength=width * height).reshape(height, width)
    obstacle_counts = np.bincount(linear[obstacle_points], minlength=width * height).reshape(height, width)
    floor = binary_closing(floor_counts >= 2, structure=np.ones((3, 3)))
    obstacle = obstacle_counts >= 2
    floor_clearance = distance_transform_edt(floor) * RESOLUTION
    obstacle_clearance = distance_transform_edt(~obstacle) * RESOLUTION
    free = floor & (floor_clearance >= PROXY_RADIUS) & (obstacle_clearance >= PROXY_RADIUS)
    components, n_components = label(free, structure=np.ones((3, 3)))
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    largest = int(np.argmax(sizes)) if n_components else 0
    main = components == largest if largest else np.zeros_like(free)

    frames = json.loads((room_dir / "iphone/long_capture/transformations_colmap.json").read_text())["frames"]
    anchors = []
    free_cells = np.argwhere(main)
    for f in frames:
        point = np.asarray(f["transform_matrix"])[:3, 3]
        if not (floor_z + 0.35 <= point[2] <= floor_z + 0.65) or len(free_cells) == 0:
            continue
        target = (point[:2] - lower) / RESOLUTION
        distances = np.linalg.norm(free_cells[:, ::-1] - target, axis=1) * RESOLUTION
        closest = int(np.argmin(distances))
        if distances[closest] <= 0.35:
            anchors.append((f["file_path"], tuple(map(int, free_cells[closest])), float(distances[closest])))
    best = None
    for i, first in enumerate(anchors):
        for second in anchors[i+1:]:
            distance = float(np.linalg.norm(np.subtract(first[1], second[1])) * RESOLUTION)
            if best is None or distance > best[0]:
                best = (distance, first, second)
    route = shortest_path(main, best[1][1], best[2][1]) if best else []
    route_xy = [[float(lower[0] + (cell[1] + .5) * RESOLUTION),
                 float(lower[1] + (cell[0] + .5) * RESOLUTION)] for cell in route]
    route_length = float(sum(np.linalg.norm(np.subtract(a, b)) for a, b in zip(route_xy, route_xy[1:])))
    nav = np.zeros((height, width, 3), dtype=np.uint8)
    nav[floor] = (50, 70, 50)
    nav[obstacle] = (15, 20, 180)
    nav[main] = (100, 180, 80)
    for r, c in route:
        nav[r, c] = (255, 230, 0)
    out_dir = ROOT / args.room
    cv2.imwrite(str(out_dir / "proxy_route.png"), cv2.resize(nav, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    result = {
        "schema": "ape.mushroom-proxy-route.v1", "room": args.room,
        "floor_z_m": floor_z, "proxy_radius_m": PROXY_RADIUS, "proxy_height_m": PROXY_HEIGHT,
        "resolution_m": RESOLUTION, "largest_connected_free_area_m2": float(sizes[largest] * RESOLUTION**2) if largest else 0.0,
        "nearby_low_recorded_anchors": len(anchors),
        "start_frame": best[1][0] if best else None,
        "goal_frame": best[2][0] if best else None,
        "start_anchor_offset_m": best[1][2] if best else None,
        "goal_anchor_offset_m": best[2][2] if best else None,
        "route_length_m": route_length,
        "route_xy_m": route_xy,
        "caution": "Faro point-raster proxy route. No articulated 50 cm body sweep or contact dynamics.",
    }
    (out_dir / "proxy_route.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "route_xy_m"})


if __name__ == "__main__":
    main()
