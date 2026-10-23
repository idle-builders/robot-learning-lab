"""Route a provisional ape footprint around a real RoboCasa island.

The actor builds known-free space from its own 72-ray planar scans and asks
for one short eye-rig translation at a time. Simulator geometry only checks
the proposed path afterward. This is a navigation-planning transfer probe,
not a qualified walking/body-clearance result or a complete teach/find task.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robosuite.models.tasks import ManipulationTask


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout2-route-v407"
START = np.array([1.05, -4.20])
GOAL = np.array([4.60, -1.20])
RADIUS_M = .17  # provisional circular body envelope, not a selected design
MARGIN_M = .10
GRID_M = .10
LIDAR_Z_M = .30
BEAMS = 72
NOISE_STD_M = .005


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_with_layout2(*, obstacle_xy: tuple[float, float] | None = None) -> mujoco.MjModel:
    random.seed(3201)
    np.random.seed(3201)
    arena = KitchenArena(layout_id=2, style_id=1, rng=np.random.default_rng(3), clutter_mode=0)
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[],
                            mujoco_objects=list(arena.fixtures.values()))
    ET.SubElement(task.worldbody, "camera", name="head",
                  pos=f"{START[0]} {START[1]} .46", xyaxes="1 0 0 0 0 1", fovy="70")
    if obstacle_xy is not None:
        body = ET.SubElement(task.worldbody, "body", name="route_obstacle",
                             pos=f"{obstacle_xy[0]} {obstacle_xy[1]} .25")
        ET.SubElement(body, "geom", name="route_obstacle_geom", type="cylinder",
                      size=".24 .25", rgba="0.86 0.69 0.15 1")
    return task.get_model()


def planar_scan(model: mujoco.MjModel, data: mujoco.MjData, xy: np.ndarray,
                max_range_m: float, rng: np.random.Generator) -> list[float | None]:
    origin = np.array([xy[0], xy[1], LIDAR_Z_M], np.float64)
    ranges = []
    for angle in np.linspace(-math.pi, math.pi, BEAMS, endpoint=False):
        ray = np.array([math.cos(angle), math.sin(angle), 0.], np.float64)
        geom_id = np.array([-1], np.int32)
        distance = mujoco.mj_ray(model, data, origin, ray, None, 1, -1, geom_id)
        noise = rng.normal(0., NOISE_STD_M)
        ranges.append(float(max(.01, distance + noise))
                      if .14 < distance < max_range_m else None)
    return ranges


class SensorGrid:
    def __init__(self) -> None:
        self.xs = np.arange(-.70, 5.11, GRID_M)
        self.ys = np.arange(-4.75, -.24, GRID_M)
        self.xx, self.yy = np.meshgrid(self.xs, self.ys)
        self.free = np.zeros(self.xx.shape, dtype=bool)
        self.occupied = np.zeros(self.xx.shape, dtype=bool)
        self.visits = np.zeros(self.xx.shape, dtype=int)

    def cell(self, xy: np.ndarray) -> tuple[int, int]:
        return (int(np.argmin(np.abs(self.ys - xy[1]))),
                int(np.argmin(np.abs(self.xs - xy[0]))))

    def xy(self, cell: tuple[int, int]) -> np.ndarray:
        return np.array([self.xs[cell[1]], self.ys[cell[0]]])

    def update(self, ranges: list[float | None], pose: np.ndarray,
               max_range_m: float) -> None:
        lengths = np.array([max_range_m if value is None else value for value in ranges])
        dx, dy = self.xx - pose[0], self.yy - pose[1]
        distance = np.hypot(dx, dy)
        angle_index = ((np.arctan2(dy, dx) + math.pi) / (2 * math.pi) * len(ranges)) % len(ranges)
        lower = np.floor(angle_index).astype(int)
        upper = (lower + 1) % len(ranges)
        clear_radius = np.minimum(lengths[lower], lengths[upper]) - RADIUS_M - MARGIN_M
        newly_free = (distance < clear_radius) & (distance < max_range_m - RADIUS_M - MARGIN_M)
        self.free |= newly_free
        for angle, reading in zip(np.linspace(-math.pi, math.pi, len(ranges), endpoint=False), ranges):
            if reading is None:
                continue
            hit_x = pose[0] + reading * math.cos(angle)
            hit_y = pose[1] + reading * math.sin(angle)
            near_hit = (self.xx - hit_x) ** 2 + (self.yy - hit_y) ** 2 <= (RADIUS_M + MARGIN_M) ** 2
            self.occupied |= near_hit
        self.free &= ~self.occupied
        self.free[self.cell(pose)] = True

    def plan(self, origin_xy: np.ndarray, goal_xy: np.ndarray) -> tuple[np.ndarray | None, str, list[tuple[int, int]]]:
        origin = self.cell(origin_xy)
        goal = self.cell(goal_xy)
        if not self.free[origin]:
            return None, "current_cell_unknown", []
        queue = [(0., origin)]
        costs = {origin: 0.}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1),
                     (1, 1), (1, -1), (-1, 1), (-1, -1))
        while queue:
            cost, cell = heapq.heappop(queue)
            if cost > costs[cell] + 1e-8:
                continue
            for dy, dx in neighbors:
                nxt = cell[0] + dy, cell[1] + dx
                if not (0 <= nxt[0] < len(self.ys) and 0 <= nxt[1] < len(self.xs)):
                    continue
                if not self.free[nxt]:
                    continue
                # Diagonal traversal requires both adjacent cardinal cells.
                if dx and dy and (not self.free[cell[0], nxt[1]] or
                                  not self.free[nxt[0], cell[1]]):
                    continue
                new_cost = cost + GRID_M * math.hypot(dx, dy) + .02 * min(5, self.visits[nxt])
                if new_cost >= costs.get(nxt, math.inf):
                    continue
                costs[nxt] = new_cost
                parent[nxt] = cell
                heapq.heappush(queue, (new_cost, nxt))
        if goal in costs:
            destination, kind = goal, "known_goal"
        else:
            frontier = []
            for cell, cost in costs.items():
                if cell == origin:
                    continue
                near_unknown = any(
                    0 <= cell[0] + dy < len(self.ys) and
                    0 <= cell[1] + dx < len(self.xs) and
                    not self.free[cell[0] + dy, cell[1] + dx] and
                    not self.occupied[cell[0] + dy, cell[1] + dx]
                    for dy, dx in neighbors[:4])
                if near_unknown:
                    heuristic = np.linalg.norm(self.xy(cell) - goal_xy)
                    frontier.append((heuristic + .12 * cost + .12 * self.visits[cell], cell))
            if not frontier:
                return None, "no_reachable_frontier", []
            destination = min(frontier)[1]
            kind = "explore_frontier"
        path = [destination]
        while path[-1] != origin:
            path.append(parent[path[-1]])
        path.reverse()
        if len(path) < 2:
            return None, "frontier_is_current_cell", path
        return self.xy(path[1]), kind, path


def evaluator_clear(model: mujoco.MjModel, data: mujoco.MjData,
                    xy: np.ndarray) -> tuple[bool, str | None]:
    # A provisional 50 cm high disk sampled at its center and rim; independent
    # simulator geometry never enters SensorGrid or its action choice.
    samples = [xy] + [xy + .17 * np.array([math.cos(a), math.sin(a)])
                      for a in np.linspace(0., 2 * math.pi, 8, endpoint=False)]
    for point in samples:
        geom_id = np.array([-1], np.int32)
        hit = mujoco.mj_ray(model, data,
                            np.array([point[0], point[1], .55], np.float64),
                            np.array([0., 0., -1.], np.float64), None, 1, -1, geom_id)
        if geom_id[0] < 0:
            return False, "no_floor"
        if hit < .51:
            return False, model.geom(int(geom_id[0])).name
    return True, None


def frame_with_map(rgb: np.ndarray, grid: SensorGrid, pose: np.ndarray,
                   path: list[tuple[int, int]], status: str, step: int,
                   max_range_m: float) -> np.ndarray:
    image = Image.new("RGB", (1040, 528), "#101619")
    image.paste(Image.fromarray(rgb), (0, 48))
    draw = ImageDraw.Draw(image)
    draw.text((10, 8), f"layout 2 • {max_range_m:.1f} m range • step {step} • {status}", fill="white")
    draw.text((10, 27), f"goal ({GOAL[0]:.2f},{GOAL[1]:.2f}); pose ({pose[0]:.2f},{pose[1]:.2f})",
              fill="#d7e6e9")
    map_x0, map_y0, cell_px = 648, 54, 6
    for row in range(grid.free.shape[0]):
        for col in range(grid.free.shape[1]):
            if grid.occupied[row, col]:
                color = "#a84448"
            elif grid.free[row, col]:
                color = "#3b6b68"
            else:
                continue
            x, y = map_x0 + col * cell_px, map_y0 + row * cell_px
            draw.rectangle((x, y, x + cell_px - 1, y + cell_px - 1), fill=color)
    for row, col in path:
        x, y = map_x0 + col * cell_px + 2, map_y0 + row * cell_px + 2
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#f3a746")
    for point, color in ((START, "#70baff"), (GOAL, "#ffe077"), (pose, "#50ec93")):
        row, col = grid.cell(point)
        x, y = map_x0 + col * cell_px + 2, map_y0 + row * cell_px + 2
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    return np.asarray(image)


def episode(max_range_m: float) -> dict:
    model = model_with_layout2()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera_id = model.camera("head").id
    rng = np.random.default_rng(407)
    grid = SensorGrid()
    pose = START.copy()
    trace = []
    frames = []
    outcome = "step_budget_exhausted"
    for step in range(120):
        model.cam_pos[camera_id, :2] = pose
        mujoco.mj_forward(model, data)
        ranges = planar_scan(model, data, pose, max_range_m, rng)
        grid.update(ranges, pose, max_range_m)
        renderer.update_scene(data, camera="head")
        rgb = renderer.render().copy()
        next_xy, kind, path = grid.plan(pose, GOAL)
        evaluator_ok, evaluator_hit = evaluator_clear(model, data, pose)
        row = {"step": step, "pose_xy_m": pose.tolist(),
               "range_sha256": hashlib.sha256(json.dumps(ranges).encode()).hexdigest(),
               "range_returns": sum(v is not None for v in ranges),
               "known_free_cells": int(grid.free.sum()),
               "occupied_cells": int(grid.occupied.sum()),
               "decision": kind, "requested_xy_m": next_xy.tolist() if next_xy is not None else None,
               "evaluator_current_clear": evaluator_ok, "evaluator_current_hit": evaluator_hit,
               "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
        if next_xy is not None:
            segment_checks = [evaluator_clear(model, data, pose + f * (next_xy - pose))
                              for f in (.25, .5, .75, 1.)]
            row["evaluator_segment_clear"] = all(ok for ok, _ in segment_checks)
            row["evaluator_segment_hits"] = [hit for ok, hit in segment_checks if not ok]
        trace.append(row)
        frames.append(frame_with_map(rgb, grid, pose, path, kind, step, max_range_m))
        if np.linalg.norm(pose - GOAL) <= .15:
            outcome = "goal_viewpoint_reached"
            break
        if next_xy is None:
            outcome = kind
            break
        if not row["evaluator_segment_clear"]:
            outcome = "actor_proposed_geometry_overlap"
            break
        grid.visits[grid.cell(pose)] += 1
        pose = next_xy
    renderer.close()
    OUT.mkdir(parents=True, exist_ok=True)
    video = OUT / f"layout2-range{max_range_m:.1f}m.mp4"
    imageio.mimsave(video, frames, fps=5, codec="libx264", quality=7)
    return {"layout": 2, "style": 1, "max_range_m": max_range_m,
            "scope": "sensor-built grid, kinematic eye rig, evaluator-only footprint witness",
            "outcome": outcome, "start_xy_m": START.tolist(), "goal_xy_m": GOAL.tolist(),
            "end_xy_m": pose.tolist(), "steps": len(trace),
            "unsafe_requested_steps": sum(row.get("evaluator_segment_clear") is False for row in trace),
            "video": {"path": str(video), "sha256": sha256(video), "frames": len(frames)},
            "trace": trace}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = []
    for max_range in (2.2, 5.0):
        row = episode(max_range)
        episodes.append(row)
        print(max_range, row["outcome"], row["steps"], row["end_xy_m"], flush=True)
    report = {"source_sha256": sha256(Path(__file__)),
              "room": "upstream RoboCasa test layout2/style1, seeded3201",
              "body_proxy_radius_m": RADIUS_M, "margin_m": MARGIN_M,
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
