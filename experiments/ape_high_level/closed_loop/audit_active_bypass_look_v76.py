"""V76: in-place lower-head RGB-D yaw sweep from failed corridor stops."""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import ImageDraw

from audit_standoff_information_v75 import (MAX_INFORMATION_CREDIT_M,
                                            MIN_NEW_CELLS,
                                            first_unknown_cells)
from body_aware_field import field_and_descent
from depth_voxel import DepthVoxelMap, VOXEL_M
from goal_directed_frontier import GRID_M, MAX_ROUTE_COST_M, cell, xy
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha


SOURCE = Path(__file__)
RUN = ROOT / "artifacts/ape-closed-loop-cove50-v72-stopstate"
V75 = ROOT / "artifacts/ape-closed-loop-cove50-v75/standoff-information-audit.json"
V51 = ROOT / "artifacts/ape-closed-loop-cove50-v51/stop-state-attribution.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v76"
SEEDS = (85, 88)
CONFIG = "chest_rgbd_range_guard_goal_map"
HEAD_YAW_RAD = (0., -.65, +.65)


def band_cells(voxel: DepthVoxelMap) -> tuple[set[tuple[int, int, int]],
                                              set[tuple[int, int, int]]]:
    def in_band(key: tuple[int, int, int]) -> bool:
        z = (key[2] + .5) * VOXEL_M
        return .18 <= z <= .43
    occupied = {key for key, stamp in voxel.occupied.items()
                if in_band(key) and stamp >= voxel.free.get(key, -math.inf)}
    free = {key for key, stamp in voxel.free.items()
            if in_band(key) and stamp > voxel.occupied.get(key, -math.inf)}
    return free, occupied


def choose_standoff(cells: dict, reachable: dict, ranges: list,
                    pose: np.ndarray, target: np.ndarray,
                    time_s: float, upper: np.ndarray) -> dict:
    _, debug, field, known_free, _ = field_and_descent(
        cells, reachable, ranges, pose, target, time_s, upper,
        return_field=True)
    candidates = []
    for node, travel in reachable.items():
        if (node == cell(pose[:2]) or node not in known_free or
                node not in field or travel > MAX_ROUTE_COST_M):
            continue
        unknown = first_unknown_cells(xy(node), pose[:2], cells, upper, time_s)
        if len(unknown) < MIN_NEW_CELLS:
            continue
        score = travel + field[node] - min(MAX_INFORMATION_CREDIT_M,
                                           .01 * len(unknown))
        candidates.append((score, node, travel, field[node], len(unknown)))
    best = min(candidates, default=None)
    return {"body_field_origin_cost_m": debug["field_origin_cost_m"],
            "candidate_count": len(candidates),
            "chosen_cell": None if best is None else best[1],
            "chosen_xy_m": None if best is None else xy(best[1]).tolist(),
            "chosen_route_cost_m": None if best is None else round(best[2], 4),
            "chosen_remaining_cost_m": None if best is None else round(best[3], 4),
            "chosen_predicted_unknown_cells": None if best is None else best[4]}


def replay(seed: int, room: Room, robot: RobotGeometry,
           v75_by_seed: dict, routes: dict) -> dict:
    episode = RUN / f"furnished_table_corridor-seed{seed}-0-{CONFIG}-none"
    state_path = episode / "first-stop-actor-state.npz"
    with np.load(state_path) as state:
        cells = {(int(x), int(y)): (float(free_s), float(hit_s))
                 for x, y, free_s, hit_s in state["range_map_cells"]}
        reachable = {(int(x), int(y)): float(cost)
                     for x, y, cost in state["reachable_costs_cells"]}
        ranges = [None if np.isnan(value) else float(value)
                  for value in state["range_72_beams_m"]]
        pose = state["estimated_pose_xyyaw"].copy()
        target = state["target_estimate_xy_m"].copy()
        saved_upper = state["fresh_upper_obstacles_xy_m"].copy()
        qpos = state["actor_qpos_estimated"].copy()
        start_time = float(state["capture_time_s"])

    side = 1 if seed % 2 else -1
    room.reset()
    room.layout(seed, absent=False, furnished=True)
    room.model.body_pos[room.model.body("loop_table").id] = (
        .40, side * .46, 0.)
    room.set_table_height(.37)
    room.data.qpos[:] = qpos
    room.actual[:] = pose
    room.estimated[:] = pose
    voxel = DepthVoxelMap()
    frames = []
    views = []
    neutral_free: set[tuple[int, int, int]] = set()
    neutral_occupied: set[tuple[int, int, int]] = set()
    neutral_upper = np.zeros((0, 2))
    for index, head_yaw in enumerate(HEAD_YAW_RAD):
        room.head = head_yaw
        room.time_s = start_time + .4 * index
        room._forward()
        rgb, depth = room.capture("lower_rgb", depth=True)
        camera_pose = room.camera_pose_estimate("lower_rgb")
        actor_qpos = room.pose_data.qpos.copy()
        robot.set_state(actor_qpos)
        solids = robot.primitives()
        status = voxel.observe(
            depth, camera_pose, room.time_s,
            fovy_deg=float(room.model.cam_fovy[
                room.model.camera("lower_rgb").id]),
            self_hit=lambda point: RobotGeometry.solid_distance(
                point, solids) < 0.)
        image = rgb.copy()
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 34), fill=(0, 0, 0))
        draw.text((8, 9), f"Cove-50 virtual head yaw {head_yaw:+.2f} rad",
                  fill=(255, 255, 255))
        frames.append(np.asarray(image))
        rgb_path = OUT / f"seed{seed}-look{index}-rgb.png"
        image.save(rgb_path)
        sensor_path = OUT / f"seed{seed}-look{index}-actor-input.npz"
        np.savez_compressed(sensor_path, depth_m=depth,
                            camera_origin_m=camera_pose[0],
                            camera_rotation=camera_pose[1],
                            actor_qpos=actor_qpos,
                            estimated_pose_xyyaw=pose,
                            capture_time_s=room.time_s)
        free, occupied = band_cells(voxel)
        if index == 0:
            neutral_free = free.copy()
            neutral_occupied = occupied.copy()
            neutral_upper = voxel.upper_occupied_xy(room.time_s).copy()
        views.append({"head_yaw_rad": head_yaw,
                      "capture_time_s": round(room.time_s, 3),
                      "rgb_sha256": sha(rgb_path),
                      "actor_input_sha256": sha(sensor_path),
                      "depth_rays_used": status["depth_rays_used"],
                      "free_band_voxels": len(free),
                      "occupied_band_voxels": len(occupied),
                      "upper_xy_marks": len(voxel.upper_occupied_xy(room.time_s))})
    video_path = OUT / f"seed{seed}-active-head-look.mp4"
    with imageio.get_writer(video_path, fps=4, codec="libx264", quality=7) as writer:
        for frame in frames:
            for _ in range(4):
                writer.append_data(frame)

    final_free, final_occupied = band_cells(voxel)
    new_free = final_free - neutral_free
    new_occupied = final_occupied - neutral_occupied
    sweep_upper = voxel.upper_occupied_xy(room.time_s)
    fused_upper = np.concatenate((saved_upper, sweep_upper))
    baseline = v75_by_seed[seed]
    after_neutral = choose_standoff(
        cells, reachable, ranges, pose, target, start_time,
        np.concatenate((saved_upper, neutral_upper)))
    after = choose_standoff(cells, reachable, ranges, pose, target,
                            room.time_s, fused_upper)
    route = routes[seed]["evaluator_route"]["path_xy_m"]
    reference_dy = route[1][1] - route[0][1]
    def route_comparison(choice: dict) -> tuple[bool | None, float | None]:
        chosen_xy = (None if choice["chosen_xy_m"] is None else
                     np.asarray(choice["chosen_xy_m"]))
        same_side = (None if chosen_xy is None else
                     float(chosen_xy[1] - pose[1]) * reference_dy > 0.)
        distance = (None if chosen_xy is None else
                    float(min(np.linalg.norm(chosen_xy - np.asarray(point))
                              for point in route)))
        return same_side, distance
    neutral_same_side, neutral_distance = route_comparison(after_neutral)
    same_side, distance_to_route = route_comparison(after)
    return {"seed": seed, "stop_state_sha256": sha(state_path),
            "estimated_pose_xyyaw": pose.tolist(),
            "views": views, "neutral_free_band_voxels": len(neutral_free),
            "neutral_occupied_band_voxels": len(neutral_occupied),
            "new_free_band_voxels": len(new_free),
            "new_occupied_band_voxels": len(new_occupied),
            "saved_upper_xy_marks": len(saved_upper),
            "neutral_upper_xy_marks": len(neutral_upper),
            "sweep_upper_xy_marks": len(sweep_upper),
            "baseline_standoff_xy_m": baseline["chosen_xy_m"],
            "after_neutral_standoff": after_neutral,
            "after_neutral_matches_reference_detour_side": neutral_same_side,
            "after_neutral_distance_to_reference_path_m": (
                None if neutral_distance is None else round(neutral_distance, 4)),
            "after_sweep_standoff": after,
            "after_sweep_matches_reference_detour_side": same_side,
            "after_sweep_distance_to_reference_path_m": (
                None if distance_to_route is None else round(distance_to_route, 4)),
            "video_sha256": sha(video_path)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    v75 = json.loads(V75.read_text())
    v75_by_seed = {int(row["seed"]): row for row in v75["rows"]}
    v51 = json.loads(V51.read_text())
    routes = {int(row["seed"]): row for row in v51["rows"]}
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        rows = [replay(seed, room, robot, v75_by_seed, routes)
                for seed in SEEDS]
    finally:
        room.close()
    seed88 = next(row for row in rows if row["seed"] == 88)
    gate = (all(row["new_free_band_voxels"] + row["new_occupied_band_voxels"] > 0
                for row in rows) and
            seed88["after_sweep_matches_reference_detour_side"] is True)
    output = {"scope": "Static selected Cove-50 kinematic head yaw 0/-0.65/+0.65rad RGB-D at exact saved V72 base pose; actor voxel free/occupied self-masked and upper marks fused into read-only body-aware standoff route. Evaluator route compared only after selection; no qualified look or motion authority.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "v75_audit_sha256": sha(V75),
              "v51_evaluator_audit_sha256": sha(V51),
              "rows": rows, "active_look_diagnostic_gate_pass": gate}
    (OUT / "active-bypass-look-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / "audit_active_bypass_look_v76_source.py").write_bytes(
        SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], "new free/occupied",
              row["new_free_band_voxels"], row["new_occupied_band_voxels"],
              "before", row["baseline_standoff_xy_m"],
              "after", row["after_sweep_standoff"],
              "reference side", row["after_sweep_matches_reference_detour_side"])
    print("active look gate:", gate)


if __name__ == "__main__":
    main()
