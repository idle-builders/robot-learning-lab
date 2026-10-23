"""Screen actor-reachable progressive viewpoints against saved RGB-D rays."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from audit_3d_target_visibility_v45 import in_frustum
from audit_target_crop_ray_bundle_v46 import crop_surface_points
from audit_viewpoint_visibility_v43 import path_to
from depth_voxel import DepthVoxelMap, VOXEL_M
from goal_map_route import GoalMapRoute
from robot_geometry import RobotGeometry
from run import ROOT, Room, sensed_range, sha


SOURCE = Path(__file__)
INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v44"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v47"
CAMERA = "chest_low50_x45"


def ray_status(voxels: DepthVoxelMap, origin: np.ndarray,
               target: np.ndarray, time_s: float) -> str:
    segment = target - origin
    length = float(np.linalg.norm(segment))
    states = Counter(voxels.state(origin + distance / length * segment, time_s)
                     for distance in np.arange(.08, max(.08, length - .06),
                                               VOXEL_M))
    return ("occupied" if states["occupied"] else
            "unknown" if states["unknown"] else "free")


def classify(voxels: DepthVoxelMap, origin: np.ndarray,
             rotation: np.ndarray, fovy: float,
             surface: np.ndarray, time_s: float) -> dict:
    states = Counter(ray_status(voxels, origin, point, time_s)
                     for point in surface
                     if in_frustum(origin, rotation, point, fovy))
    total = sum(states.values())
    return {"in_frustum_rays": total,
            "free_rays": states["free"],
            "occupied_rays": states["occupied"],
            "unknown_rays": states["unknown"],
            "ray_supported": bool(total and states["free"] >= 3 and
                                  states["free"] / total >= .25)}


def target_pixels(room: Room, xy: np.ndarray, yaw: float) -> int:
    room.actual[:] = (float(xy[0]), float(xy[1]), yaw)
    room._forward()
    box = room.truth_box("TomatoSauce", CAMERA)
    return 0 if box is None else int((box[2] - box[0]) * (box[3] - box[1]))


def main() -> None:
    rows = []
    counts = {}
    inputs = {}
    room = Room()
    try:
        for seed in (85, 88):
            room.reset()
            room.layout(seed, absent=False, furnished=True)
            side = 1 if seed % 2 else -1
            room.model.body_pos[room.model.body("loop_table").id] = (
                .40, side * .46, 0.)
            room.set_table_height(.37)
            room._forward()
            episode = INPUT / (f"furnished_table_corridor-seed{seed}-0-"
                               "chest_rgbd_range_guard_goal_map-none")
            trace_path = episode / "trace.json"
            chest_path = episode / "owner-confirmed-rgbd.npz"
            head_path = episode / "head-depth-frame-1.npz"
            inputs[str(seed)] = {"trace": sha(trace_path),
                                 "chest": sha(chest_path),
                                 "head": sha(head_path)}
            trace = json.loads(trace_path.read_text())
            first = next(e for e in trace["events"]
                         if e["action"] == "initial_goal_map_route_planned")
            chest = np.load(chest_path)
            head = np.load(head_path)
            pose = np.asarray(first["pose_estimate_xyyaw"], dtype=float)
            target_xy = np.asarray(first["target_estimate_xy_m"], dtype=float)
            stamp = float(first["t_s"])
            scan = first["range_72_beams_m"]
            room.actual[2] = pose[2] - .003
            room._forward()
            fovy = float(room.model.cam_fovy[room.model.camera(CAMERA).id])
            surface = crop_surface_points(chest, fovy)
            robot = RobotGeometry(room.model)
            robot.set_state(robot.pose_qpos(
                room.model.qpos0.copy(), chest["estimated_pose_xyyaw"]))
            chest_solids = robot.primitives()
            robot.set_state(head["actor_qpos"])
            head_solids = robot.primitives()
            voxels = DepthVoxelMap()
            voxels.observe(chest["depth_m"],
                           (chest["camera_origin_m"], chest["camera_rotation"]),
                           float(chest["capture_time_s"]), fovy_deg=fovy,
                           self_hit=lambda point: RobotGeometry.solid_distance(
                               point, chest_solids) < 0.)
            voxels.observe(head["depth_m"],
                           (head["camera_origin_m"], head["camera_rotation"]),
                           float(head["capture_time_s"]),
                           fovy_deg=float(room.model.cam_fovy[
                               room.model.camera("lower_rgb").id]),
                           self_hit=lambda point: RobotGeometry.solid_distance(
                               point, head_solids) < 0.)
            upper = voxels.upper_occupied_xy(stamp)
            route = GoalMapRoute()
            route.observe(sensed_range(room, seed, -1), pose, 1.6)
            route.observe(scan, pose, stamp)
            route.plan(scan, pose, target_xy, None, stamp, searching=False,
                       extra_obstacles_xy=upper,
                       allow_confirmed_frontier=True)
            if abs(len(route.last_reachable_costs) -
                   first["goal_map_route"]["reachable_cells"]) > 1:
                raise AssertionError("First actor map differs from live V44")
            origin_cell = route.cell(pose[:2])
            candidate_rows = []
            initial_distance = float(np.linalg.norm(target_xy - pose[:2]))
            for cell, cost in route.last_reachable_costs.items():
                if cell == origin_cell or cell[0] % 2 or cell[1] % 2:
                    continue
                if cost > .9:
                    continue
                path = path_to(cell, origin_cell, route.last_parent)
                if path is None or len(path) - 1 > 20:
                    continue
                xy = route.xy(cell)
                distance = float(np.linalg.norm(target_xy - xy))
                if not .27 <= distance <= .70:
                    continue
                room.estimated[:] = (xy[0], xy[1], pose[2])
                camera_origin, camera_rotation = room.camera_pose_estimate(CAMERA)
                view = classify(voxels, camera_origin, camera_rotation,
                                fovy, surface, stamp)
                improvement = initial_distance - distance
                candidate_rows.append({"cell": list(cell),
                                       "xy_m": xy.tolist(),
                                       "route_cost": round(float(cost), 4),
                                       "path_steps": len(path) - 1,
                                       "path_cells": [list(p) for p in path],
                                       "target_distance_est_m": round(distance, 4),
                                       "target_distance_improvement_m": round(improvement, 4),
                                       "actor_score": round(improvement - .15 * cost, 4),
                                       **view})
            supported = sorted((r for r in candidate_rows if r["ray_supported"]),
                               key=lambda r: r["actor_score"], reverse=True)
            evaluated = []
            for rank, candidate in enumerate(supported[:5], start=1):
                xy = np.asarray(candidate["xy_m"])
                pixels = target_pixels(room, xy, pose[2] - .003)
                row = {"rank": rank, **candidate,
                       "evaluator_target_pixels_at_goal": pixels,
                       "evaluator_visible_at_goal": pixels >= 150}
                if rank == 1:
                    row["evaluator_path_target_pixels"] = [
                        target_pixels(room, route.xy(tuple(cell)), pose[2] - .003)
                        for cell in candidate["path_cells"]]
                evaluated.append(row)
            rows.extend({"seed": seed, **row} for row in evaluated)
            counts[str(seed)] = {"reachable_cells_replayed": len(route.last_reachable_costs),
                                 "reachable_cells_live": first["goal_map_route"]["reachable_cells"],
                                 "candidate_cells": len(candidate_rows),
                                 "supported_cells": len(supported),
                                 "progressive_supported_8cm": sum(
                                     r["target_distance_improvement_m"] >= .08
                                     for r in supported),
                                 "best_actor_candidate": evaluated[0] if evaluated else None}
    finally:
        room.close()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_visible_progress_v47_source.py").write_bytes(
        SOURCE.read_bytes())
    output = {"scope": "Actor first-observation 8cm-grid route/viewpoint screen with independent evaluator RGB pixel visibility on top five candidates and top path only. No live motion or detector, no motion authority.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "inputs_sha256": inputs,
              "counts": counts, "evaluated_top_rows": rows}
    (OUT / "visible-progress-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    for seed in (85, 88):
        print(seed, {k:v for k,v in counts[str(seed)].items()
                     if k != "best_actor_candidate"})
        best = counts[str(seed)]["best_actor_candidate"]
        if best:
            print("best", best["xy_m"], best["target_distance_improvement_m"],
                  best["evaluator_target_pixels_at_goal"],
                  best.get("evaluator_path_target_pixels"))


if __name__ == "__main__":
    main()
