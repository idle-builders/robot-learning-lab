"""Screen actor XYZ target and 3-D ray visibility at saved frontier candidates."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap, VOXEL_M
from robot_geometry import RobotGeometry
from run import H, W, ROOT, Room, locate_world, sha


SOURCE = Path(__file__)
INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v44"
CANDIDATES = ROOT / "artifacts/ape-closed-loop-cove50-v43/viewpoint-visibility-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v45"


def ray_evidence(voxels: DepthVoxelMap, origin: np.ndarray,
                 target: np.ndarray, time_s: float) -> dict:
    segment = target - origin
    length = float(np.linalg.norm(segment))
    direction = segment / length
    states = [voxels.state(origin + distance * direction, time_s)
              for distance in np.arange(.08, max(.08, length - .06),
                                        VOXEL_M)]
    counts = Counter(states)
    occupied = np.asarray([
        [(x + .5) * VOXEL_M, (y + .5) * VOXEL_M, (z + .5) * VOXEL_M]
        for (x, y, z), stamp in voxels.occupied.items()
        if stamp >= time_s - 1.6 and
        stamp >= voxels.free.get((x, y, z), -math.inf)],
        dtype=float).reshape(-1, 3)
    minimum_mark_gap = None
    if len(occupied):
        fractions = (occupied - origin) @ segment / (length * length)
        interior = (fractions > .08) & (fractions < .90)
        if np.any(interior):
            gaps = np.linalg.norm(
                occupied[interior] -
                (origin + fractions[interior, None] * segment), axis=1)
            minimum_mark_gap = float(gaps.min())
    return {"status": ("occupied" if counts["occupied"] else
                       "unknown" if counts["unknown"] else "free"),
            "free_cells": counts["free"],
            "occupied_cells": counts["occupied"],
            "unknown_cells": counts["unknown"],
            "minimum_occupied_mark_gap_m": (
                None if minimum_mark_gap is None else
                round(minimum_mark_gap, 4)),
            "near_occupied_mark_4cm": (minimum_mark_gap is not None and
                                        minimum_mark_gap < .04)}


def observed_endpoints(frame, fovy_deg: float, solids: list) -> np.ndarray:
    """Replay the same sparse valid depth returns without voxel rounding."""
    depth = frame["depth_m"]
    height, width = depth.shape
    stride = 8
    rows, cols = np.mgrid[stride // 2:height:stride,
                          stride // 2:width:stride]
    readings = depth[rows, cols]
    valid = np.isfinite(readings) & (readings > .08) & (readings < 1.4)
    focal = height / (2 * math.tan(math.radians(fovy_deg) / 2))
    rays = np.stack(((cols - width / 2) / focal,
                     -(rows - height / 2) / focal,
                     -np.ones_like(readings)), axis=-1)[valid]
    origin = frame["camera_origin_m"]
    rotation = frame["camera_rotation"]
    points = origin + (rays * readings[valid, None]) @ rotation.T
    return np.asarray([
        point for point in points
        if np.linalg.norm(point - origin) >= .12
        and np.linalg.norm(point[:2] - origin[:2]) >= .12
        and RobotGeometry.solid_distance(point, solids) >= 0.
    ], dtype=float).reshape(-1, 3)


def point_ray_gap(points: np.ndarray, origin: np.ndarray,
                  target: np.ndarray) -> dict:
    segment = target - origin
    length2 = float(segment @ segment)
    fractions = (points - origin) @ segment / length2
    interior = (fractions > .08) & (fractions < .90)
    if not np.any(interior):
        return {"minimum_raw_hit_gap_m": None,
                "nearest_raw_hit_ray_fraction": None,
                "raw_hits_within_1cm": 0, "raw_hits_within_2cm": 0}
    gaps = np.linalg.norm(points[interior] -
                          (origin + fractions[interior, None] * segment), axis=1)
    nearest = int(np.argmin(gaps))
    return {"minimum_raw_hit_gap_m": round(float(gaps.min()), 4),
            "nearest_raw_hit_ray_fraction": round(float(fractions[interior][nearest]), 4),
            "raw_hits_within_1cm": int(np.count_nonzero(gaps < .01)),
            "raw_hits_within_2cm": int(np.count_nonzero(gaps < .02))}


def in_frustum(origin: np.ndarray, rotation: np.ndarray,
               target: np.ndarray, fovy_deg: float) -> bool:
    point = rotation.T @ (target - origin)
    if point[2] >= -.08:
        return False
    focal = H / (2 * math.tan(math.radians(fovy_deg) / 2))
    u = W / 2 + focal * point[0] / -point[2]
    v = H / 2 - focal * point[1] / -point[2]
    return bool(0 <= u < W and 0 <= v < H)


def main() -> None:
    candidate_data = json.loads(CANDIDATES.read_text())
    rows = []
    input_hashes = {"v43_candidates": sha(CANDIDATES)}
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
            chest_path = episode / "owner-confirmed-rgbd.npz"
            head_path = episode / "head-depth-frame-1.npz"
            input_hashes[str(seed)] = {"chest": sha(chest_path),
                                       "head": sha(head_path)}
            chest = np.load(chest_path)
            head = np.load(head_path)
            target = np.asarray(locate_world(
                chest["proposal_box_xyxy_px"].tolist(),
                (chest["camera_origin_m"], chest["camera_rotation"]),
                chest["depth_m"]))
            robot = RobotGeometry(room.model)
            chest_qpos = robot.pose_qpos(
                room.model.qpos0.copy(), chest["estimated_pose_xyyaw"])
            robot.set_state(chest_qpos)
            chest_solids = robot.primitives()
            robot.set_state(head["actor_qpos"])
            head_solids = robot.primitives()
            modes = {}
            point_modes = {}
            head_fovy = float(room.model.cam_fovy[
                room.model.camera("lower_rgb").id])
            chest_fovy = float(room.model.cam_fovy[
                room.model.camera("chest_low50_x45").id])
            head_points = observed_endpoints(head, head_fovy, head_solids)
            chest_points = observed_endpoints(chest, chest_fovy, chest_solids)
            for mode in ("head_only", "chest_and_head"):
                voxels = DepthVoxelMap()
                if mode == "chest_and_head":
                    voxels.observe(chest["depth_m"],
                                   (chest["camera_origin_m"],
                                    chest["camera_rotation"]),
                                   float(chest["capture_time_s"]),
                                   fovy_deg=chest_fovy,
                                   self_hit=lambda point: RobotGeometry.solid_distance(
                                       point, chest_solids) < 0.)
                voxels.observe(head["depth_m"],
                               (head["camera_origin_m"],
                                head["camera_rotation"]),
                               float(head["capture_time_s"]),
                               fovy_deg=head_fovy,
                               self_hit=lambda point: RobotGeometry.solid_distance(
                                   point, head_solids) < 0.)
                modes[mode] = voxels
                point_modes[mode] = (head_points if mode == "head_only" else
                                     np.concatenate((chest_points, head_points)))
            for candidate in candidate_data["rows"]:
                if candidate["seed"] != seed:
                    continue
                pose = np.array((*candidate["xy_m"],
                                 float(head["estimated_pose_xyyaw"][2])))
                room.estimated[:] = pose
                origin, rotation = room.camera_pose_estimate("chest_low50_x45")
                fovy = float(room.model.cam_fovy[
                    room.model.camera("chest_low50_x45").id])
                rows.append({"seed": seed, "candidate_xy_m": candidate["xy_m"],
                             "actor_target_xyz_m": target.tolist(),
                             "actor_target_in_chest_frustum": in_frustum(
                                 origin, rotation, target, fovy),
                             "evaluator_visible_at_goal": candidate[
                                 "evaluator_target_visible_at_goal"],
                             "evaluator_target_pixels_at_goal": candidate[
                                 "evaluator_target_pixels_at_goal"],
                             "head_only": {
                                 **ray_evidence(modes["head_only"], origin,
                                                target, 2.),
                                 **point_ray_gap(point_modes["head_only"],
                                                 origin, target)},
                             "chest_and_head": {
                                 **ray_evidence(modes["chest_and_head"],
                                                origin, target, 2.),
                                 **point_ray_gap(point_modes["chest_and_head"],
                                                 origin, target)}})
    finally:
        room.close()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_3d_target_visibility_v45_source.py").write_bytes(
        SOURCE.read_bytes())
    output = {"scope": "Actor RGB-D target XYZ and calibrated head/chest ray maps projected from saved V44 observations to V43 reachable viewpoints; simulator target pixels used only to score. Unknown ray segments remain unknown. No live motion, no physical depth artifacts, and no motion authority.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "inputs_sha256": input_hashes, "rows": rows}
    (OUT / "target-visibility-3d-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    for mode in ("head_only", "chest_and_head"):
        count = Counter(row[mode]["status"] for row in rows)
        marked = sum(row[mode]["near_occupied_mark_4cm"] for row in rows)
        print(mode, dict(count), "near_marks", marked,
              "frustum", sum(row["actor_target_in_chest_frustum"] for row in rows),
              "render_visible", sum(row["evaluator_visible_at_goal"] for row in rows))


if __name__ == "__main__":
    main()
