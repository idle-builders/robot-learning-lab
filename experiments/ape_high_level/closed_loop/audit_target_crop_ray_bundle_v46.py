"""Audit actor RGB-D crop-surface rays at saved reachable viewpoints."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from audit_3d_target_visibility_v45 import in_frustum, ray_evidence
from depth_voxel import DepthVoxelMap
from robot_geometry import RobotGeometry
from run import H, W, ROOT, Room, locate_world, sha


SOURCE = Path(__file__)
INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v44"
CANDIDATES = ROOT / "artifacts/ape-closed-loop-cove50-v43/viewpoint-visibility-audit.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v46"


def crop_surface_points(frame, fovy_deg: float) -> np.ndarray:
    depth = frame["depth_m"]
    box = frame["proposal_box_xyxy_px"]
    origin = frame["camera_origin_m"]
    rotation = frame["camera_rotation"]
    center = locate_world(box.tolist(), (origin, rotation), depth)
    if center is None:
        return np.zeros((0, 3), dtype=float)
    center = np.asarray(center)
    focal = H / (2 * math.tan(math.radians(fovy_deg) / 2))
    center_u = int(np.clip(round((box[0] + box[2]) / 2), 0, W - 1))
    center_v = int(np.clip(round((box[1] + box[3]) / 2), 0, H - 1))
    center_depth = float(np.median(depth[max(0, center_v - 2):center_v + 3,
                                         max(0, center_u - 2):center_u + 3]))
    points = []
    for vf in np.linspace(.2, .8, 5):
        for uf in np.linspace(.2, .8, 5):
            u = float(box[0] + uf * (box[2] - box[0]))
            v = float(box[1] + vf * (box[3] - box[1]))
            x, y = int(np.clip(round(u), 0, W - 1)), int(np.clip(round(v), 0, H - 1))
            distance = float(np.median(depth[max(0, y - 1):y + 2,
                                            max(0, x - 1):x + 2]))
            if not (.08 < distance < 1.4) or abs(distance - center_depth) > .06:
                continue
            ray = np.asarray([(u - W / 2) / focal,
                              -(v - H / 2) / focal, -1.])
            point = origin + rotation @ (distance * ray)
            if np.linalg.norm(point - center) <= .09:
                points.append(point)
    return np.asarray(points, dtype=float).reshape(-1, 3)


def classify_bundle(voxels: DepthVoxelMap, origin: np.ndarray,
                    rotation: np.ndarray, fovy_deg: float,
                    points: np.ndarray, time_s: float) -> dict:
    statuses = [ray_evidence(voxels, origin, point, time_s)["status"]
                for point in points if in_frustum(origin, rotation, point, fovy_deg)]
    count = Counter(statuses)
    total = len(statuses)
    return {"in_frustum_rays": total,
            "free_rays": count["free"],
            "occupied_rays": count["occupied"],
            "unknown_rays": count["unknown"],
            "ray_supported": bool(total and count["free"] >= 3 and
                                  count["free"] / total >= .25)}


def main() -> None:
    candidate_data = json.loads(CANDIDATES.read_text())
    rows = []
    source_inputs = {"v43_candidates": sha(CANDIDATES),
                     "v45_source": sha(ROOT / "experiments/ape_high_level/closed_loop/"
                                       "audit_3d_target_visibility_v45.py")}
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
            source_inputs[str(seed)] = {"chest": sha(chest_path),
                                        "head": sha(head_path)}
            chest = np.load(chest_path)
            head = np.load(head_path)
            chest_fovy = float(room.model.cam_fovy[
                room.model.camera("chest_low50_x45").id])
            head_fovy = float(room.model.cam_fovy[
                room.model.camera("lower_rgb").id])
            surface = crop_surface_points(chest, chest_fovy)
            if len(surface) < 3:
                raise AssertionError(f"Only {len(surface)} crop points in seed {seed}")
            robot = RobotGeometry(room.model)
            chest_qpos = robot.pose_qpos(room.model.qpos0.copy(),
                                         chest["estimated_pose_xyyaw"])
            robot.set_state(chest_qpos)
            chest_solids = robot.primitives()
            robot.set_state(head["actor_qpos"])
            head_solids = robot.primitives()
            modes = {}
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
            for candidate in candidate_data["rows"]:
                if candidate["seed"] != seed:
                    continue
                room.estimated[:] = (*candidate["xy_m"],
                                     float(head["estimated_pose_xyyaw"][2]))
                origin, rotation = room.camera_pose_estimate("chest_low50_x45")
                rows.append({"seed": seed,
                             "candidate_xy_m": candidate["xy_m"],
                             "crop_surface_point_count": len(surface),
                             "evaluator_visible_at_goal": candidate[
                                 "evaluator_target_visible_at_goal"],
                             "evaluator_target_pixels_at_goal": candidate[
                                 "evaluator_target_pixels_at_goal"],
                             "head_only": classify_bundle(
                                 modes["head_only"], origin, rotation,
                                 chest_fovy, surface, 2.),
                             "chest_and_head": classify_bundle(
                                 modes["chest_and_head"], origin, rotation,
                                 chest_fovy, surface, 2.)})
    finally:
        room.close()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_target_crop_ray_bundle_v46_source.py").write_bytes(
        SOURCE.read_bytes())
    output = {"scope": "Actor-only confirmed-crop RGB-D surface sample and time-stamped head/chest depth voxel ray bundle at V43 reachable viewpoints. Simulator target pixels score only. No live motion or physical sensor authority.",
              "source_sha256": sha(SOURCE),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "inputs_sha256": source_inputs,
              "rows": rows}
    (OUT / "target-crop-ray-bundle-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    for mode in ("head_only", "chest_and_head"):
        supported = [row for row in rows if row[mode]["ray_supported"]]
        print(mode, "supported", len(supported),
              "visible_supported", sum(row["evaluator_visible_at_goal"]
                                        for row in supported),
              "visible_total", sum(row["evaluator_visible_at_goal"]
                                   for row in rows),
              "crop_points", [row["crop_surface_point_count"]
                              for row in rows if row["candidate_xy_m"] == [.04, 0.]])


if __name__ == "__main__":
    main()
