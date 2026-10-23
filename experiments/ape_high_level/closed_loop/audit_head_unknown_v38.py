"""Attribute first-look unknown upper-body space without changing the actor.

The V37 first RGB-D frame is replayed into the same actor-only voxel map. The
camera-frustum/depth/segmentation attribution below is evaluator-only: it
explains missing coverage and never enters planning or motion decisions.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from depth_voxel import DepthVoxelMap
from run import H, W, ROOT, Room, sha


SOURCE = Path(__file__)
INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v37"
OUTPUT = ROOT / "artifacts/ape-closed-loop-cove50-v38"


def samples(pose: np.ndarray):
    c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
    for z in (.18, .22, .26, .30, .34, .38):
        for dx in (-.06, 0., .06):
            for dy in (-.20, -.16, -.10, 0., .10, .16, .20):
                yield np.array((pose[0] + c * dx - s * dy,
                                pose[1] + s * dx + c * dy, z))


def is_robot_geom(model: mujoco.MjModel, geom_id: int) -> bool:
    body_id = int(model.geom_bodyid[geom_id])
    while body_id > 1:
        body_id = int(model.body_parentid[body_id])
    return body_id == 1


def explain(point: np.ndarray, camera_pose: tuple[np.ndarray, np.ndarray],
            depth: np.ndarray, segmentation: np.ndarray, model: mujoco.MjModel,
            fovy_deg: float) -> str:
    origin, rotation = camera_pose
    local = rotation.T @ (point - origin)
    forward = -float(local[2])
    if forward <= 0:
        return "behind_camera"
    if forward <= .08:
        return "inside_minimum_depth"
    if forward >= 1.4:
        return "beyond_maximum_depth"
    focal = H / (2 * math.tan(math.radians(fovy_deg) / 2))
    col = int(round(W / 2 + focal * local[0] / forward))
    row = int(round(H / 2 - focal * local[1] / forward))
    if col < 0 or col >= W or row < 0 or row >= H:
        return "outside_field_of_view"
    hit_depth = float(depth[row, col])
    if not math.isfinite(hit_depth) or hit_depth <= .08:
        return "invalid_depth"
    if hit_depth < forward - .035:
        geom_id, kind = map(int, segmentation[row, col, :2])
        if kind == int(mujoco.mjtObj.mjOBJ_GEOM) and geom_id >= 0:
            return ("occluded_by_robot" if is_robot_geom(model, geom_id)
                    else "occluded_by_scene")
        return "occluded_unclassified"
    if hit_depth > forward + .035:
        return "visible_free_ray_not_in_voxel_map"
    return "near_measured_surface"


def main() -> None:
    rows = []
    for seed in (85, 88):
        episode = INPUT / (f"furnished_moving-seed{seed}-0-"
                           "chest_rgbd_range_guard_goal_map-none")
        archived = np.load(episode / "head-depth-first.npz")
        depth = archived["depth_m"]
        estimate = archived["estimated_pose_xyyaw"]
        trace = json.loads((episode / "trace.json").read_text())
        room = Room()
        try:
            room.layout(seed, absent=False, furnished=True)
            room.configure_continuous_obstacle(seed)
            room.time_s = 2.0
            room._update_moving_obstacle()
            room.estimated[:] = estimate
            room._forward()
            room.renderer.enable_segmentation_rendering()
            room.renderer.update_scene(room.data, camera="lower_rgb")
            segmentation = room.renderer.render().copy()
            room.renderer.disable_segmentation_rendering()
            camera_pose = room.camera_pose_estimate("lower_rgb")
            fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
            for stride in (8, 4, 2):
                voxel = DepthVoxelMap()
                started = time.perf_counter()
                observation = voxel.observe(depth, camera_pose, 2.,
                                            fovy_deg=fovy, stride=stride)
                processing_ms = (time.perf_counter() - started) * 1000
                # Initial proposed motion stays within one 4 cm map cell.
                # Eight headings expose whether a result depends on one route.
                for heading in (0., math.pi / 4, math.pi / 2,
                                3 * math.pi / 4, math.pi, -3 * math.pi / 4,
                                -math.pi / 2, -math.pi / 4):
                    future = estimate.copy()
                    future[:2] += .04 * np.array((math.cos(heading),
                                                  math.sin(heading)))
                    counts = Counter()
                    for point in samples(future):
                        state = voxel.state(point, 2.)
                        counts[state] += 1
                        if state == "unknown":
                            counts[explain(point, camera_pose, depth,
                                           segmentation, room.model, fovy)] += 1
                    rows.append({"seed": seed, "stride_px": stride,
                                 "heading_rad": round(heading, 4),
                                 "future_pose_xyyaw": [round(float(x), 4)
                                                       for x in future],
                                 "coverage": {key: counts[key] for key in
                                              ("free", "occupied", "unknown")},
                                 "unknown_attribution": {
                                     key: value for key, value in counts.items()
                                     if key not in ("free", "occupied", "unknown")},
                                 "depth_rays_used": observation["depth_rays_used"],
                                 "processing_wall_ms": round(processing_ms, 2)})
            # The reproduced first frame must match the archived room event.
            head_event = next(event for event in trace["events"]
                              if event["action"] == "head_depth_shadow_snapshot")
            if head_event["voxel"]["depth_rays_used"] != rows[-24]["depth_rays_used"]:
                raise AssertionError("first-frame ray count differs from V37")
        finally:
            room.close()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "audit_head_unknown_v38_source.py").write_bytes(SOURCE.read_bytes())
    result = {"scope": "Evaluator-only first-frame attribution for V37 moving-room development seeds. Same selected Cove-50 shell and archived actor depth; 8 nearby hypothetical 4 cm routes per seed. Exact segmentation labels only occluder cause, never actor input. No motion authority, no held-out promotion.",
              "source_sha256": sha(SOURCE),
              "archived_first_depth_sha256": {
                  str(seed): sha(INPUT / (f"furnished_moving-seed{seed}-0-"
                                         "chest_rgbd_range_guard_goal_map-none") /
                                 "head-depth-first.npz") for seed in (85, 88)},
              "rows": rows}
    (OUTPUT / "head-unknown-attribution.json").write_text(
        json.dumps(result, indent=2) + "\n")
    for seed in (85, 88):
        for stride in (8, 4, 2):
            subset = [row for row in rows if row["seed"] == seed and
                      row["stride_px"] == stride]
            all_causes = Counter()
            for row in subset:
                all_causes.update(row["unknown_attribution"])
            print(seed, stride,
                  "free", round(float(np.mean([row["coverage"]["free"]
                                               for row in subset])), 1),
                  "unknown", round(float(np.mean([row["coverage"]["unknown"]
                                                  for row in subset])), 1),
                  "fully_free", sum(row["coverage"]["unknown"] == 0 and
                                    row["coverage"]["occupied"] == 0
                                    for row in subset),
                  "causes", dict(all_causes))


if __name__ == "__main__":
    main()
