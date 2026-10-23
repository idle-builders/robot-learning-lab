"""Screen paired shoulder 8x8 ToF views as upper-obstacle complements.

The optical mount is a virtual candidate on the selected Cove-50 torso, not
a packaging decision. Truth geometry only scores hypothetical positions.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_depth_table_visibility import depth_world_points
from audit_footprint_calibration import segmented_gap
from audit_table_height import set_table_height
from footprint_route import scan_points
from run import H, W, ROOT, Room, sha


OUT = ROOT / "artifacts/ape-closed-loop-cove50-v39"
SOURCE = Path(__file__)


def tof_points(room: Room, camera: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Use center returns in 8x8 zones spanning a 45-degree square view."""
    _, depth = room.capture(camera, depth=True)
    dense_upper_bound = depth_world_points(room, camera, depth)
    room.renderer.enable_segmentation_rendering()
    room.renderer.update_scene(room.data, camera=camera)
    seg = room.renderer.render().copy()
    room.renderer.disable_segmentation_rendering()
    center = np.array([int(H * (i + .5) / 8) for i in range(8)])
    # A square crop makes the horizontal and vertical FOV equally close to
    # the 45 x 45 degree detection cone in the candidate device datasheet.
    columns = np.array([int((W - H) / 2 + H * (i + .5) / 8)
                        for i in range(8)])
    rows, cols = np.meshgrid(center, columns, indexing="ij")
    readings = depth[rows, cols]
    valid = np.isfinite(readings) & (readings >= .02) & (readings <= 4.)
    fovy = float(room.model.cam_fovy[room.model.camera(camera).id])
    focal = H / (2 * math.tan(math.radians(fovy) / 2))
    rays = np.stack(((cols - W / 2) / focal,
                     -(rows - H / 2) / focal,
                     -np.ones_like(readings)), axis=-1)
    origin, rotation = room.camera_pose_estimate(camera)
    points = origin + (rays[valid] * readings[valid, None]) @ rotation.T
    table_ids = {room.model.geom(name).id for name in
                 ("loop_table_top", "loop_table_leg_-1_-1",
                  "loop_table_leg_-1_1", "loop_table_leg_1_-1",
                  "loop_table_leg_1_1")
                 if mujoco.mj_name2id(room.model, mujoco.mjtObj.mjOBJ_GEOM,
                                     name) >= 0}
    hit_ids = seg[rows, cols, 0]
    return points, dense_upper_bound, {
        "valid_zones": int(valid.sum()),
        "table_zones_evaluator_only": int(np.sum(
            valid & np.isin(hit_ids, list(table_ids)))),
        "near_zones_lt_12cm": int(np.sum(valid & (readings < .12)))}


def upper_points(points: np.ndarray, sensor_xy: np.ndarray) -> np.ndarray:
    radius = np.linalg.norm(points[:, :2] - sensor_xy, axis=1)
    # Fixed near-body reject approximates a self-mask for this upper-bound
    # screen. A real actor needs joint-aware masking and return confidence.
    return points[(points[:, 2] >= .20) & (points[:, 2] <= .45) &
                  (radius > .14) & (radius < 1.2)]


def main() -> None:
    room = Room(shoulder_tof_audit=True)
    rows = []
    captures = []
    try:
        for seed in (85, 88):
            side = 1 if seed % 2 else -1
            for height in (.25, .29, .33, .37):
                room.reset()
                room.layout(seed, absent=False, furnished=True)
                set_table_height(room, height)
                room.actual[:] = (.12, side * .24, 0.)
                room.estimated[:] = room.actual
                room.head = 0.
                room._forward()
                _, depth = room.capture("lower_rgb", depth=True)
                head = upper_points(depth_world_points(room, "lower_rgb", depth),
                                    room.actual[:2])
                tof_clouds = []
                dense_clouds = []
                for camera in ("shoulder_tof_left", "shoulder_tof_right"):
                    cloud, dense_cloud, metadata = tof_points(room, camera)
                    upper = upper_points(cloud, room.actual[:2])
                    tof_clouds.append(upper)
                    dense_clouds.append(upper_points(dense_cloud,
                                                     room.actual[:2]))
                    captures.append({"seed": seed, "height_m": height,
                                     "camera": camera, **metadata,
                                     "upper_points_after_self_radius": len(upper)})
                tof = np.concatenate(tof_clouds).reshape(-1, 3)
                dense = np.concatenate(dense_clouds).reshape(-1, 3)
                combos = {"low_only": np.zeros((0, 2)),
                          "low_head": head[:, :2],
                          "low_tof": tof[:, :2],
                          "low_head_tof": np.concatenate((head[:, :2],
                                                          tof[:, :2])),
                          "low_shoulder_dense_upper_bound": dense[:, :2],
                          "low_head_shoulder_dense_upper_bound": np.concatenate(
                              (head[:, :2], dense[:, :2]))}
                for x in np.arange(.48, .961, .04):
                    for yy in np.arange(.16, .761, .04):
                        room.actual[:] = (x, side * yy, 0.)
                        room._forward()
                        truth, body, obstacle = room._current_obstacle_clearance()
                        if not obstacle.startswith("loop_table_"):
                            continue
                        low = scan_points(room.range_scan(), room.actual)
                        low_gap = segmented_gap(low, room.actual[:2], tight=True)
                        candidate = {key: round(float(min(
                            low_gap, segmented_gap(points, room.actual[:2]))), 4)
                            if len(points) else round(float(low_gap), 4)
                            for key, points in combos.items()}
                        rows.append({"seed": seed, "height_m": height,
                                     "xy_m": [round(float(x), 3),
                                              round(float(side * yy), 3)],
                                     "true_gap_m": round(float(truth), 4),
                                     "closest_body": body,
                                     "proxy_gap_m": candidate})
    finally:
        room.close()
    counts = {}
    for height in (.25, .29, .33, .37):
        subset = [row for row in rows if row["height_m"] == height]
        counts[str(height)] = {}
        for config in ("low_only", "low_head", "low_tof", "low_head_tof",
                       "low_shoulder_dense_upper_bound",
                       "low_head_shoulder_dense_upper_bound"):
            tally = Counter("false_safe" if row["true_gap_m"] < .05 and
                            row["proxy_gap_m"][config] >= .055 else
                            "false_block" if row["true_gap_m"] >= .05 and
                            row["proxy_gap_m"][config] < .055 else "correct"
                            for row in subset)
            counts[str(height)][config] = dict(tally)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit_near_body_tof_v39_source.py").write_bytes(SOURCE.read_bytes())
    run_source = ROOT / "experiments/ape_high_level/closed_loop/run.py"
    (OUT / "run_source.py").write_bytes(run_source.read_bytes())
    result = {"scope": "Offline sensor geometry screen: virtual paired 8x8 45-degree ToF center returns at torso-relative (+4cm,+/-14cm,+12cm), yaw +/-40 deg, pitch -15 deg, plus selected head RGB-D and current low scan. A dense depth projection from the same shoulder mounts is an optical upper bound, not an 8x8 ToF output. One safe-stance capture per height/seed; hypothetical neutral fixed-yaw table poses. Exact contacts and table labels evaluator-only. ToF point projection and static self-radius are optimistic simplifications; no electronics fit, ranging noise, live step or dynamic high obstacle claim.",
              "source_sha256": sha(SOURCE),
              "run_source_sha256": sha(run_source),
              "selected_body_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "captures": captures, "counts": counts, "rows": rows}
    (OUT / "near-body-tof-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"captures": captures, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
