"""Score selected-body depth views of variable-height tabletops from a safe pose.

The RGB-D frame is actor-style. Known table bounds and segmentation are used
only for evaluator scoring, never supplied to the perception/planner loop.
"""

from __future__ import annotations

import json
import math

import mujoco
import numpy as np

from audit_table_height import set_table_height
from run import OUT, Room


def depth_world_points(room: Room, camera: str, depth: np.ndarray,
                       *, stride: int = 6) -> np.ndarray:
    height, width = depth.shape
    fovy = float(room.model.cam_fovy[room.model.camera(camera).id])
    focal = height / (2 * math.tan(math.radians(fovy) / 2))
    rows, columns = np.mgrid[stride // 2:height:stride,
                             stride // 2:width:stride]
    distance = depth[rows, columns]
    valid = np.isfinite(distance) & (distance > .08) & (distance < 1.8)
    rays = np.stack(((columns - width / 2) / focal,
                     -(rows - height / 2) / focal,
                     -np.ones_like(distance)), axis=-1)
    origin, rotation = room.camera_pose_estimate(camera)
    return origin + (rays[valid] * distance[valid, None]) @ rotation.T


def visible_pixels(room: Room, camera: str, geom_name: str) -> int:
    gid = room.model.geom(geom_name).id
    room.renderer.enable_segmentation_rendering()
    room.renderer.update_scene(room.data, camera=camera)
    mask = room.renderer.render().copy()
    room.renderer.disable_segmentation_rendering()
    return int(np.sum((mask[:, :, 0] == gid) &
                      (mask[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))))


def main() -> None:
    room = Room()
    rows = []
    try:
        room.layout(85, absent=False, furnished=True)
        room.actual[:] = (.12, .24, 0.)
        room.estimated[:] = room.actual
        room.head = 0.
        for height in (.25, .29, .33, .37):
            set_table_height(room, height)
            room._forward()
            true_gap = room._current_obstacle_clearance()[0]
            for camera in ("chest_low50_x45", "chest_low30_x45", "lower_rgb"):
                rgb, depth = room.capture(camera, depth=True)
                points = depth_world_points(room, camera, depth)
                tabletop = points[(points[:, 0] > .55) & (points[:, 0] < 1.02) &
                                  (points[:, 1] > .46) & (points[:, 1] < .85) &
                                  (np.abs(points[:, 2] - height) < .027)]
                pixels = visible_pixels(room, camera, "loop_table_top")
                rows.append({"table_top_height_m": height,
                             "camera": camera,
                             "true_robot_table_clearance_m": round(float(true_gap), 4),
                             "projected_tabletop_points": len(tabletop),
                             "evaluator_visible_top_pixels": pixels,
                             "nearest_tabletop_xy_distance_m": (
                                 round(float(np.linalg.norm(
                                     tabletop[:, :2] - room.actual[:2], axis=1).min()), 4)
                                 if len(tabletop) else None)})
                if height == .33 and camera in ("chest_low50_x45", "lower_rgb"):
                    OUT.mkdir(parents=True, exist_ok=True)
                    rgb.save(OUT / f"table-height33-{camera}.png")
    finally:
        room.close()
    OUT.mkdir(parents=True, exist_ok=True)
    output = {"scope": "Safe fixed-yaw pose (0.12,0.24)m, furnished seed85; selected-body camera views and real depth rendering. Known table bounds/segmentation evaluator-only.",
              "rows": rows}
    (OUT / "table-depth-visibility-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    for row in rows:
        print(row["table_top_height_m"], row["camera"],
              row["projected_tabletop_points"],
              row["evaluator_visible_top_pixels"])


if __name__ == "__main__":
    main()
