"""Compare virtual lower-head optical pitch for target, table and body coverage."""

from __future__ import annotations

import json

import numpy as np

from audit_depth_table_visibility import depth_world_points, visible_pixels
from audit_depth_voxel import upper_body_states
from audit_table_height import set_table_height
from depth_voxel import DepthVoxelMap
from run import OUT, Room


CAMERAS = ("lower_rgb", "lower_rgb_p55", "lower_rgb_p70", "lower_rgb_p85")


def main() -> None:
    room = Room()
    rows = []
    try:
        room.layout(85, absent=False, furnished=True)
        for height in (.25, .29, .33, .37):
            set_table_height(room, height)
            for camera in CAMERAS:
                for pose_name, xy in (("start", (0., 0.)),
                                      ("safe_waypoint", (.12, .24))):
                    room.actual[:] = (*xy, 0.)
                    room.estimated[:] = room.actual
                    room.head = 0.
                    room._forward()
                    rgb, depth = room.capture(camera, depth=True)
                    cloud = depth_world_points(room, camera, depth)
                    tabletop = cloud[(cloud[:, 0] > .55) & (cloud[:, 0] < 1.02) &
                                     (cloud[:, 1] > .46) & (cloud[:, 1] < .85) &
                                     (np.abs(cloud[:, 2] - height) < .027)]
                    voxels = DepthVoxelMap()
                    update = voxels.observe(
                        depth, room.camera_pose_estimate(camera), 0.,
                        fovy_deg=float(room.model.cam_fovy[
                            room.model.camera(camera).id]))
                    future = ((0., .04), (0., .08), (.08, .12)) if pose_name == "start" else (
                        (.16, .24), (.20, .32), (.24, .32))
                    probe = {}
                    for point in future:
                        states = upper_body_states(voxels, np.asarray(point), 0.)
                        probe[str(point)] = {status: states.count(status)
                                              for status in ("free", "occupied", "unknown")}
                    rows.append({"height_m": height, "camera": camera,
                                 "pose": pose_name, "xy_m": list(xy),
                                 "target_visible_pixels_evaluator_only":
                                     visible_pixels(room, camera,
                                                    "loop_TomatoSauce_geom"),
                                 "tabletop_visible_pixels_evaluator_only":
                                     visible_pixels(room, camera,
                                                    "loop_table_top"),
                                 "projected_tabletop_points": len(tabletop),
                                 "voxel": update,
                                 "buffered_body_probe": probe})
                    if height == .33 and pose_name == "safe_waypoint":
                        OUT.mkdir(parents=True, exist_ok=True)
                        rgb.save(OUT / f"head-pitch-{camera}-table33.png")
    finally:
        room.close()
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"scope": "Virtual optical-pitch alternatives at the selected lower-head camera position; evaluator segmentation only scores visibility. Depth rays and buffered-volume probe are actor-style. Fixed neutral body, one seed/pose; no live controller or physical mount approval.",
              "rows": rows}
    (OUT / "head-pitch-depth-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    for row in rows:
        if row["height_m"] == .33:
            print(row["camera"], row["pose"],
                  "target px", row["target_visible_pixels_evaluator_only"],
                  "table px", row["tabletop_visible_pixels_evaluator_only"],
                  "table points", row["projected_tabletop_points"],
                  "probe", row["buffered_body_probe"])


if __name__ == "__main__":
    main()
