"""Check whether a ray-cleared head-depth map rejects hidden table space."""

from __future__ import annotations

from collections import Counter
import json

import numpy as np

from audit_footprint_calibration import segmented_gap
from audit_table_height import set_table_height
from depth_voxel import DepthVoxelMap
from footprint_route import scan_points
from run import OUT, Room


def upper_body_states(voxels: DepthVoxelMap, xy: np.ndarray,
                      time_s: float) -> list[str]:
    # Probe a buffered upper-body volume, not only geom centerlines. The
    # shell/arm envelope plus 5 cm safety band extends roughly 20 cm sideways.
    # This intentionally tests whether the view can *clear* that volume.
    samples = ((dx, dy, z)
               for z in (.18, .22, .26, .30, .34, .38)
               for dx in (-.06, 0., .06)
               for dy in (-.20, -.16, -.10, 0., .10, .16, .20))
    return [voxels.state(np.array([xy[0] + dx, xy[1] + dy, z]), time_s)
            for dx, dy, z in samples]


def main() -> None:
    room = Room()
    rows = []
    observations = []
    try:
        room.layout(85, absent=False, furnished=True)
        for height in (.25, .29, .33, .37):
            set_table_height(room, height)
            views = (((.12, .24), angle) for angle in (-.65, 0., .65))
            for mode, requests in (
                    ("one_head_view", (((.12, .24), 0.),)),
                    ("three_head_views", tuple(views)),
                    ("two_poses_three_views",
                     tuple(((.12, .24), angle) for angle in (-.65, 0., .65)) +
                     tuple(((.20, .32), angle) for angle in (-.65, 0., .65)))):
                voxels = DepthVoxelMap()
                for xy, angle in requests:
                    room.actual[:] = (xy[0], xy[1], 0.)
                    room.estimated[:] = room.actual
                    room.head = angle
                    room._forward()
                    _, depth = room.capture("lower_rgb", depth=True)
                    observed = voxels.observe(
                        depth, room.camera_pose_estimate("lower_rgb"), 0.,
                        fovy_deg=float(room.model.cam_fovy[
                            room.model.camera("lower_rgb").id]))
                    observations.append({"height_m": height, "mode": mode,
                                         "sensor_xy_m": list(xy),
                                         "head_yaw_rad": angle, **observed})
                for x in np.arange(.48, .961, .04):
                    for yy in np.arange(.16, .761, .04):
                        room.actual[:] = (x, yy, 0.)
                        room.head = 0.
                        room._forward()
                        truth, body, obstacle = room._current_obstacle_clearance()
                        if not obstacle.startswith("loop_table_"):
                            continue
                        low = scan_points(room.range_scan(), room.actual)
                        low_gap = segmented_gap(low, room.actual[:2], tight=True)
                        states = upper_body_states(voxels, room.actual[:2], 0.)
                        decision = ("safe" if low_gap >= .055 and
                                    all(state == "free" for state in states)
                                    else "stop_or_look")
                        rows.append({"height_m": height, "mode": mode,
                                     "xy_m": [round(float(x), 3),
                                              round(float(yy), 3)],
                                     "true_gap_m": round(float(truth), 4),
                                     "closest_body": body,
                                     "low_tight_gap_m": round(low_gap, 4),
                                     "upper_states": states,
                                     "decision": decision})
    finally:
        room.close()
    counts = {}
    for height in (.25, .29, .33, .37):
        counts[str(height)] = {}
        for mode in ("one_head_view", "three_head_views",
                     "two_poses_three_views"):
            subset = [r for r in rows if r["height_m"] == height and
                      r["mode"] == mode]
            labels = Counter("false_safe" if r["true_gap_m"] < .05 and
                             r["decision"] == "safe" else
                             "false_block_or_unknown" if r["true_gap_m"] >= .05 and
                             r["decision"] != "safe" else "correct"
                             for r in subset)
            counts[str(height)][mode] = {**dict(labels),
                                         "known_safe": sum(r["decision"] == "safe"
                                                           for r in subset),
                                         "tested": len(subset)}
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"scope": "Offline one-/three-view actor depth voxel map from safe pose; fixed-yaw neutral hypothetical table poses, exact collision truth used only to score. Unknown is stop/ask-to-look. No temporal movement, camera pose noise or articulated sweep.",
              "counts": counts, "observations": observations, "rows": rows}
    (OUT / "depth-voxel-table-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
