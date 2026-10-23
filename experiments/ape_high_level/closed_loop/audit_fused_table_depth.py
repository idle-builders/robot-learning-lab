"""Screen a remembered lower-head depth view plus fast low range on table heights.

Depth is captured once at a safe pose and projected with the actor camera pose.
Known geometry scores subsequent hypothetical poses only; no truth labels enter
the candidate obstacle set. This is a sensor-coverage screen, not a rollout.
"""

from __future__ import annotations

from collections import Counter
import json

import numpy as np

from audit_depth_table_visibility import depth_world_points
from audit_footprint_calibration import segmented_gap
from audit_table_height import set_table_height
from footprint_route import scan_points
from run import OUT, Room


def main() -> None:
    room = Room()
    rows = []
    depth_counts = {}
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
                cloud = depth_world_points(room, "lower_rgb", depth)
                # Lower scan guards soles/jaws near the floor; the depth memory
                # supplies surfaces able to meet the upper limbs/torso.
                upper = cloud[(cloud[:, 2] >= .20) & (cloud[:, 2] <= .45) &
                              (np.linalg.norm(cloud[:, :2] - room.actual[:2],
                                              axis=1) > .12) &
                              (np.linalg.norm(cloud[:, :2] - room.actual[:2],
                                              axis=1) < 1.2)]
                depth_counts[f"{seed}-{height}"] = len(upper)
                for x in np.arange(.48, .961, .04):
                    for yy in np.arange(.16, .761, .04):
                        room.actual[:] = (x, side * yy, 0.)
                        room._forward()
                        truth, body, obstacle = room._current_obstacle_clearance()
                        if not obstacle.startswith("loop_table_"):
                            continue
                        low = scan_points(room.range_scan(), room.actual)
                        low_gap = segmented_gap(low, room.actual[:2], tight=True)
                        upper_gap = segmented_gap(upper[:, :2], room.actual[:2])
                        rows.append({"seed": seed, "height_m": height,
                                     "xy_m": [round(float(x), 3),
                                              round(float(side * yy), 3)],
                                     "true_gap_m": round(float(truth), 4),
                                     "closest_body": body,
                                     "low_tight_gap_m": round(low_gap, 4),
                                     "head_depth_upper_gap_m": round(upper_gap, 4),
                                     "fused_gap_m": round(min(low_gap, upper_gap), 4)})
    finally:
        room.close()
    counts = {}
    for height in (.25, .29, .33, .37):
        subset = [r for r in rows if r["height_m"] == height]
        counts[str(height)] = {}
        for proxy in ("low_tight", "fused"):
            count = Counter("false_safe" if r["true_gap_m"] < .05 and
                            r[proxy + "_gap_m"] >= .055 else
                            "false_block" if r["true_gap_m"] >= .05 and
                            r[proxy + "_gap_m"] < .055 else "correct"
                            for r in subset)
            counts[str(height)][proxy] = dict(count)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"scope": "One head RGB-D frame at (0.12,+/-0.24)m plus 11cm 72-ray scan at each hypothetical fixed-yaw pose; exact geometry evaluator-only. No depth updates, pose noise, dynamic high obstacle or live actions.",
              "head_upper_points": depth_counts, "counts": counts, "rows": rows}
    (OUT / "fused-head-depth-table-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
