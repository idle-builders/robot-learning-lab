"""V292: diagnostic head-depth views on V291's four-shin error cases."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from audit_lateral_clutter_depth_v285 import project
from audit_shin_range_v190 import scan as shin_scan
from audit_varied_low_clutter_v291 import CASES, motion, point_gap
from direct_structural_rooms_v270 import DirectStructuralRoom
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v292"
V291 = ROOT / "artifacts/ape-closed-loop-cove50-v291/varied-low-clutter-audit.json"
CASES_TO_INSPECT = ((139, 1), (140, 2), (147, 1))
MARGIN_M = .065


def main() -> None:
    prior = json.loads(V291.read_text())
    specs = {case[0]: case for case in CASES}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed, pose_index in CASES_TO_INSPECT:
        _, kind, crate_x, crate_y, size_x, size_y, height, yaw = specs[seed]
        room = DirectStructuralRoom(kind=kind, mode="absent",
                                    side=1 if seed % 2 else -1)
        robot = RobotGeometry(room.model)
        mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H)
        try:
            room.layout(seed, absent=True, furnished=True)
            crate = room.model.geom("loop_crate_geom")
            crate.size[:] = (size_x / 2, size_y / 2, height / 2)
            room.model.body_pos[room.model.body("loop_crate").id] = (
                crate_x, crate_y, height / 2)
            hazard = -1 if crate_y < 0 else 1
            pose = np.array((0., 0., yaw))
            pose[:2] += pose_index * .88 * motion(pose, hazard)
            room.actual[:] = pose
            room.estimated[:] = pose
            room._forward()
            for direction in (-1, 1):
                room.head = -1.2 if direction < 0 else 1.2
                room._forward()
                qpos = room.data.qpos.copy()
                rgb, depth = room.capture("lower_rgb", depth=True)
                depth[(depth < .07) | (depth > .5)] = np.nan
                filtered, info = mask.apply(depth, qpos)
                camera = room.camera_pose_estimate("lower_rgb")
                fovy = float(room.model.cam_fovy[
                    room.model.camera("lower_rgb").id])
                head = project(filtered, camera, fovy)
                shin, shin_counts = shin_scan(room, robot, {
                    "case": f"v292-seed{seed}-pose{pose_index}-dir{direction}"})
                gap_head = point_gap(robot, qpos, pose,
                                     motion(pose, direction), head)
                gap_shin = point_gap(robot, qpos, pose,
                                     motion(pose, direction), shin["four"])
                baseline = next(row for row in prior["rows"]
                                if row["seed"] == seed and
                                row["pose_index"] == pose_index and
                                row["direction"] == direction)
                saved = OUT / f"seed{seed}-pose{pose_index}-dir{direction:+d}.npz"
                np.savez_compressed(saved, rgb=np.asarray(rgb),
                                    depth_short_m=filtered,
                                    camera_origin_m=camera[0],
                                    camera_rotation=camera[1],
                                    actor_qpos=qpos,
                                    estimated_pose_xyyaw=pose)
                rows.append({"seed": seed, "kind": kind,
                             "full_height_m": height, "pose_index": pose_index,
                             "direction": direction,
                             "evaluator_valid_start_5cm": baseline[
                                 "evaluator_valid_start_5cm"],
                             "evaluator_safe_5cm": baseline["evaluator_safe_5cm"],
                             "evaluator_min_clearance_m": baseline[
                                 "evaluator_min_clearance_m"],
                             "v291_four_shin_point_gap_m": baseline[
                                 "actor_point_gap_m"]["shin_four"],
                             "head_short_point_gap_m": None if not math.isfinite(gap_head)
                                                       else round(gap_head, 5),
                             "fresh_four_shin_point_gap_m": None if not math.isfinite(gap_shin)
                                                          else round(gap_shin, 5),
                             "head_short_points": len(head),
                             "shin_points": len(shin["four"]),
                             "shin_self_first": shin_counts.get("two_self_first", 0) +
                                                shin_counts.get("four_self_first", 0),
                             "masked_self_pixels": info["masked_self_pixels"],
                             "head_short_point_pass": gap_head >= MARGIN_M,
                             "shin_plus_head_point_pass": min(gap_head, gap_shin) >= MARGIN_M,
                             "frame_sha256": sha(saved)})
                print(seed, pose_index, direction, "safe", baseline[
                    "evaluator_safe_5cm"], "head gap", round(gap_head, 3),
                    "points", len(head), flush=True)
        finally:
            mask.close()
            room.close()
    report = {"scope": "V292 post-hoc diagnostic on the three V291 four-shin error poses, with both lateral directions per pose. Recreate same selected Direct Frame open/raised room geometry and pose; capture one side-looking yaw±1.2 lower-head 7-50cm ideal rendered depth frame, model-predicted self mask and raw stride8 low obstacle points. Compare frozen 6.5cm point gap for head alone and four shin+head; evaluator labels read only after actor gaps. Targeted diagnosis cannot estimate held-out accuracy, head return gaps elsewhere or deployable camera fit; missing points are unknown, not free.",
              "source_sha256": sha(SOURCE), "v291_sha256": sha(V291),
              "inspected_cases": CASES_TO_INSPECT, "rows": rows}
    (OUT / "varied-head-depth-diagnostic.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
