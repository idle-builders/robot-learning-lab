"""V289: correct half-height ambiguity with true 2/4/8 cm floor objects."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from audit_c1_plane_mount_v237 import MOUNTS, scan as c1_scan
from audit_lateral_clutter_depth_v285 import project
from audit_lateral_clutter_sensors_v283 import (DIRECTIONS, POSE_Y, actor_gap,
                                                evaluator_gap, virtual_cloud)
from audit_shin_range_v190 import scan as shin_scan
from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from preflight_lateral_clutter_v282 import SEED, make_room
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sha
from self_depth_mask import SelfDepthMask


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v289"
HEIGHTS_M = (.02, .04, .08)  # Full physical height; MuJoCo box half-size is height/2.
MARGIN_M = .065


def set_full_height(room, height_m: float) -> None:
    crate = room.model.geom("loop_crate_geom")
    crate.size[2] = height_m / 2
    room.model.body_pos[room.model.body("loop_crate").id, 2] = height_m / 2
    room._forward()
    actual_height = 2 * float(crate.size[2])
    if not math.isclose(actual_height, height_m, abs_tol=1e-9):
        raise AssertionError("MuJoCo box half-size/full-height confusion")


def head_short_points(room, mask, height_m: float, pose_index: int,
                      direction: int) -> tuple[np.ndarray, dict]:
    room.head = -1.2 if direction < 0 else 1.2
    room._forward()
    rgb, depth = room.capture("lower_rgb", depth=True)
    depth[(depth < .07) | (depth > .5)] = np.nan
    filtered, info = mask.apply(depth, room.data.qpos.copy())
    pose = room.camera_pose_estimate("lower_rgb")
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    cloud = project(filtered, pose, fovy)
    saved = OUT / f"height{int(height_m*100)}cm-pose{pose_index}-side{direction:+d}.npz"
    np.savez_compressed(saved, rgb=np.asarray(rgb), depth_short_m=filtered,
                        camera_origin_m=pose[0], camera_rotation=pose[1],
                        actor_qpos=room.data.qpos.copy(),
                        estimated_pose_xyyaw=room.estimated.copy(),
                        capture_time_s=room.time_s)
    return cloud, {"points": len(cloud), "masked_self_pixels": info[
        "masked_self_pixels"], "frame_sha256": sha(saved)}


def score(rows: list[dict]) -> dict:
    keys = rows[0]["actor_pass"] if rows else ()
    return {key: {"false_safe": sum(row["actor_pass"][key] and
                                    not row["evaluator_safe_5cm"] for row in rows),
                  "false_block": sum(not row["actor_pass"][key] and
                                     row["evaluator_safe_5cm"] for row in rows),
                  "passed": sum(row["actor_pass"][key] for row in rows)}
            for key in keys}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    preflight = []
    observations = []
    for height in HEIGHTS_M:
        room = make_room(SEED, "present")
        robot = RobotGeometry(room.model)
        mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H)
        try:
            set_full_height(room, height)
            obstacles = scene_geoms(room)
            bodies = body_geoms(room, obstacles)
            crate = room.model.geom("loop_crate_geom").id
            target = room.model.geom("loop_TomatoSauce_geom").id
            crate_target_gap = float(mujoco.mj_geomDistance(
                room.model, room.data, crate, target, 10., None))
            starts = []
            for index, y in enumerate(POSE_Y):
                pose = np.array((0., y, 0.))
                room.actual[:] = pose
                room.estimated[:] = pose
                room.head = 0.
                room._forward()
                start_gap = float(full_body_clearance(room, obstacles, bodies)[0])
                starts.append({"pose_y_m": y, "start_clearance_m": round(start_gap, 5),
                               "valid_start_5cm": start_gap >= .05})
                # Precompute evaluator labels before exposing any actor cloud.
                labels = {direction: evaluator_gap(room, obstacles, bodies,
                                                   pose, direction)
                          for direction in DIRECTIONS}
                room.actual[:] = pose
                room._forward()
                virtual = np.concatenate((virtual_cloud(room, .04),
                                           virtual_cloud(room, .11)))
                c1 = {name: c1_scan(room, pose, MOUNTS[name])
                      for name in ("front_30cm", "front_11cm", "front_4cm")}
                shin, shin_counts = shin_scan(room, robot, {
                    "case": f"v289-h{height:.3f}-pose{index}"})
                for direction in DIRECTIONS:
                    room.actual[:] = pose
                    room.estimated[:] = pose
                    room._forward()
                    head, head_info = head_short_points(room, mask,
                                                        height, index, direction)
                    qpos = room.data.qpos.copy()
                    clouds = {"virtual_4_11": virtual,
                              "front_30cm": c1["front_30cm"][0],
                              "front_11cm": c1["front_11cm"][0],
                              "front_4cm": c1["front_4cm"][0],
                              "shin_two": shin["two"], "shin_four": shin["four"],
                              "head_side_short": head,
                              "shin_four_plus_head_short":
                                  np.concatenate((shin["four"], head))}
                    gaps = {name: actor_gap(robot, qpos, pose, cloud, direction)
                            for name, cloud in clouds.items()}
                    closest = labels[direction]
                    rows.append({"height_m": height, "pose_y_m": y,
                                 "direction": direction,
                                 "evaluator_valid_start_5cm": start_gap >= .05,
                                 "evaluator_safe_5cm": closest[0] >= .05,
                                 "evaluator_min_clearance_m": round(closest[0], 5),
                                 "actor_point_gap_m": {name: None if not math.isfinite(gap)
                                                       else round(gap, 5)
                                                       for name, gap in gaps.items()},
                                 "actor_pass": {name: gap >= MARGIN_M
                                                for name, gap in gaps.items()},
                                 "head": head_info})
                observations.append({"height_m": height, "pose_y_m": y,
                                     "c1_crate_hits_evaluator_only": {
                                         name: c1[name][1]["crate_hits"]
                                         for name in c1},
                                     "shin_counts": shin_counts})
            preflight.append({"height_m": height,
                              "box_half_height_m": float(room.model.geom(
                                  "loop_crate_geom").size[2]),
                              "crate_target_gap_m": round(crate_target_gap, 5),
                              "starts": starts})
            print("height_cm", int(height * 100),
                  "valid_starts", sum(x["valid_start_5cm"] for x in starts),
                  flush=True)
        finally:
            mask.close()
            room.close()
    scores = {f"{int(height*100)}cm": score([
        row for row in rows if row["height_m"] == height and
        row["evaluator_valid_start_5cm"]]) for height in HEIGHTS_M}
    report = {"scope": "V289 corrects V282-V288 half-height mislabel: frozen same Direct Frame seed138 room and x/y/footprint, physical floor-box heights 2, 4, 8cm via half-height=height/2 and center z=height/2. Evaluate each pose's full-body start clearance and both 8cm actions before actor sensor exposure; score only >=5cm valid starts. Compare virtual4+11, C1 front30/11/4 with self-first, two/four noisy articulated shin ToF, side-looking 7-50cm self-masked head depth and fusion at same actions. Point-only 6.5cm screen is not free-space proof. Paired RGB/depth frames, counts and evaluator labels retained. Stationary primitive toy, same room and posture, ideal rendered depth, kinematic action and no native stop.",
              "source_sha256": sha(SOURCE), "heights_full_m": HEIGHTS_M,
              "preflight": preflight, "observations": observations,
              "rows": rows, "valid_start_scores": scores}
    (OUT / "true-low-obstacle-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
