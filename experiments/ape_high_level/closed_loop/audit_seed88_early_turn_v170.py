"""V170: read-only earlier yaw opportunities in seed88's information path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_side_look_target_v81 import infer
from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from depth_voxel import DepthVoxelMap
from footprint_route import observed_free
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Vision, iou, sensed_range, sha
from run_moved_props_v134 import MovedPropRoom
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v170"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149"
V151 = ROOT / "artifacts/ape-closed-loop-cove50-v151"
V168 = ROOT / "artifacts/ape-closed-loop-cove50-v168"
SCENES = ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"
SEED = 88
LIGHT = 1.35
TURN_RAD = -.6
MARGIN = .09
TIMES = {10.0, 12.4, 13.2, 14.0, 14.8}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(SCENES.read_text())
    variant = next(v for row in scenes["rows"] if row["seed"] == SEED
                   for v in row["passable_variants"]
                   if v["variant"].endswith("-2"))
    trace_path = V168 / f"scene{SEED}/seed{SEED}/trace.json"
    trace = json.loads(trace_path.read_text())
    positions = {"CreamCheese": tuple(variant["cream_cheese_xy_m"]),
                 "Parmesan": tuple(variant["parmesan_xy_m"])}
    room = MovedPropRoom(positions)
    mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H)
    vision = Vision("generic")
    exemplar = np.load(V149 / "owner-taught-exemplar.npy")
    try:
        room.reset()
        room.layout(SEED, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (
            .40, (1 if SEED % 2 else -1) * .46, 0.)
        room.set_table_height(.37)
        room.model.light_diffuse[0] *= LIGHT
        fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
        voxel = DepthVoxelMap()
        source_paths = list(sorted((V151 / f"seed{SEED}-light{LIGHT:.2f}-present")
                                   .glob("lower-angle-*.npz")))
        frame_events = [e for e in trace["events"]
                        if e.get("estimated_pose_xyyaw") is not None and
                        e.get("actor_input_sha256") is not None]
        for path in source_paths:
            with np.load(path) as z:
                filtered, _ = mask.apply(z["depth_m"], z["actor_qpos"])
                voxel.observe(filtered, (z["camera_origin_m"],
                                         z["camera_rotation"]),
                              float(z["capture_time_s"]), fovy_deg=fovy,
                              self_masked=True)
        scene = scene_geoms(room)
        body = body_geoms(room, scene)
        robot = RobotGeometry(room.model)
        rows = []
        for index, event in enumerate(frame_events, 1):
            path = trace_path.parent / f"frame-{index}.npz"
            with np.load(path) as z:
                qpos = z["actor_qpos"].copy()
                pose = z["estimated_pose_xyyaw"].copy()
                stamp = float(z["capture_time_s"])
                filtered, _ = mask.apply(z["depth_m"], qpos)
                voxel.observe(filtered, (z["camera_origin_m"],
                                         z["camera_rotation"]), stamp,
                              fovy_deg=fovy, self_masked=True)
            if event["action"] != "guarded_approach_step" or round(stamp, 1) not in TIMES:
                continue
            actual = np.asarray(event["evaluator_actual_pose_xyyaw"], dtype=float)
            room.data.qpos[:] = qpos
            room.actual[:] = actual
            room.estimated[:] = pose
            room.head = float(qpos[room.head_addr])
            room.time_s = stamp
            room._forward()
            scans = {h: sensed_range(room, SEED, 1620000 + index * 10 + int(h * 100),
                                     height_m=h) for h in (.04, .08, .11)}
            point = TwoHeightClearance(robot, qpos, pose, scans[.11], scans[.04],
                                       voxel, stamp, range_8=scans[.08])
            actor_turn_gap = min(point.edge_gap(pose[:2], pose[:2], yaw)
                                 for yaw in np.linspace(pose[2],
                                                        pose[2] + TURN_RAD, 13))
            evaluator_swept = (float("inf"), "", "")
            for yaw in np.linspace(actual[2], actual[2] + TURN_RAD, 25):
                room.actual[:] = actual
                room.actual[2] = yaw
                room._forward()
                gap = full_body_clearance(room, scene, body)
                if gap[0] < evaluator_swept[0]:
                    evaluator_swept = gap
            room.actual[:] = actual
            room.estimated[:] = pose
            room.actual[2] += TURN_RAD
            room.estimated[2] += TURN_RAD
            room._forward()
            rgb, depth = room.capture("lower_rgb", depth=True)
            camera = room.camera_pose_estimate("lower_rgb")
            target = np.asarray(event["actor_target_xy_m"])
            image_path = OUT / f"turn-after-frame-{index}.png"
            annotated_path = OUT / f"turn-after-frame-{index}-annotated.png"
            rgb.save(image_path)
            detection = infer(vision, exemplar, rgb, depth, camera, target,
                              None, annotated_path)
            truth = room.truth_box("TomatoSauce", "lower_rgb")
            accepted_correct = (detection["accepted_box"] is not None and
                                truth is not None and
                                iou(detection["accepted_box"], truth) >= .25)
            post_scans = {h: sensed_range(room, SEED, 1625000 + index * 10 + int(h * 100),
                                          height_m=h) for h in (.04, .08, .11)}
            post_point = TwoHeightClearance(robot, qpos, room.estimated,
                                            post_scans[.11], post_scans[.04],
                                            voxel, stamp, range_8=post_scans[.08])
            safe_neighbors = []
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                end = room.estimated[:2] + .04 * np.asarray([dx, dy])
                if (observed_free(end, room.estimated, post_scans[.11]) and
                        observed_free(end, room.estimated, post_scans[.04])):
                    gap = post_point.edge_gap(room.estimated[:2], end,
                                              room.estimated[2])
                    if gap >= MARGIN:
                        safe_neighbors.append({"delta_cell": [dx, dy],
                                               "actor_point_gap_m": round(gap, 4)})
            rows.append({"frame_index": index, "time_s": stamp,
                         "actor_pose_xyyaw": pose.tolist(),
                         "evaluator_actual_xyyaw": actual.tolist(),
                         "actor_target_xy_m": target.tolist(),
                         "actor_turn_swept_point_gap_m": round(actor_turn_gap, 4),
                         "evaluator_turn_swept_full_scene_gap_m":
                             round(evaluator_swept[0], 4),
                         "evaluator_limiting_geoms": list(evaluator_swept[1:]),
                         "post_turn_actor_safe_neighbor_count": len(safe_neighbors),
                         "post_turn_actor_safe_neighbors": safe_neighbors,
                         "post_turn_accepted_similarity": detection["accepted_similarity"],
                         "post_turn_evaluator_accepted_correct": accepted_correct,
                         "post_turn_evaluator_truth_box": truth,
                         "source_frame_sha256": sha(path),
                         "image_sha256": sha(image_path)})
        result = {"scope": "V170 read-only earlier -0.6rad turn screen from five V168 seed88 information-path sensor frames. Actor point sweep and fresh 4/8/11cm post-turn safe neighbors; evaluator full-scene sweep and target correctness afterward. No turn executed or evaluator-selected route fed to actor.",
                  "source_sha256": sha(SOURCE),
                  "v168_trace_sha256": sha(trace_path),
                  "rows": rows}
        (OUT / "turn-timing-audit.json").write_text(json.dumps(result, indent=2) + "\n")
        (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
        for row in rows:
            print(row["time_s"], row["actor_turn_swept_point_gap_m"],
                  row["evaluator_turn_swept_full_scene_gap_m"],
                  row["post_turn_actor_safe_neighbor_count"],
                  row["post_turn_accepted_similarity"], flush=True)
    finally:
        mask.close()
        room.close()


if __name__ == "__main__":
    main()
