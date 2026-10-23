"""V169: read-only turn and target-route screen at seed88 V168 endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_origin_route_v152 import score_path, selected_path
from audit_side_look_target_v81 import infer
from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from depth_voxel import DepthVoxelMap
from goal_map_route import GoalMapRoute
from robot_geometry import RobotGeometry
from run import H, ROOT, W, Vision, iou, sensed_range, sha
from run_moved_props_v134 import MovedPropRoom
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v169"
V149 = ROOT / "artifacts/ape-closed-loop-cove50-v149"
V151 = ROOT / "artifacts/ape-closed-loop-cove50-v151"
V168 = ROOT / "artifacts/ape-closed-loop-cove50-v168"
SCENES = ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"
SEED = 88
LIGHT = 1.35
MARGIN = .09


def build_planner(events: list[dict]) -> GoalMapRoute:
    planner = GoalMapRoute()
    last_pose = None
    last_stamp = None
    for event in events:
        if (event["action"] == "guarded_approach_step" and
                last_pose is not None):
            scan = event.get("detail", {}).get("range_11_m")
            if scan is not None:
                planner.observe(scan, last_pose, last_stamp)
        if event.get("estimated_pose_xyyaw") is not None:
            last_pose = np.asarray(event["estimated_pose_xyyaw"], dtype=float)
            last_stamp = float(event["time_s"])
    return planner


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(SCENES.read_text())
    variant = next(v for row in scenes["rows"] if row["seed"] == SEED
                   for v in row["passable_variants"]
                   if v["variant"].endswith("-2"))
    trace_path = V168 / "scene88/seed88/trace.json"
    trace = json.loads(trace_path.read_text())
    target = np.asarray(trace["final_actor_target_xy_m"])
    pose = np.asarray(trace["final_estimated_pose_xyyaw"])
    actual = np.asarray(trace["evaluator_final_actual_xyyaw"])
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
        paths = list(sorted((V151 / f"seed{SEED}-light{LIGHT:.2f}-present")
                            .glob("lower-angle-*.npz")))
        paths += list(sorted(trace_path.parent.glob("frame-*.npz"),
                             key=lambda path: int(path.stem.split("-")[-1])))
        qpos = None
        stamp = None
        for path in paths:
            with np.load(path) as z:
                qpos = z["actor_qpos"].copy()
                stamp = float(z["capture_time_s"])
                filtered, _ = mask.apply(z["depth_m"], qpos)
                voxel.observe(filtered, (z["camera_origin_m"],
                                         z["camera_rotation"]), stamp,
                              fovy_deg=fovy, self_masked=True)
        assert qpos is not None and stamp is not None
        room.data.qpos[:] = qpos
        room.actual[:] = actual
        room.estimated[:] = pose
        room.head = float(qpos[room.head_addr])
        room.time_s = stamp
        room._forward()
        scene = scene_geoms(room)
        body = body_geoms(room, scene)
        robot = RobotGeometry(room.model)
        rows = []
        for delta_yaw in (0., -.3, -.6):
            start_scans = {h: sensed_range(
                room, SEED, 1690000 + int(abs(delta_yaw) * 1000) + int(h * 100),
                height_m=h) for h in (.04, .08, .11)}
            actor_point = TwoHeightClearance(
                robot, qpos, pose, start_scans[.11], start_scans[.04],
                voxel, stamp, range_8=start_scans[.08])
            actor_turn_gap = min(actor_point.edge_gap(
                pose[:2], pose[:2], yaw)
                for yaw in np.linspace(pose[2], pose[2] + delta_yaw, 13))
            evaluator_turn_gap = (float("inf"), "", "")
            for yaw in np.linspace(actual[2], actual[2] + delta_yaw, 25):
                room.actual[:] = actual
                room.actual[2] = yaw
                room._forward()
                gap = full_body_clearance(room, scene, body)
                if gap[0] < evaluator_turn_gap[0]:
                    evaluator_turn_gap = gap
            room.actual[:] = actual
            room.estimated[:] = pose
            room.actual[2] += delta_yaw
            room.estimated[2] += delta_yaw
            room._forward()
            rgb, depth = room.capture("lower_rgb", depth=True)
            camera = room.camera_pose_estimate("lower_rgb")
            image_path = OUT / f"yaw-{delta_yaw:+.1f}.png"
            annotated_path = OUT / f"yaw-{delta_yaw:+.1f}-annotated.png"
            rgb.save(image_path)
            detection = infer(vision, exemplar, rgb, depth, camera, target,
                              None, annotated_path)
            truth = room.truth_box("TomatoSauce", "lower_rgb")
            correct = (detection["accepted_box"] is not None and
                       truth is not None and
                       iou(detection["accepted_box"], truth) >= .25)
            post_scans = {h: sensed_range(
                room, SEED, 1695000 + int(abs(delta_yaw) * 1000) + int(h * 100),
                height_m=h) for h in (.04, .08, .11)}
            post_point = TwoHeightClearance(
                robot, qpos, room.estimated.copy(), post_scans[.11],
                post_scans[.04], voxel, stamp, range_8=post_scans[.08])
            planner = build_planner(trace["events"])
            command = planner.plan(
                post_scans[.11], room.estimated.copy(), target, None,
                stamp, searching=False, edge_clearance=post_point.edge_gap,
                edge_clearance_margin_m=MARGIN, local_all_point_edges=True,
                point_all_graph_edges=True)
            path_cells = selected_path(planner, room.estimated)
            evaluator_route = score_path(room, room.actual.copy(), path_cells)
            rows.append({"delta_body_yaw_rad": delta_yaw,
                         "actor_turn_swept_point_gap_m": round(actor_turn_gap, 4),
                         "evaluator_turn_swept_full_scene_gap_m":
                             round(evaluator_turn_gap[0], 4),
                         "evaluator_limiting_geoms": list(evaluator_turn_gap[1:]),
                         "post_turn_actor_accepted_similarity":
                             detection["accepted_similarity"],
                         "post_turn_evaluator_accepted_correct": correct,
                         "post_turn_actor_annulus_command": command is not None,
                         "post_turn_planner_debug": planner.last_debug,
                         "post_turn_path_cells": path_cells,
                         "post_turn_evaluator_route": evaluator_route,
                         "image_sha256": sha(image_path)})
            room.actual[:] = actual
            room.estimated[:] = pose
            room.head = float(qpos[room.head_addr])
            room._forward()
        result = {"scope": "V169 read-only 0/-0.3/-0.6rad body yaw screens from V168 seed88 information endpoint, using causal actor RGB-D/4/8/11cm scans and reconstructed prior scan-map evidence. Evaluator turn/route clearance and target correctness afterward only. No turn executed or evaluator route fed to actor.",
                  "source_sha256": sha(SOURCE),
                  "v168_trace_sha256": sha(trace_path),
                  "actor_input_sha256": [sha(path) for path in paths],
                  "rows": rows}
        (OUT / "seed88-turn-route-audit.json").write_text(
            json.dumps(result, indent=2) + "\n")
        (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
        for row in rows:
            print(row["delta_body_yaw_rad"],
                  row["actor_turn_swept_point_gap_m"],
                  row["evaluator_turn_swept_full_scene_gap_m"],
                  row["post_turn_actor_annulus_command"],
                  row["post_turn_planner_debug"].get("reason"), flush=True)
    finally:
        mask.close()
        room.close()


if __name__ == "__main__":
    main()
