"""V182: read-only 9 cm actor and full-body screen one grid edge past V181."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from depth_voxel import DepthVoxelMap
from goal_map_route import GoalMapRoute
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sensed_range, sha
from run_moved_props_v134 import MovedPropRoom
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v182"
V176 = ROOT / "artifacts/ape-closed-loop-cove50-v176/heldout-feasible-scene-audit.json"
V177 = ROOT / "artifacts/ape-closed-loop-cove50-v177/heldout-origin-audit.json"
TRACE = ROOT / "artifacts/ape-closed-loop-cove50-v181/scene93/seed93/trace.json"
FRAMES = TRACE.parent
ORIGIN = ROOT / "artifacts/ape-closed-loop-cove50-v177/seed93-light1.00-present"


def main() -> None:
    scene = next(row for row in json.loads(V176.read_text())["rows"]
                 if row["seed"] == 93)
    positions = {name: tuple(scene["positions_xy_m"][name])
                 for name in ("CreamCheese", "Parmesan")}
    entry = next(row for row in json.loads(V177.read_text())["rows"]
                 if row["seed"] == 93 and not row["absent"])
    trace = json.loads(TRACE.read_text())
    last = next(row for row in reversed(trace["events"])
                if row.get("action") == "guarded_approach_step")
    pose = np.asarray(last["estimated_pose_xyyaw"], dtype=float)
    actual = np.asarray(last["evaluator_actual_pose_xyyaw"], dtype=float)
    target = np.asarray(last["actor_target_xy_m"], dtype=float)
    origin_paths = [ORIGIN / f"lower-angle-{row['yaw_command_rad']:+.1f}.npz"
                    for row in entry["rows"]]
    live_paths = sorted(FRAMES.glob("frame-*.npz"),
                        key=lambda path: int(path.stem.split("-")[-1]))
    paths = origin_paths + live_paths
    room = MovedPropRoom(positions)
    mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H)
    try:
        room.reset()
        room.layout(93, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (.40, .46, 0.)
        room.set_table_height(.37)
        fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
        voxel = DepthVoxelMap()
        qpos = None
        stamp = 0.
        hashes = []
        for path in paths:
            with np.load(path) as saved:
                qpos = saved["actor_qpos"].copy()
                depth, _ = mask.apply(saved["depth_m"], qpos)
                stamp = float(saved["capture_time_s"])
                voxel.observe(depth, (saved["camera_origin_m"],
                                      saved["camera_rotation"]), stamp,
                              fovy_deg=fovy, self_masked=True)
            hashes.append(sha(path))
        assert qpos is not None
        room.data.qpos[:] = qpos
        room.actual[:] = actual
        room.estimated[:] = pose
        room.head = float(qpos[room.head_addr])
        room.time_s = stamp
        room._forward()
        robot = RobotGeometry(room.model)
        planner = GoalMapRoute()
        origin_cell = planner.cell(pose[:2])
        obstacles, bodies = scene_geoms(room), None
        bodies = body_geoms(room, obstacles)
        rows = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            cell = (origin_cell[0] + dx, origin_cell[1] + dy)
            future = planner.xy(cell)
            delta = future - pose[:2]
            actor_gaps = []
            for draw in range(5):
                room.actual[:] = actual
                room._forward()
                scans = {height: sensed_range(
                    room, 93, 1820000 + draw * 100 + int(height * 100),
                    height_m=height) for height in (.04, .08, .11)}
                point = TwoHeightClearance(
                    robot, qpos, pose, scans[.11], scans[.04], voxel,
                    stamp, range_8=scans[.08])
                actor_gaps.append(float(point.edge_gap(pose[:2], future, pose[2])))
            evaluator_gaps = []
            for fraction in np.linspace(0., 1., 21):
                room.actual[:] = actual
                room.actual[:2] += .88 * fraction * delta
                room._forward()
                evaluator_gaps.append(full_body_clearance(room, obstacles, bodies)[0])
            actor_target_gap = float(np.linalg.norm(future - target))
            true_target = np.asarray(room.positions["TomatoSauce"])
            actual_endpoint = actual[:2] + .88 * delta
            actual_target_gap = float(np.linalg.norm(actual_endpoint - true_target))
            rows.append({"cell": list(cell), "waypoint_xy_m": future.tolist(),
                         "actor_endpoint_target_gap_m": round(actor_target_gap, 5),
                         "evaluator_endpoint_target_gap_m": round(actual_target_gap, 5),
                         "actor_edge_point_gaps_m": [round(v, 5) for v in actor_gaps],
                         "min_actor_edge_point_gap_m": round(min(actor_gaps), 5),
                         "evaluator_full_body_edge_min_m": round(min(evaluator_gaps), 5),
                         "read_only_candidate_gate": (
                             .27 <= actor_target_gap <= .37 and
                             .27 <= actual_target_gap <= .40 and
                             min(actor_gaps) >= .09 and min(evaluator_gaps) >= .05)})
    finally:
        mask.close()
        room.close()
    result = {"scope": "Read-only V181 seed93 route endpoint. Reconstructed actor self-masked depth history; five deterministic fresh 4/8/11cm scans per adjacent 4cm edge, selected-body 9cm point margin and independent 21-sample full-body evaluator. No motion; target truth only scoring.",
              "source_sha256": sha(SOURCE), "trace_sha256": sha(TRACE),
              "source_frame_sha256": hashes, "actor_pose_xyyaw": pose.tolist(),
              "evaluator_pose_xyyaw": actual.tolist(), "actor_target_xy_m": target.tolist(),
              "rows": rows, "any_read_only_candidate_pass": any(
                  row["read_only_candidate_gate"] for row in rows)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "final-annulus-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        if row["actor_endpoint_target_gap_m"] <= .40:
            print(row["cell"], "actor gap", row["actor_endpoint_target_gap_m"],
                  "point", row["min_actor_edge_point_gap_m"],
                  "body", row["evaluator_full_body_edge_min_m"],
                  "pass", row["read_only_candidate_gate"], flush=True)
    print("any candidate", result["any_read_only_candidate_pass"], flush=True)


if __name__ == "__main__":
    main()
