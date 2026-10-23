"""V188: stationary noisy range rescan of V187 marginal stopped edges."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from depth_voxel import DepthVoxelMap
from robot_geometry import RobotGeometry
from run import H, ROOT, W, sensed_range, sha
from run_moved_props_v134 import MovedPropRoom
from self_depth_mask import SelfDepthMask
from two_height_clearance import TwoHeightClearance

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v188"
SCENES = ROOT / "artifacts/ape-closed-loop-cove50-v184/second-split-scene-audit.json"
ENTRY = ROOT / "artifacts/ape-closed-loop-cove50-v185/second-split-origin-audit.json"


def score(seed: int, scene: dict, entry: dict) -> dict:
    trace_path = ROOT / f"artifacts/ape-closed-loop-cove50-v187/scene{seed}/seed{seed}/trace.json"
    branch = trace_path.parent
    trace = json.loads(trace_path.read_text())
    stop = trace["events"][-1]
    last = next(e for e in reversed(trace["events"])
                if e.get("action") == "guarded_approach_step")
    assert stop["action"] == "local_3d_route_blocked_stop"
    pose = np.asarray(last["estimated_pose_xyyaw"], dtype=float)
    actual = np.asarray(last["evaluator_actual_pose_xyyaw"], dtype=float)
    waypoint = np.asarray(stop["route_choice"]["next_waypoint_xy_m"])
    delta = waypoint - pose[:2]
    margin = .09 + (.005 if last["detail"]["consecutive_scan_rejections"] else 0.)
    positions = {name: tuple(scene["positions_xy_m"][name])
                 for name in ("CreamCheese", "Parmesan")}
    room = MovedPropRoom(positions)
    mask = SelfDepthMask(room.model, camera="lower_rgb", width=W, height=H)
    try:
        room.reset()
        room.layout(seed, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (
            .40, (1 if seed % 2 else -1) * .46, 0.)
        room.set_table_height(.37)
        fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
        voxel = DepthVoxelMap()
        origin = ROOT / f"artifacts/ape-closed-loop-cove50-v185/seed{seed}-light1.00-present"
        origin_paths = [origin / f"lower-angle-{row['yaw_command_rad']:+.1f}.npz"
                        for row in entry["rows"]]
        live_paths = sorted(branch.glob("frame-*.npz"),
                            key=lambda path: int(path.stem.split("-")[-1]))
        qpos = None
        stamp = 0.
        hashes = []
        for path in origin_paths + live_paths:
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
        actor_gaps = []
        first_source_gaps = None
        for draw in range(20):
            room.actual[:] = actual
            room._forward()
            scans = {height: sensed_range(
                room, seed, 1880000 + seed * 1000 + draw * 100 + int(height * 100),
                height_m=height) for height in (.04, .08, .11)}
            point = TwoHeightClearance(robot, qpos, pose, scans[.11], scans[.04],
                                       voxel, stamp, range_8=scans[.08])
            actor_gaps.append(float(point.edge_gap(pose[:2], waypoint, pose[2])))
            if draw == 0:
                first_source_gaps = {}
                for name, source in point.sources.items():
                    if not len(source):
                        first_source_gaps[name] = None
                        continue
                    gaps = []
                    for fraction in (.25, .5, .75, 1.):
                        future = pose.copy()
                        future[:2] = pose[:2] + fraction * delta
                        robot.set_state(robot.pose_qpos(qpos, future))
                        gaps.append(float(np.min(robot.solid_distances(
                            source, robot.primitives()))))
                    first_source_gaps[name] = round(min(gaps), 5)
        obstacles = scene_geoms(room)
        bodies = body_geoms(room, obstacles)
        evaluator = []
        for fraction in np.linspace(0., 1., 21):
            room.actual[:] = actual
            room.actual[:2] += .88 * fraction * delta
            room._forward()
            evaluator.append(full_body_clearance(room, obstacles, bodies)[0])
    finally:
        mask.close()
        room.close()
    return {"seed": seed, "trace_sha256": sha(trace_path),
            "actor_frame_sha256": hashes,
            "actor_pose_xyyaw": pose.tolist(),
            "evaluator_pose_xyyaw": actual.tolist(),
            "next_waypoint_xy_m": waypoint.tolist(),
            "last_recorded_actor_point_gap_m": stop["actor_point_gap_m"],
            "actor_margin_m": margin,
            "stationary_rescan_point_gaps_m": [round(value, 5) for value in actor_gaps],
            "first_rescan_gap_by_source_m": first_source_gaps,
            "rescan_passes": sum(value >= margin for value in actor_gaps),
            "evaluator_full_body_edge_min_m": round(min(evaluator), 5)}


def main() -> None:
    scenes = {row["seed"]: row for row in json.loads(SCENES.read_text())["rows"]}
    entries = {row["seed"]: row for row in json.loads(ENTRY.read_text())["rows"]
               if not row["absent"]}
    rows = [score(seed, scenes[seed], entries[seed]) for seed in (96, 97)]
    result = {"scope": "Read-only 20 deterministic stationary noisy/dropout 4/8/11cm rescans at frozen V187 seeds96/97 marginal local-3D stops, reconstructed actor depth history and current selected-body point-edge margin; independent full-body exact edge score. No motion or threshold change.",
              "source_sha256": sha(SOURCE), "v184_sha256": sha(SCENES),
              "v185_sha256": sha(ENTRY), "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "marginal-edge-rescan-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], "pass", row["rescan_passes"], "/20",
              "range", min(row["stationary_rescan_point_gaps_m"]),
              max(row["stationary_rescan_point_gaps_m"]),
              "body", row["evaluator_full_body_edge_min_m"], flush=True)


if __name__ == "__main__":
    main()
