"""V89: read-only actor RGB-D side choice versus independent route label."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from depth_voxel import DepthVoxelMap
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha
from upper_obstacle_side import choose_side


SOURCE = Path(__file__)
SIDE_CODE = SOURCE.with_name("upper_obstacle_side.py")
V76 = ROOT / "artifacts/ape-closed-loop-cove50-v76"
V80 = ROOT / "artifacts/ape-closed-loop-cove50-v80"
V82 = ROOT / "artifacts/ape-closed-loop-cove50-v82"
V51 = ROOT / "artifacts/ape-closed-loop-cove50-v51/stop-state-attribution.json"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v89"


def replay(seed: int, room: Room, robot: RobotGeometry,
           route: dict) -> dict:
    target_path = V82 / f"seed{seed}-oracle/trace.json"
    target = np.asarray(json.loads(target_path.read_text())[
        "target_estimate_xy_m"], dtype=float)
    voxel = DepthVoxelMap()
    views = []
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    for index in range(3):
        path = V76 / f"seed{seed}-look{index}-actor-input.npz"
        with np.load(path) as z:
            qpos = z["actor_qpos"].copy()
            pose = z["estimated_pose_xyyaw"].copy()
            time_s = float(z["capture_time_s"])
            depth = z["depth_m"].copy()
            camera = (z["camera_origin_m"].copy(),
                      z["camera_rotation"].copy())
        robot.set_state(qpos)
        solids = robot.primitives()
        observed = voxel.observe(depth, camera, time_s, fovy_deg=fovy,
                                 self_hit=lambda hit:
                                 RobotGeometry.solid_distance(hit, solids) < 0.)
        decision = choose_side(voxel, pose, target, time_s)
        views.append({"look_index": index, "actor_input_sha256": sha(path),
                      "capture_time_s": time_s,
                      "depth_rays_used": observed["depth_rays_used"],
                      "decision": decision})
    guarded_initial = []
    for branch in ("left", "right"):
        path = V80 / f"seed{seed}-{branch}/frame-1.npz"
        with np.load(path) as z:
            qpos = z["actor_qpos"].copy()
            start_pose = z["estimated_pose_xyyaw"].copy()
            start_time = float(z["capture_time_s"])
            depth = z["depth_m"].copy()
            camera = (z["camera_origin_m"].copy(),
                      z["camera_rotation"].copy())
        robot.set_state(qpos)
        solids = robot.primitives()
        fresh = DepthVoxelMap()
        fresh.observe(depth, camera, start_time, fovy_deg=fovy,
                      self_hit=lambda hit:
                      RobotGeometry.solid_distance(hit, solids) < 0.)
        guarded_initial.append({"branch": branch,
                                "actor_input_sha256": sha(path),
                                "decision": choose_side(
                                    fresh, start_pose, target, start_time)})
    route_xy = np.asarray(route["evaluator_route"]["path_xy_m"])
    start = pose[:2]
    forward = target - start
    forward /= np.linalg.norm(forward)
    left = np.array([-forward[1], forward[0]])
    route_lateral = float((route_xy[1] - route_xy[0]) @ left)
    reference_side = "left" if route_lateral > 0 else "right"
    selected = views[-1]["decision"]["side"]
    return {"seed": seed, "target_trace_sha256": sha(target_path),
            "actor_target_estimate_xy_m": target.tolist(),
            "actor_stop_pose_xyyaw": pose.tolist(),
            "views": views,
            "guarded_v80_initial_views_posthoc": guarded_initial,
            "independent_reference_side_evaluator_only": reference_side,
            "selected_side_matches_reference": selected == reference_side}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    routes = {int(row["seed"]): row for row in
              json.loads(V51.read_text())["rows"]}
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        rows = [replay(seed, room, robot, routes[seed]) for seed in (85, 88)]
    finally:
        room.close()
    gate = all(row["selected_side_matches_reference"] and
               row["views"][-1]["decision"]["side"] is not None
               for row in rows)
    result = {"scope": "Actor-only high-band voxel side nomination from saved in-place V76 lower-head RGB-D yaw sweep at exact V72 stops; independent V51 simulator path labels choice only after nomination. V80 guarded neutral start frames are a post-hoc replication, not part of the frozen gate. Read-only, no safety authority or new motion.",
              "source_sha256": sha(SOURCE),
              "side_code_sha256": sha(SIDE_CODE),
              "v51_evaluator_audit_sha256": sha(V51),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": rows, "two_scene_side_cue_gate_pass": gate}
    (OUT / "upper-obstacle-side-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "audit_upper_obstacle_side_v89_source.py").write_bytes(
        SOURCE.read_bytes())
    (OUT / "upper_obstacle_side_source.py").write_bytes(
        SIDE_CODE.read_bytes())
    for row in rows:
        print(row["seed"], "neutral", row["views"][0]["decision"],
              "sweep", row["views"][-1]["decision"],
              "reference", row["independent_reference_side_evaluator_only"],
              flush=True)
    print("side cue gate:", gate, flush=True)


if __name__ == "__main__":
    main()
