"""V79: replay candidate depth against the remaining evaluator path."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from audit_active_route_coverage_v77 import route_coverage
from audit_movable_viewpoint_v78 import base_depth_map
from depth_voxel import DepthVoxelMap
from robot_geometry import RobotGeometry
from run import ROOT, Room, sha


SOURCE = Path(__file__)
V77 = ROOT / "artifacts/ape-closed-loop-cove50-v77/active-route-coverage-audit.json"
V78 = ROOT / "artifacts/ape-closed-loop-cove50-v78/movable-viewpoint-potential-audit.json"
V78_OUT = ROOT / "artifacts/ape-closed-loop-cove50-v78"
STOP_RUN = ROOT / "artifacts/ape-closed-loop-cove50-v72-stopstate"
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v79"
CONFIG = "chest_rgbd_range_guard_goal_map"


def replay(seed: int, room: Room, robot: RobotGeometry,
           route_row: dict, candidate_row: dict) -> dict:
    stop_path = STOP_RUN / (f"furnished_table_corridor-seed{seed}-0-"
                            f"{CONFIG}-none/first-stop-actor-state.npz")
    with np.load(stop_path) as stop:
        qpos = stop["actor_qpos_estimated"].copy()
        stop_time = float(stop["capture_time_s"])
    room.reset()
    base, hashes = base_depth_map(seed, room, robot)
    route = np.asarray(route_row["reference_route"]["path_xy_m"], dtype=float)
    actual = np.asarray(route_row["reference_start_actual_xyyaw"])
    pose = np.asarray(route_row["actor_estimated_xyyaw"])
    actor_path = np.column_stack((pose[:2] + route - actual[:2],
                                  np.full(len(route), pose[2])))
    fovy = float(room.model.cam_fovy[room.model.camera("lower_rgb").id])
    rows = []
    for index, candidate in enumerate(candidate_row["nominated_viewpoints"]):
        point = np.asarray(candidate["snapped_xy_m"])
        distances = np.linalg.norm(actor_path[:, :2] - point, axis=1)
        nearest = int(np.argmin(distances))
        downstream_path = actor_path[nearest:]
        baseline = route_coverage(robot, qpos, base, downstream_path,
                                  stop_time + .8)["first_eight"]
        voxel: DepthVoxelMap = copy.deepcopy(base)
        input_hashes = []
        for view_index in range(3):
            frame_path = V78_OUT / (f"seed{seed}-candidate{index}-"
                                    f"view{view_index}.npz")
            input_hashes.append(sha(frame_path))
            with np.load(frame_path) as frame:
                robot.set_state(frame["actor_qpos"])
                solids = robot.primitives()
                time_s = float(frame["capture_time_s"])
                voxel.observe(
                    frame["depth_m"],
                    (frame["camera_origin_m"], frame["camera_rotation"]),
                    time_s, fovy_deg=fovy,
                    self_hit=lambda hit: RobotGeometry.solid_distance(
                        hit, solids) < 0.)
        after = route_coverage(robot, qpos, voxel, downstream_path,
                               time_s)["first_eight"]
        relevant = bool(distances[nearest] <= .12 and
                        candidate["evaluator_endpoint_clearance_m"] >= .05 and
                        len(downstream_path) - 1 >= 4)
        passes = (relevant and after["occupied"] == 0 and
                  after["unknown_fraction"] is not None and
                  after["unknown_fraction"] <= .25 and
                  after["unknown"] < baseline["unknown"])
        rows.append({"candidate_index": index,
                     "side": candidate["side"],
                     "side_radius_m": candidate["side_radius_m"],
                     "snapped_xy_m": candidate["snapped_xy_m"],
                     "nearest_route_step": nearest,
                     "nearest_route_distance_m": round(
                         float(distances[nearest]), 4),
                     "remaining_route_segments": len(downstream_path) - 1,
                     "evaluator_endpoint_clearance_m": candidate[
                         "evaluator_endpoint_clearance_m"],
                     "actor_input_sha256": input_hashes,
                     "in_place_sweep_downstream_first_eight": baseline,
                     "after_candidate_downstream_first_eight": after,
                     "relevant_to_route": relevant,
                     "downstream_observability_pass": passes})
        print(seed, index, "nearest", nearest,
              "distance", round(float(distances[nearest]), 3),
              "baseline", baseline["unknown_fraction"],
              "after", after["unknown_fraction"],
              "occupied", after["occupied"], flush=True)
    return {"seed": seed,
            "stop_state_sha256": sha(stop_path),
            "v76_actor_input_sha256": hashes,
            "candidates": rows,
            "scene_downstream_observability_pass": any(
                row["downstream_observability_pass"] for row in rows)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    routes = {int(row["seed"]): row for row in json.loads(
        V77.read_text())["rows"]}
    candidates = {int(row["seed"]): row for row in json.loads(
        V78.read_text())["rows"]}
    room = Room()
    try:
        robot = RobotGeometry(room.model)
        rows = [replay(seed, room, robot, routes[seed], candidates[seed])
                for seed in (85, 88)]
    finally:
        room.close()
    result = {
        "scope": "Read-only V78 actor-depth replay scored on up to eight remaining V77 evaluator path segments from each counterfactual viewpoint. Evaluator route never enters actor decision or motion; no move executed.",
        "source_sha256": sha(SOURCE),
        "v77_audit_sha256": sha(V77),
        "v78_audit_sha256": sha(V78),
        "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "rows": rows,
        "two_scene_downstream_observability_gate_pass": all(
            row["scene_downstream_observability_pass"] for row in rows)}
    (OUT / "downstream-view-coverage-audit.json").write_text(
        json.dumps(result, indent=2) + "\n")
    (OUT / "audit_downstream_view_v79_source.py").write_bytes(
        SOURCE.read_bytes())
    print("two-scene downstream gate:",
          result["two_scene_downstream_observability_gate_pass"])


if __name__ == "__main__":
    main()
